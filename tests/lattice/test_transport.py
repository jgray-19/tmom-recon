"""Tests for tmom_recon.lattice.transport."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymadng_utils.accelerators import PSB
from pymadng_utils.mad import AcceleratorMadInterface

from tmom_recon.lattice.transport import kick_response_from_twiss

#: PSB ring-3 skew quadrupole knobs for the coupled test: about 6% beta mixing
#: between the planes at the BPMs. Ten times this makes MAD-NG's coupled twiss fail.
SKEW_KNOBS = {"kbr3qsk210l3": 0.003, "kbr3qskh0": -0.003}


def _uncoupled(beta0: float, beta1: float, delta_mu: float) -> pd.DataFrame:
    """A two-element uncoupled Twiss with the same optics in both planes."""
    return pd.DataFrame(
        {
            "beta11": [beta0, beta1],
            "beta12": 0.0,
            "beta21": 0.0,
            "beta22": [beta0, beta1],
            "mu1": [0.0, delta_mu],
            "mu2": [0.0, delta_mu],
            "ev1": 1.0,
            "ev2": 1.0,
        },
        index=pd.Index(["kicker", "BPM1"], name="name"),
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("beta0", "beta1", "delta_mu"), [(1.0, 1.0, 0.25), (2.5, 1.8, 0.13), (10.0, 4.0, 0.37)]
)
def test_uncoupled_response_is_courant_snyder_r12(beta0, beta1, delta_mu) -> None:
    response = kick_response_from_twiss(
        _uncoupled(beta0, beta1, delta_mu), source="kicker", target="BPM1"
    )
    r12 = np.sqrt(beta0 * beta1) * np.sin(2.0 * np.pi * delta_mu)
    assert response == pytest.approx(np.diag([r12, r12]), abs=1e-14)


@pytest.mark.unit
def test_upstream_target_wraps_by_the_tune() -> None:
    """Reversing a quarter-wave pair with tune 1 leaves a forward advance of 0.75."""
    response = kick_response_from_twiss(
        _uncoupled(1.0, 1.0, 0.25), source="BPM1", target="kicker", tunes=(1.0, 1.0)
    )
    assert response == pytest.approx(-np.eye(2), abs=1e-12)


@pytest.mark.unit
def test_upstream_target_without_tune_raises() -> None:
    """A line has no wrap, so an upstream target is an error, not a sign flip."""
    with pytest.raises(ValueError, match="upstream"):
        kick_response_from_twiss(_uncoupled(1.0, 1.0, 0.25), source="BPM1", target="kicker")


_MAPS_LUA = """
local track, damap, matrix in MAD
local mtbl = track{sequence=loaded_sequence, X0=damap{nv=6, mo=1}, observe=0, savemap=true, method=6}
py:send(#mtbl)
for i = 1, #mtbl do
  local m, M = mtbl.__map[i], matrix(6, 6)
  for k = 1, 6 do for j = 1, 6 do M:set(k, j, m[k]:get(j + 1)) end end
  py:send(M)
end
"""


@pytest.fixture(scope="module")
def coupled_psb(seq_psb3):
    """Coupled PSB ring-3 Twiss and MAD-NG's exact linear map from the start to every element."""
    interface = AcceleratorMadInterface(
        PSB(sequence_file=str(seq_psb3), ring=3, kinetic_energy=0.16)
    )
    try:
        interface.set_madx_variables(**SKEW_KNOBS)
        twiss = interface.run_twiss(observe=0, coupling=True)
        interface.mad.send(_MAPS_LUA)
        maps = np.array([np.asarray(interface.mad.recv()) for _ in range(interface.mad.recv())])
    finally:
        interface.close()
    twiss.index = twiss.index.astype(str).str.upper()
    return twiss, maps


@pytest.mark.integration
@pytest.mark.psb
def test_coupled_response_matches_the_mad_ng_transfer_map(coupled_psb) -> None:
    """Every BPM's response to a kick matches MAD-NG's own map, cross-plane terms included.

    Sources are elements half-way between BPMs around the ring, so both the
    direct and the tune-wrapped (upstream) BPMs are checked.
    """
    twiss, maps = coupled_psb
    tunes = (float(twiss.headers["q1"]), float(twiss.headers["q2"]))
    names = list(twiss.index)
    s = twiss["s"].to_numpy(float)
    bpms = [i for i, name in enumerate(names) if ".BPM" in name]
    cross = direct = 0.0
    sources = [
        int(np.argmin(abs(s - (s[bpms[k]] + s[bpms[k + 1]]) / 2)))
        for k in (0, len(bpms) // 2, len(bpms) - 2)
    ]
    for source in sources:
        for target in bpms:
            one_pass = maps[target] if target > source else maps[target] @ maps[-1]
            exact = (one_pass @ np.linalg.inv(maps[source]))[np.ix_([0, 2], [1, 3])]
            response = kick_response_from_twiss(
                twiss, source=names[source], target=names[target], tunes=tunes
            )
            assert response == pytest.approx(exact, abs=1e-10 * np.abs(exact).max())
            cross = max(cross, np.abs(exact[[0, 1], [1, 0]]).max())
            direct = max(direct, np.abs(exact[[0, 1], [0, 1]]).max())
    # The skew knobs must couple the planes, or this test proves nothing about coupling.
    assert cross > 0.05 * direct
