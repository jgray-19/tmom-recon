"""Kicker reconstruction against an analytically generated single kick.

The lattice is the simplest ring that still exercises every part of the
reconstruction: ``beta = 1``, ``alpha = 0`` everywhere, so the kicker-to-BPM
transport is the pure rotation ``x_i = sin(2 pi d_mu_i) * dpx``.  The kicker
sits half-way round, so the kick is solved from BPM5, the first BPM downstream,
on the turn the kicker fired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tfs

from tmom_recon.kicker.core import find_kick, first_reading, reconstruct_kick
from tmom_recon.orbit_reference import build_orbit_reference

pytestmark = pytest.mark.unit

KICKER = "KICKER"
TUNES = (0.28, 0.31)
CIRCUMFERENCE = 9.0
KICKER_S = 4.5
BPM_S = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
BPMS = tuple(f"BPM{i}" for i in range(1, len(BPM_S) + 1))
N_FREE = 10
DELTA_PX = 1e-5
DELTA_PY = 4e-6


@pytest.fixture()
def twiss() -> tfs.TfsDataFrame:
    """Flat uncoupled ring: unit beta, phase advancing linearly with ``s``."""
    positions = np.array([KICKER_S, *BPM_S])
    return tfs.TfsDataFrame(
        {
            "s": positions,
            "beta11": 1.0,
            "beta12": 0.0,
            "beta21": 0.0,
            "beta22": 1.0,
            "mu1": TUNES[0] * positions / CIRCUMFERENCE,
            "mu2": TUNES[1] * positions / CIRCUMFERENCE,
            "ev1": 1.0,
            "ev2": 1.0,
        },
        index=pd.Index([KICKER, *BPMS], name="name"),
        headers={"q1": TUNES[0], "q2": TUNES[1]},
    )


@pytest.fixture()
def reference(twiss: tfs.TfsDataFrame):
    """A zero closed orbit at every BPM."""
    zero_model = pd.DataFrame(0.0, index=twiss.index, columns=["x", "px", "y", "py"])
    return build_orbit_reference(
        zero_model.loc[list(BPMS), ["x", "y"]], "dynamic", zero_model, BPMS
    )


def _phase_advance(twiss: tfs.TfsDataFrame, bpm: str, plane: int) -> float:
    """Phase advance in tune units from the kicker to *bpm*, wrapped into one turn."""
    mu, tune = (("mu1", TUNES[0]), ("mu2", TUNES[1]))[plane]
    return float((twiss.at[bpm, mu] - twiss.at[KICKER, mu]) % tune)


def make_data(
    twiss: tfs.TfsDataFrame,
    *,
    delta_px: float = DELTA_PX,
    delta_py: float = DELTA_PY,
    n_turns: int = N_FREE + 3,
    noise: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Turn-by-turn data for a single kick fired at turn :data:`N_FREE`.

    With unit beta and zero alpha the response to the kick is simply
    ``z = dp * sin(2 pi (d_mu + n Q))``, where *n* counts the full turns the beam
    has made since it passed the kicker.  Later turns are exact too, so the
    reconstruction has no model error to absorb.
    """
    rows = []
    for turn in range(n_turns):
        for bpm in BPMS:
            # Full turns completed since the kick; negative before the kicker fired.
            passes = turn - N_FREE - (0 if twiss.at[bpm, "s"] > KICKER_S else 1)
            kicked = passes >= 0
            rows.append(
                {
                    "name": bpm,
                    "turn": turn,
                    "x": kicked * delta_px * _sin(_phase_advance(twiss, bpm, 0), passes, 0),
                    "y": kicked * delta_py * _sin(_phase_advance(twiss, bpm, 1), passes, 1),
                }
            )
    data = pd.DataFrame(rows)
    if noise:
        rng = np.random.default_rng(seed)
        data[["x", "y"]] += rng.normal(0.0, noise, size=(len(data), 2))
        data["var_x"] = noise**2
        data["var_y"] = noise**2
    return data


def _sin(phase_advance: float, passes: int, plane: int) -> float:
    return float(np.sin(2.0 * np.pi * (phase_advance + max(passes, 0) * TUNES[plane])))


def test_find_kick_reports_the_turn_the_kicker_fired(twiss) -> None:
    data = make_data(twiss)
    _bpm, kick_turn = find_kick(data, n_turns_free=N_FREE)
    assert kick_turn == N_FREE


def test_find_kick_requires_a_kick(twiss) -> None:
    data = make_data(twiss, delta_px=0.0, delta_py=0.0)
    with pytest.raises(ValueError, match="No kick found"):
        find_kick(data, n_turns_free=N_FREE)


def test_clean_kick_is_recovered_exactly(twiss, reference) -> None:
    """Noise-free data determines the kick exactly, up to floating point."""
    result = reconstruct_kick(
        make_data(twiss), reference=reference, twiss=twiss, kicker=KICKER, n_turns_free=N_FREE
    )
    assert len(result) == 1
    row = result.iloc[0]
    assert row["name"] == KICKER
    assert row["turn"] == N_FREE
    assert row["x"] == 0.0 and row["y"] == 0.0
    assert row["px"] == pytest.approx(DELTA_PX, rel=1e-12)
    assert row["py"] == pytest.approx(DELTA_PY, rel=1e-12)
    assert result.attrs["kick"].bpm == "BPM5"


def test_only_the_first_bpm_after_the_kicker_is_used(twiss, reference) -> None:
    """Corrupting every other BPM leaves the kick untouched."""
    data = make_data(twiss)
    data.loc[data["name"] != "BPM5", ["x", "y"]] += 1e-3
    result = reconstruct_kick(
        data, reference=reference, twiss=twiss, kicker=KICKER, n_turns_free=N_FREE
    )
    assert result.iloc[0]["px"] == pytest.approx(DELTA_PX, rel=1e-12)
    assert result.iloc[0]["py"] == pytest.approx(DELTA_PY, rel=1e-12)


def test_the_first_bpm_wraps_round_the_ring(twiss) -> None:
    """A kicker after the last BPM is read first by BPM1 on the next turn."""
    twiss = twiss.copy()
    twiss.loc[KICKER, "s"] = 8.5
    rows = pd.DataFrame({"name": list(BPMS), "x": 0.0, "y": 0.0})
    assert first_reading(rows, twiss, KICKER)["name"].tolist() == ["BPM1"]


def test_the_variance_is_the_single_bpm_variance(twiss, reference) -> None:
    """With ``beta = 1`` one reading of variance ``sigma^2`` gives a kick of
    variance ``sigma^2 / sin^2`` of the phase advance to that BPM."""
    noise = 1e-7
    result = reconstruct_kick(
        make_data(twiss, noise=noise),
        reference=reference,
        twiss=twiss,
        kicker=KICKER,
        n_turns_free=N_FREE,
    )
    kick = result.attrs["kick"]
    for plane, variance in ((0, kick.var_px), (1, kick.var_py)):
        response = _sin(_phase_advance(twiss, "BPM5", plane), 0, plane)
        assert np.sqrt(variance) == pytest.approx(noise / abs(response), rel=1e-12)


def test_the_closed_orbit_at_zero_is_removed_before_solving(twiss, reference) -> None:
    """A static orbit offset present in both the data and the frame cancels."""
    offset = 3e-4
    data = make_data(twiss)
    shifted = data.copy()
    shifted[["x", "y"]] += offset
    orbit_zero = pd.DataFrame(offset, index=pd.Index(BPMS, name="name"), columns=["x", "y"])
    zero_model = pd.DataFrame(0.0, index=twiss.index, columns=["x", "px", "y", "py"])
    shifted_reference = build_orbit_reference(orbit_zero, "dynamic", zero_model, BPMS)

    baseline = reconstruct_kick(
        data, reference=reference, twiss=twiss, kicker=KICKER, n_turns_free=N_FREE
    )
    result = reconstruct_kick(
        shifted,
        reference=shifted_reference,
        twiss=twiss,
        kicker=KICKER,
        n_turns_free=N_FREE,
    )
    assert result.iloc[0]["px"] == pytest.approx(baseline.iloc[0]["px"], rel=1e-12)
    assert result.iloc[0]["py"] == pytest.approx(baseline.iloc[0]["py"], rel=1e-12)


def test_the_closed_orbit_is_removed_and_reported_at_the_kicker(twiss, reference) -> None:
    """Off momentum the beam is not at the frame's origin before the kick.

    The frame removes a measured *setting-zero* orbit, taken on momentum, so an
    off-momentum beam still rides its dispersive closed orbit -- in angle as
    well as position.  That orbit must come off the BPM readings before the
    kick is solved for, leaving the kick itself untouched, and go back onto the
    reported kicker state.
    """
    closed_orbit = pd.DataFrame(
        {
            "x": np.linspace(-2e-3, 2e-3, len(twiss)),
            "px": np.linspace(1e-4, -1e-4, len(twiss)),
            "y": np.linspace(3e-4, -3e-4, len(twiss)),
            "py": np.linspace(-5e-5, 5e-5, len(twiss)),
        },
        index=twiss.index,
    )
    data = make_data(twiss)
    data[["x", "y"]] += closed_orbit.loc[data["name"], ["x", "y"]].to_numpy()

    result = reconstruct_kick(
        data,
        reference=reference,
        twiss=twiss,
        kicker=KICKER,
        closed_orbit=closed_orbit,
        n_turns_free=N_FREE,
    )

    row = result.iloc[0]
    assert result.attrs["kick"].delta_px == pytest.approx(DELTA_PX, rel=1e-12)
    assert result.attrs["kick"].delta_py == pytest.approx(DELTA_PY, rel=1e-12)
    at_kicker = closed_orbit.loc[KICKER]
    assert row["x"] == pytest.approx(at_kicker["x"])
    assert row["y"] == pytest.approx(at_kicker["y"])
    assert row["px"] == pytest.approx(at_kicker["px"] + DELTA_PX)
    assert row["py"] == pytest.approx(at_kicker["py"] + DELTA_PY)


def test_the_kick_does_not_depend_on_the_model_orbit(twiss, reference) -> None:
    """The beam's own pre-kick turns fix the orbit; a wrong model orbit moves
    only the reported position, never the kick."""
    wrong = pd.DataFrame(
        np.random.default_rng(3).normal(0.0, 1e-4, size=(len(twiss), 4)),
        index=twiss.index,
        columns=["x", "px", "y", "py"],
    )
    result = reconstruct_kick(
        make_data(twiss),
        reference=reference,
        twiss=twiss,
        kicker=KICKER,
        closed_orbit=wrong,
        n_turns_free=N_FREE,
    )
    assert result.attrs["kick"].delta_px == pytest.approx(DELTA_PX, rel=1e-12)
    assert result.attrs["kick"].delta_py == pytest.approx(DELTA_PY, rel=1e-12)
