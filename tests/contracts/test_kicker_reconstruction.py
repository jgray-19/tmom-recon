"""PSB single-kick reconstruction contract.

Every number asserted here has an independent source: the kick is the strength
the exciter was configured with, and the closed orbit the kicker state is
reported on is xsuite's own off-momentum Twiss.  Neither comes from the MAD-NG
model the reconstruction uses, so the cell measures the whole chain.

The kicker is a **thin** corrector on purpose.  MAD-NG puts a Twiss row at each
element's *exit*, while a kick from a thick element is applied at its centre, so
a thick kicker would be reconstructed from optics half its length downstream of
where the beam was actually deflected -- see
:func:`~tmom_recon.kicker.core.solve_kick`.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pytest
from xtrack_tools.kicker import run_kicker_track

from tmom_recon import KickerConfig, ModelDetails, calculate_kicker_pz

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow, pytest.mark.psb]

_KICKER = "br3.dhz12l4"
_KICK_TURN = 10
_N_TURNS = 20
_STRENGTH = 1e-5
#: An ``xt.Exciter`` with ``knl[0] = k`` deflects by ``-k`` horizontally and by
#: ``+k`` vertically, the usual multipole convention.
_TRUE_KICK = {
    "horizontal": (-_STRENGTH, 0.0),
    "vertical": (0.0, _STRENGTH),
    "diagonal": (-_STRENGTH, _STRENGTH),
}
#: MAD-NG optics against xsuite tracking, with the kicker's Twiss row exactly at
#: the kick point.  The measured worst case across this matrix is 6.5e-12 rad,
#: i.e. 6.5e-7 of the kick; the bound leaves an order of magnitude.
_KICK_TOL = 1e-10
#: The dispersive closed orbit at the kicker is 43x the kick at dp/p = 1e-3, so
#: the exact model orbit is worth far more than the kick's own precision.  The
#: measured worst case is 1e-9 m / 1e-9 rad.
_ORBIT_TOL = 1e-8


def _track(setup, plane: str, delta_p: float):
    return run_kicker_track(
        sequence_file=setup.machine.accelerator.sequence_file,
        seq_name="psb3",
        kinetic_energy=0.160,
        nturns=_N_TURNS,
        tkicker_name=_KICKER,
        kick_strength=_STRENGTH,
        plane=plane,
        kick_turn=_KICK_TURN,
        delta_p=delta_p,
        # Anchored on the trailing ring number, the Python-regex equivalent of
        # PSB.BPM_PATTERN_TEMPLATE ("^BR3%.BPM.*3$"). run_kicker_track goes
        # through xtrack, whose monitors take a regex, so MAD's Lua pattern
        # cannot be passed through verbatim -- "%." is a Lua escape and matches
        # nothing here.
        #
        # The anchor is the point: a bare "br3\.bpm.*" also matches
        # BR3.BPMT3L1, the BPMT pseudo-monitor for tune measurement, which is
        # not a reconstruction BPM and which build_orbit_reference rejects.
        # See NOTES_offmom_investigation_2026-08-11.md.
        bpm_pattern=r"(?i)^br3\.bpm.*3$",
        add_variance_columns=True,
    )


def _twiss(tws) -> pd.DataFrame:
    frame = tws.to_pandas()
    frame.index = frame["name"].astype(str).str.upper()
    return frame


@pytest.fixture(scope="module")
def setting_zero(psb_tracking_setup):
    """The measured on-momentum orbit, the model Twiss restoring it, and that Twiss.

    One extra on-momentum acquisition, shared by every cell: this is the orbit
    an operator records with the machine at its nominal setting, and it is the
    only orbit the frame is allowed to remove.  Every off-momentum cell
    therefore really does carry its dispersive closed orbit into the
    reconstruction.
    """
    data, tws, *_ = _track(psb_tracking_setup(0.0), "horizontal", 0.0)
    orbit_zero = data.loc[data["turn"] < _KICK_TURN].groupby("name", sort=False)[["x", "y"]].mean()
    reference = _twiss(tws)
    return orbit_zero, reference


@pytest.mark.parametrize("delta_p", (-1e-3, 0.0, 1e-3), ids=lambda value: f"dp_{value:+.0e}")
@pytest.mark.parametrize("plane", ("horizontal", "vertical", "diagonal"))
def test_psb_kicker_recovers_the_applied_kick(
    psb_tracking_setup,
    setting_zero,
    delta_p: float,
    plane: Literal["horizontal", "vertical", "diagonal"],
) -> None:
    orbit_zero, on_momentum_twiss = setting_zero
    setup = psb_tracking_setup(delta_p)
    data, tracked_twiss, _line, _s, _turn = _track(setup, plane, delta_p)
    at_momentum = _twiss(tracked_twiss)

    result = calculate_kicker_pz(
        data,
        ModelDetails(setup.machine.accelerator, pt=setup.measurement.pt),
        KickerConfig(kicker=_KICKER, n_turns_free=_KICK_TURN),
        closed_orbit_at_zero=orbit_zero,
        orbit_mode="dynamic",
    )

    row, kick = result.iloc[0], result.attrs["kick"]
    assert int(row["turn"]) == _tracked_kick_turn(data)

    true_px, true_py = _TRUE_KICK[plane]
    assert kick.delta_px == pytest.approx(true_px, abs=_KICK_TOL)
    assert kick.delta_py == pytest.approx(true_py, abs=_KICK_TOL)

    # The reported state is the closed orbit the beam rides plus that kick.
    for coordinate, kicked in (
        ("x", 0.0),
        ("px", kick.delta_px),
        ("y", 0.0),
        ("py", kick.delta_py),
    ):
        expected = (
            at_momentum.at[_KICKER.upper(), coordinate]
            - on_momentum_twiss.at[_KICKER.upper(), coordinate]
        )
        assert row[coordinate] - kicked == pytest.approx(expected, abs=_ORBIT_TOL), coordinate

    downstream = at_momentum.loc[at_momentum.index.isin(data["name"].str.upper()), "s"]
    downstream = downstream[downstream > at_momentum.at[_KICKER.upper(), "s"]]
    assert kick.bpm == downstream.idxmin()


def _tracked_kick_turn(data) -> int:
    """The first turn on which the tracked *momenta* leave the closed orbit.

    Truth taken from ``px``/``py``, which the reconstruction never sees.  The
    exciter's own firing turn is not usable here: its kernel indexes samples by
    the particle's arrival time, so an off-momentum particle whose ``zeta``
    slips can be caught a turn after the requested one.
    """
    baseline = data.loc[data["turn"] < _KICK_TURN].groupby("name")[["px", "py"]].mean()
    moved = data[["px", "py"]].to_numpy() - baseline.reindex(data["name"]).to_numpy()
    return int(data.loc[abs(moved).max(axis=1) > 1e-9, "turn"].min())
