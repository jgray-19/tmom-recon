"""Integration tests for kicker momentum reconstruction using xtrack-tools kicker."""

from __future__ import annotations

import pytest

pytest.importorskip("xtrack_tools")

from pathlib import Path

import pandas as pd

from tmom_recon.kicker.core import reconstruct_momentum_kick

from .kicker_test_utils import KICK_TURN, KickerTrackResult


def _true_delta_pz(
    tracking_df: pd.DataFrame,
    twiss_recon: pd.DataFrame,
    s_kicker: float,
    kick_turn: int,
    n_turns_free: int,
) -> tuple[float, float, float, float]:
    """Extract the true delta px/py at the kicker using the first downstream BPM.

    At the first BPM immediately downstream of the kicker, ``px`` and ``py``
    are approximately equal to ``delta_px`` / ``delta_py`` (the kick-induced
    momentum change), because there has been minimal betatron evolution between
    the kicker and that BPM.

    Args:
        tracking_df: Full turn-by-turn tracking DataFrame.
        twiss_recon: Twiss table (with ``"kicker"`` row) in reconstruction format.
        s_kicker: Longitudinal position of the kicker [m].
        kick_turn: Turn index on which the kick fired.
        n_turns_free: Number of pre-kick turns used to estimate the closed orbit.

    Returns:
        ``(px, delta_px, py, delta_py)`` where ``delta_*`` is the CO-subtracted
        momentum at the first downstream BPM on the kick turn.
    """
    bpm_s = twiss_recon.loc[twiss_recon.index != "kicker", "s"]
    first_bpm = str(bpm_s[bpm_s > s_kicker].idxmin())

    bpm_data = tracking_df[tracking_df["name"] == first_bpm].sort_values("turn")
    co_px = float(bpm_data[bpm_data["turn"] < n_turns_free]["px"].mean())
    co_py = float(bpm_data[bpm_data["turn"] < n_turns_free]["py"].mean())
    kick_row = bpm_data[bpm_data["turn"] == kick_turn].iloc[0]
    px = float(kick_row["px"])
    py = float(kick_row["py"])
    return px, px - co_px, py, py - co_py


@pytest.mark.slow
def test_kicker_momentum_reconstruction(
    kicker_track: KickerTrackResult,
    seq_b1: Path,
) -> None:
    """Kick strength is recovered from BPM data to within the per-plane tolerance.

    Parametrised over three kick planes via the session-scoped
    :func:`kicker_track` fixture, which runs xtrack tracking once per plane
    and caches the result.  Tolerances are set per-plane in
    :data:`conftest._KICKER_CASES` because the Twiss uncoupled method has
    different accuracy for horizontal, vertical, and diagonal kicks.
    """
    n_turns_free = int(KICK_TURN / 2)

    result = reconstruct_momentum_kick(
        kicker_track.tracking_df,
        twiss=kicker_track.twiss_recon,
        n_turns_free=n_turns_free,
        n_turns_after_kick=3,
    )

    nonzero_col = "px" if kicker_track.check_px else "py"
    kick_rows = result[result[nonzero_col] != 0.0]
    assert not kick_rows.empty, f"No non-zero {nonzero_col} found in reconstruction result"
    kick_row = kick_rows.iloc[0]
    kick_turn_found = int(kick_row["turn"])

    reconstructed_px = kick_row["px"]
    reconstructed_py = kick_row["py"]

    _true_px, true_delta_px, _true_py, true_delta_py = _true_delta_pz(
        kicker_track.tracking_df,
        kicker_track.twiss_recon,
        kicker_track.s_kicker,
        kick_turn_found,
        kicker_track.actual_kick_turn,
    )

    if kicker_track.check_px:
        assert (
            abs(reconstructed_px - true_delta_px) / abs(true_delta_px) < kicker_track.rel_tol_px
        ), (
            f"[{kicker_track.plane}] Reconstructed px {reconstructed_px:.4e} should match "
            f"true delta_px {true_delta_px:.4e} within {kicker_track.rel_tol_px:.0%}"
        )
    else:
        assert abs(reconstructed_px) < kicker_track.abs_tol_zero, (
            f"[{kicker_track.plane}] Reconstructed px {reconstructed_px:.4e} "
            f"should be ~0 (tol={kicker_track.abs_tol_zero:.0e})"
        )

    if kicker_track.check_py:
        assert (
            abs(reconstructed_py - true_delta_py) / abs(true_delta_py) < kicker_track.rel_tol_py
        ), (
            f"[{kicker_track.plane}] Reconstructed py {reconstructed_py:.4e} should match "
            f"true delta_py {true_delta_py:.4e} within {kicker_track.rel_tol_py:.0%}"
        )
    else:
        assert abs(reconstructed_py) < kicker_track.abs_tol_zero, (
            f"[{kicker_track.plane}] Reconstructed py {reconstructed_py:.4e} "
            f"should be ~0 (tol={kicker_track.abs_tol_zero:.0e})"
        )
