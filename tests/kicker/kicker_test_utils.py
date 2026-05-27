"""Shared data structures and helpers for kicker momentum reconstruction tests."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

TKICKER_NAME = "mkd.o5l6.b1"
KICK_STRENGTH = 1e-5  # rad
KICK_TURN = 10
NTURNS = 20


@dataclass
class KickerTrackResult:
    """Cached output of a single-turn kicker tracking run.

    Attributes:
        tracking_df: Turn-by-turn BPM measurements.
        twiss_recon: Twiss table in the format expected by
            :func:`tmom_recon.kicker.core.reconstruct_momentum_kick`.
        s_kicker: Longitudinal position of the kicker [m].
        actual_kick_turn: Turn index on which the kick fired (0-based xtrack
            convention).
        plane: Kick plane — ``"horizontal"``, ``"vertical"``, or
            ``"diagonal"``.
        check_px: Whether the reconstructed :math:`p_x` should match the true
            kick within *rel_tol_px*.
        check_py: Whether the reconstructed :math:`p_y` should match the true
            kick within *rel_tol_py*.
        rel_tol_px: Relative tolerance for the :math:`p_x` reconstruction
            accuracy assertion (only used when *check_px* is ``True``).
        rel_tol_py: Relative tolerance for the :math:`p_y` reconstruction
            accuracy assertion (only used when *check_py* is ``True``).
        abs_tol_zero: Absolute tolerance used when asserting that the
            off-plane reconstructed momentum is consistent with zero (used
            when *check_px* or *check_py* is ``False``).
    """

    tracking_df: pd.DataFrame
    twiss_recon: pd.DataFrame
    s_kicker: float
    actual_kick_turn: int
    plane: str
    check_px: bool
    check_py: bool
    rel_tol_px: float
    rel_tol_py: float
    abs_tol_zero: float


# (plane, kick_strength, check_px, check_py, rel_tol_px, rel_tol_py, abs_tol_zero)
#
# rel_tol_*    — relative tolerance for the "should match" assertion on each plane.
# abs_tol_zero — absolute tolerance for the "should be ~0" assertion on the
#                off-plane momentum; the Twiss uncoupled path gives exactly zero
#                for pure single-plane kicks, but a small residual from the
#                closed-orbit subtraction is possible.
KICKER_CASES = [
    pytest.param(("horizontal", KICK_STRENGTH, True, False, 2e-4, None, 1e-16), id="horizontal"),
    pytest.param(
        ("vertical", KICK_STRENGTH, False, True, None, 2e-4, KICK_STRENGTH / 100), id="vertical"
    ),
    pytest.param(("diagonal", KICK_STRENGTH, True, True, 1e-2, 1e-2, None), id="diagonal"),
]
