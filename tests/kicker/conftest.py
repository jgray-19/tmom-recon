"""Shared fixtures for kicker momentum reconstruction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .kicker_test_utils import (
    KICK_TURN,
    KICKER_CASES,
    NTURNS,
    TKICKER_NAME,
    KickerTrackResult,
    build_twiss_for_recon,
)


@pytest.fixture(scope="session", params=KICKER_CASES)
def kicker_track(request, seq_b1: Path) -> KickerTrackResult:
    """Run single-turn kicker tracking once per plane and cache the result.

    The fixture is session-scoped so that the (expensive) xtrack tracking is
    executed only once per kick plane, regardless of how many tests consume it.

    Args:
        request: Pytest parametrize request carrying ``(plane, kick_strength,
            check_px, check_py, rel_tol_px, rel_tol_py, abs_tol_zero)``.
        seq_b1: Path to the LHC beam-1 MAD-X sequence file.

    Returns:
        :class:`KickerTrackResult` with tracking data, Twiss table, and
        metadata for the assertions.

    Raises:
        ``pytest.skip``: if ``xtrack_tools`` is not installed.
    """
    pytest.importorskip("xtrack_tools")
    from xtrack_tools.kicker import run_kicker_track

    plane, kick_strength, check_px, check_py, rel_tol_px, rel_tol_py, abs_tol_zero = request.param

    tracking_df, tws, _line, s_kicker, actual_kick_turn = run_kicker_track(
        sequence_file=seq_b1,
        nturns=NTURNS,
        tkicker_name=TKICKER_NAME,
        kick_strength=kick_strength,
        plane=plane,
        kick_turn=KICK_TURN,
    )

    return KickerTrackResult(
        tracking_df=tracking_df,
        twiss_recon=build_twiss_for_recon(tws, TKICKER_NAME),
        s_kicker=s_kicker,
        actual_kick_turn=actual_kick_turn,
        plane=plane,
        check_px=check_px,
        check_py=check_py,
        rel_tol_px=rel_tol_px,
        rel_tol_py=rel_tol_py,
        abs_tol_zero=abs_tol_zero,
    )
