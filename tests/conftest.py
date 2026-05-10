"""
Common pytest fixtures for MAD interface tests.

This module contains shared fixtures used across MAD interface test modules.
"""

from __future__ import annotations

import logging
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Configure logging for tests
logging.getLogger("xdeps").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def seq_b1(data_dir: Path) -> Path:
    """Path to the example sequence file for beam 1 used by several tests."""
    return data_dir / "sequences" / "lhcb1.seq"


@pytest.fixture(scope="session")
def xsuite_json_path(data_dir: Path) -> Callable[[str], Path]:
    """Get the xsuite JSON path for a given sequence file.

    Returns a callable that takes a sequence file name (e.g., "lhcb1.seq")
    and returns the path to its pre-generated JSON file in data/sequences.
    """
    sequences_dir = data_dir / "sequences"

    def _get_json_path(seq_file: str) -> Path:
        # Extract base name without extension and create JSON path
        return sequences_dir / Path(seq_file).with_suffix(".json")

    return _get_json_path


@pytest.fixture(scope="session")
def tracking_setup(xsuite_json_path):
    """Return cached turn-by-turn tracking data, twiss, and truth for a sequence."""

    @cache
    def _load(seq_path: str, json_path: str, delta_p: float, ramp_turns: int, flattop_turns: int):
        run_acd_track = pytest.importorskip("xtrack_tools.acd").run_acd_track
        from tests.momentum.momentum_test_utils import get_truth, xsuite_to_ngtws

        tracking_df, tws, _baseline_line = run_acd_track(
            json_path=Path(json_path),
            sequence_file=Path(seq_path),
            delta_p=delta_p,
            ramp_turns=ramp_turns,
            flattop_turns=flattop_turns,
        )
        tws = xsuite_to_ngtws(tws)
        truth = get_truth(tracking_df, tws)
        return tracking_df, tws, truth

    def _get(
        seq_file: str,
        data_dir: Path,
        delta_p: float = 0.0,
        ramp_turns: int = 1000,
        flattop_turns: int = 100,
    ):
        seq = data_dir / "sequences" / seq_file
        json_path = xsuite_json_path(seq_file)
        tracking_df, tws, truth = _load(
            str(seq),
            str(json_path),
            delta_p,
            ramp_turns,
            flattop_turns,
        )
        return tracking_df.copy(deep=True), tws.copy(deep=True), truth.copy(deep=True)

    return _get
