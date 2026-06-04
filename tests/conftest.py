"""
Common pytest fixtures for MAD interface tests.

This module contains shared fixtures used across MAD interface test modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

# Configure logging for tests
logging.getLogger("xdeps").setLevel(logging.WARNING)


@dataclass(frozen=True)
class TrackingArtifacts:
    """Cached tracking outputs shared across slow integration tests."""

    tracking_df: Any
    tws: Any
    truth: Any
    tws_xsuite: Any
    baseline_line: Any
    baseline_twiss_4d: Any


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


def _copy_if_possible(value: Any):
    copy_method = getattr(value, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(deep=True)
        except TypeError:
            return copy_method()
    return value


def _copy_tracking_artifacts(artifacts: TrackingArtifacts) -> TrackingArtifacts:
    return TrackingArtifacts(
        tracking_df=artifacts.tracking_df.copy(deep=True),
        tws=artifacts.tws.copy(deep=True),
        truth=artifacts.truth.copy(deep=True),
        tws_xsuite=_copy_if_possible(artifacts.tws_xsuite),
        baseline_line=_copy_if_possible(artifacts.baseline_line),
        baseline_twiss_4d=_copy_if_possible(artifacts.baseline_twiss_4d),
    )


@pytest.fixture(scope="session")
def tracking_artifacts(xsuite_json_path):
    """Return cached ACD tracking artifacts for a sequence."""

    @cache
    def _load(seq_path: str, json_path: str, delta_p: float, ramp_turns: int, flattop_turns: int):
        run_acd_track = pytest.importorskip("xtrack_tools.acd").run_acd_track
        from tests.momentum.momentum_test_utils import get_truth, xsuite_to_ngtws

        tracking_df, tws_xsuite, baseline_line = run_acd_track(
            json_path=Path(json_path),
            sequence_file=Path(seq_path),
            delta_p=delta_p,
            ramp_turns=ramp_turns,
            flattop_turns=flattop_turns,
        )
        tws = xsuite_to_ngtws(tws_xsuite)
        truth = get_truth(tracking_df, tws)
        return TrackingArtifacts(
            tracking_df=tracking_df,
            tws=tws,
            truth=truth,
            tws_xsuite=tws_xsuite,
            baseline_line=baseline_line,
            baseline_twiss_4d=baseline_line.twiss(method="4d"),
        )

    def _get(
        seq_file: str,
        data_dir: Path,
        delta_p: float = 0.0,
        ramp_turns: int = 1000,
        flattop_turns: int = 100,
    ):
        seq = data_dir / "sequences" / seq_file
        json_path = xsuite_json_path(seq_file)
        return _copy_tracking_artifacts(
            _load(
                str(seq),
                str(json_path),
                delta_p,
                ramp_turns,
                flattop_turns,
            )
        )

    return _get


@pytest.fixture(scope="session")
def tracking_setup(tracking_artifacts):
    """Compatibility wrapper returning tracking data, twiss, and truth."""

    def _get(
        seq_file: str,
        data_dir: Path,
        delta_p: float = 0.0,
        ramp_turns: int = 1000,
        flattop_turns: int = 100,
    ):
        artifacts = tracking_artifacts(
            seq_file,
            data_dir,
            delta_p=delta_p,
            ramp_turns=ramp_turns,
            flattop_turns=flattop_turns,
        )
        return artifacts.tracking_df, artifacts.tws, artifacts.truth

    return _get


@pytest.fixture(scope="session")
def acd_tracking_setup(tracking_artifacts):
    """Compatibility wrapper exposing the richer tracking artifact bundle."""

    def _get(
        seq_file: str,
        data_dir: Path,
        *,
        delta_p: float = 0.0,
        ramp_turns: int = 1000,
        flattop_turns: int = 100,
    ):
        artifacts = tracking_artifacts(
            seq_file,
            data_dir,
            delta_p=delta_p,
            ramp_turns=ramp_turns,
            flattop_turns=flattop_turns,
        )
        return {
            "tracking_df": artifacts.tracking_df,
            "tws": artifacts.tws,
            "truth": artifacts.truth,
            "tws_xsuite": artifacts.tws_xsuite,
            "baseline_line": artifacts.baseline_line,
            "baseline_twiss_4d": artifacts.baseline_twiss_4d,
        }

    return _get
