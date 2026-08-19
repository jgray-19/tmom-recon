"""LHC tracking artifact construction for integration-test fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xtrack_tools.acd import run_acd_track

from tests.support.truth import get_truth, xsuite_to_ngtws

_LHC_HORIZONTAL_EXCITATION = 2 * 0.042 / 180.0**0.5
_LHC_VERTICAL_EXCITATION = 2 * 0.042 / 177.0**0.5
_SEQUENCE_DECLARATION = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*:\s*sequence\b", re.IGNORECASE)


@dataclass(frozen=True)
class TrackingArtifacts:
    """Cached LHC tracking outputs shared across integration tests."""

    data: Any
    measurement_twiss: Any
    truth: Any
    xsuite_twiss: Any
    baseline_line: Any
    baseline_twiss_4d: Any


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
        data=artifacts.data.copy(deep=True),
        measurement_twiss=artifacts.measurement_twiss.copy(deep=True),
        truth=artifacts.truth.copy(deep=True),
        xsuite_twiss=_copy_if_possible(artifacts.xsuite_twiss),
        baseline_line=_copy_if_possible(artifacts.baseline_line),
        baseline_twiss_4d=_copy_if_possible(artifacts.baseline_twiss_4d),
    )


def _sequence_name_from_file(seq: Path) -> str:
    """Return the MAD-X sequence name declared in a file."""
    with seq.open(encoding="utf-8") as handle:
        for line in handle:
            match = _SEQUENCE_DECLARATION.match(line)
            if match:
                return match.group(1)
    return seq.stem


def tracking_artifacts_loader(xsuite_json_path):
    """Return a module-scoped-style loader for LHC tracking artifacts."""
    cache: dict[tuple, TrackingArtifacts] = {}

    def load(
        seq_path: str,
        json_path: str,
        acd_marker: str,
        sequence_name: str,
        delta_p: float,
        ramp_turns: int,
        flattop_turns: int,
        state_markers: bool,
    ) -> TrackingArtifacts:
        key = (
            seq_path,
            json_path,
            acd_marker,
            sequence_name,
            delta_p,
            ramp_turns,
            flattop_turns,
            state_markers,
        )
        if key not in cache:
            tracking_df, tws_xsuite, baseline_line = run_acd_track(
                sequence_file=Path(seq_path),
                acd_marker=acd_marker,
                sequence_name=sequence_name,
                json_path=Path(json_path),
                delta_p=delta_p,
                ramp_turns=ramp_turns,
                flattop_turns=flattop_turns,
                state_markers=state_markers,
                horizontal_excitation=_LHC_HORIZONTAL_EXCITATION,
                vertical_excitation=_LHC_VERTICAL_EXCITATION,
            )
            tws = xsuite_to_ngtws(tws_xsuite)
            cache[key] = TrackingArtifacts(
                data=tracking_df,
                measurement_twiss=tws,
                truth=get_truth(tracking_df, tws),
                xsuite_twiss=tws_xsuite,
                baseline_line=baseline_line,
                baseline_twiss_4d=baseline_line.twiss(method="4d"),
            )
        return cache[key]

    def get(
        seq_file: str,
        data_dir: Path,
        *,
        delta_p: float = 0.0,
        ramp_turns: int = 1000,
        flattop_turns: int = 100,
        state_markers: bool = False,
        acd_marker: str = "MKQA.6L4.B1",
    ) -> TrackingArtifacts:
        seq = data_dir / "sequences" / seq_file
        return _copy_tracking_artifacts(
            load(
                str(seq),
                str(xsuite_json_path(seq_file)),
                acd_marker,
                _sequence_name_from_file(seq),
                delta_p,
                ramp_turns,
                flattop_turns,
                state_markers,
            )
        )

    return get
