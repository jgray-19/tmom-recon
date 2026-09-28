"""Fast contracts for serial-session caches and committed external inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tests.support import lhc as lhc_support
from tests.support.external_strengths import load_external_strength_fixture


class _FakeLine:
    def copy(self):
        return self

    def twiss(self, *, method: str):
        assert method == "4d"
        return {"method": method}


def _tracking_frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["BPM.1"],
            "turn": [0],
            "x": [value],
            "px": [value],
            "y": [value],
            "py": [value],
        }
    )


def test_lhc_dataframe_cache_reuses_data_and_returns_defensive_copies(monkeypatch) -> None:
    calls: list[Path] = []

    def fake_track(**kwargs):
        calls.append(kwargs["sequence_file"])
        return _tracking_frame(float(len(calls))), object(), _FakeLine()

    monkeypatch.setattr(lhc_support, "run_acd_track", fake_track)
    loader = lhc_support.tracking_artifacts_loader(lambda path: path.with_suffix(".json"))
    first = loader(Path("lhcb1.seq"))
    first.data.loc[0, "x"] = 999.0
    second = loader(Path("lhcb1.seq"))

    assert len(calls) == 1
    assert second.data.loc[0, "x"] == 1.0
    assert first.data is not second.data


def test_lhc_dataframe_cache_evicts_least_recently_used_by_bytes(monkeypatch) -> None:
    calls: list[Path] = []

    def fake_track(**kwargs):
        calls.append(kwargs["sequence_file"])
        return _tracking_frame(float(len(calls))), object(), _FakeLine()

    frame_bytes = int(_tracking_frame(0.0).memory_usage(index=True, deep=True).sum())
    monkeypatch.setattr(lhc_support, "TRACKING_DATA_CACHE_BYTES", frame_bytes + 1)
    monkeypatch.setattr(lhc_support, "run_acd_track", fake_track)
    loader = lhc_support.tracking_artifacts_loader(lambda path: path.with_suffix(".json"))

    loader(Path("first.seq"))
    loader(Path("second.seq"))
    loader(Path("first.seq"))

    assert calls == [Path("first.seq"), Path("second.seq"), Path("first.seq")]


def test_lhc_lines_are_opt_in_and_never_cached(monkeypatch) -> None:
    calls = 0

    def fake_track(**_kwargs):
        nonlocal calls
        calls += 1
        return _tracking_frame(float(calls)), object(), _FakeLine()

    monkeypatch.setattr(lhc_support, "run_acd_track", fake_track)
    loader = lhc_support.tracking_artifacts_loader(lambda path: path.with_suffix(".json"))

    assert loader(Path("lhcb1.seq")).baseline_line is None
    assert loader(Path("lhcb1.seq"), include_line=True).baseline_line is not None
    assert loader(Path("lhcb1.seq")).baseline_line is None
    assert loader(Path("lhcb1.seq"), include_line=True).baseline_line is not None
    assert calls == 3


@pytest.mark.parametrize("machine", ["psb", "lhcb1", "b1_120cm_crossing"])
def test_committed_external_strength_fixture_is_valid(machine: str) -> None:
    path = Path(__file__).parents[1] / "data" / "external_strengths" / f"{machine}.json"
    fixture = load_external_strength_fixture(path)
    assert fixture.machine == machine
    assert fixture.strengths
    assert fixture.fingerprint == fixture.metadata["strength_fingerprint_sha256"]


def test_external_strength_fixture_rejects_a_tampered_mapping(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "data" / "external_strengths" / "psb.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    first = next(iter(payload["strengths"]))
    payload["strengths"][first] += 1.0
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_external_strength_fixture(tampered)
