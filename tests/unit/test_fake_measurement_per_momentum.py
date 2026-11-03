"""The fake measurement must not pool phase across momenta.

omc3 computes phase from *every* input file unless ``analyse_dpp`` is set
(``omc3.optics_measurements.phase``: ``dpp_value = meas_input.analyse_dpp`` and,
when that is ``None``, a joined frame over all files). Feeding one optics run
the -1e-3, 0 and +1e-3 acquisitions together therefore produces a single phase
that belongs to no momentum, and reusing it at +/-1e-3 is a systematic the
reconstruction cannot see.

``psb_md.acd_workflow`` avoids this by running phase analysis once per orbit
group against a model built at that group's ``dpp``, and copying the combined
run's dispersion in afterwards. These tests pin the fixture to that shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from tests.support import fake_measurement
from tests.support.fake_measurement import (
    DISPERSION_MEASUREMENT_FILES,
    Acquisition,
    run_fake_measurement,
)

PTS = (-1e-3, 0.0, 1e-3)


def _frame() -> pd.DataFrame:
    rows = [
        {"name": name, "turn": turn, "x": 0.0, "y": 0.0}
        for name in ("BPM1", "BPM2")
        for turn in range(4)
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def calls(monkeypatch, tmp_path) -> list[dict]:
    """Record every hole_in_one call and fake the files optics would write."""
    recorded: list[dict] = []

    def fake_entrypoint(**kwargs):
        recorded.append(kwargs)
        outputdir = Path(kwargs["outputdir"])
        outputdir.mkdir(parents=True, exist_ok=True)
        if kwargs.get("optics"):
            for filename in DISPERSION_MEASUREMENT_FILES:
                (outputdir / filename).write_text(str(kwargs["model_dir"]))
        else:
            for file in kwargs["files"]:
                (outputdir / Path(file.meta["file"]).name).write_text("lin")

    monkeypatch.setattr(fake_measurement, "hole_in_one_entrypoint", fake_entrypoint)
    return recorded


@pytest.fixture()
def acquisitions(tmp_path) -> list[Acquisition]:
    built = []
    for index, pt in enumerate(PTS):
        model_dir = tmp_path / f"model_{index}"
        model_dir.mkdir()
        built.append(
            Acquisition(
                data=_frame(),
                pt=pt,
                model_dir=model_dir,
                natural_tunes=(0.17 + pt, 0.225 + pt),
                driven_tunes=(0.16, 0.24),
            )
        )
    return built


def _optics_calls(calls: list[dict]) -> list[dict]:
    return [call for call in calls if call.get("optics")]


def test_every_momentum_gets_its_own_optics_run(calls, acquisitions, tmp_path):
    """A per-momentum run reads exactly one lin file, so no pooling can occur."""
    run_fake_measurement(acquisitions, output_dir=tmp_path / "out")

    single_file = [call for call in _optics_calls(calls) if len(call["files"]) == 1]
    # Three momenta, for the driven and the free workflow.
    assert len(single_file) == 2 * len(PTS)


def test_the_only_pooled_run_is_the_combined_one(calls, acquisitions, tmp_path):
    """Dispersion needs the momentum spread; nothing else may see it."""
    run_fake_measurement(acquisitions, output_dir=tmp_path / "out")

    pooled = [call for call in _optics_calls(calls) if len(call["files"]) > 1]
    assert [Path(call["outputdir"]).name for call in pooled] == ["combined", "combined"]


def test_each_momentum_uses_its_own_model(calls, acquisitions, tmp_path):
    """A shared pt=0 model would put phase and its reference at different pt."""
    run_fake_measurement(acquisitions, output_dir=tmp_path / "out")

    used = {Path(call["model_dir"]) for call in _optics_calls(calls) if len(call["files"]) == 1}
    assert used == {acquisition.model_dir for acquisition in acquisitions}


def test_harpy_follows_the_acquisition_tunes(calls, acquisitions, tmp_path):
    """The natural tune moves with momentum; the AC-dipole drive does not."""
    run_fake_measurement(acquisitions, output_dir=tmp_path / "out")

    harpy = [call for call in calls if call.get("harpy")]
    assert [call["nattunes"][:2] for call in harpy] == [
        list(acquisition.natural_tunes) for acquisition in acquisitions
    ]
    assert {tuple(call["tunes"][:2]) for call in harpy} == {(0.16, 0.24)}


def test_results_are_keyed_by_momentum_with_combined_dispersion(calls, acquisitions, tmp_path):
    """Each directory pairs its own phase with the shared dispersion."""
    results = run_fake_measurement(acquisitions, output_dir=tmp_path / "out")

    assert set(results) == {"acd", "all"}
    for workflow, per_momentum in results.items():
        assert set(per_momentum) == set(PTS)
        assert len(set(per_momentum.values())) == len(PTS)
        combined_model = str(
            min(acquisitions, key=lambda acquisition: abs(acquisition.pt)).model_dir
        )
        for directory in per_momentum.values():
            for filename in DISPERSION_MEASUREMENT_FILES:
                # Copied from the combined run, not measured single-momentum.
                assert (directory / filename).read_text() == combined_model, workflow
