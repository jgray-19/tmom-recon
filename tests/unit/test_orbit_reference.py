from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.orbit_reference import build_orbit_reference, normalize_closed_orbit


def _model() -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [1.0, 2.0], "px": [3.0, 4.0], "y": [5.0, 6.0], "py": [7.0, 8.0]},
        index=pd.Index(["BPM1", "BPM2"], name="name"),
    )


def test_closed_orbit_is_case_insensitive_and_aligned() -> None:
    orbit = pd.DataFrame({"name": ["bpm2", "BpM1"], "x": [20.0, 10.0], "y": [2.0, 1.0]})
    result = normalize_closed_orbit(orbit, ["BPM1", "bpm2"])
    assert result.index.tolist() == ["BPM1", "BPM2"]
    assert result["x"].tolist() == [10.0, 20.0]


@pytest.mark.parametrize(
    ("orbit", "match"),
    [
        (pd.DataFrame({"name": ["BPM1"], "x": [0.0], "y": [0.0]}), "missing"),
        (pd.DataFrame({"name": ["BPM1", "bpm1"], "x": [0.0, 0.0], "y": [0.0, 0.0]}), "duplicate"),
        (
            pd.DataFrame({"name": ["BPM1", "BPM2"], "x": [0.0, np.nan], "y": [0.0, 0.0]}),
            "non-finite",
        ),
    ],
)
def test_invalid_closed_orbit_rejected(orbit, match) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_closed_orbit(orbit, ["BPM1", "BPM2"])


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="orbit_mode"):
        build_orbit_reference(_model()[["x", "y"]], "relative", _model(), ["BPM1"])


def test_dynamic_restores_complete_generated_model_state() -> None:
    measured = _model()[["x", "y"]] * 10.0
    reference = build_orbit_reference(measured, "dynamic", _model(), ["BPM1", "BPM2"])
    pd.testing.assert_frame_equal(reference.restored, _model().astype(float))


def test_absolute_restores_measured_positions_and_generated_angles() -> None:
    measured = _model()[["x", "y"]] * 10.0
    reference = build_orbit_reference(measured, "absolute", _model(), ["BPM1", "BPM2"])
    pd.testing.assert_frame_equal(reference.restored[["x", "y"]], measured.astype(float))
    pd.testing.assert_frame_equal(
        reference.restored[["px", "py"]], _model()[["px", "py"]].astype(float)
    )
