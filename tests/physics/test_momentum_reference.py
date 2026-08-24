"""Contracts for the strict measured orbit-zero reconstruction frame."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from tmom_recon import ReconstructionFrame, calculate_pz

pytestmark = pytest.mark.unit
NAMES = pd.Index(["bpm1", "bpm2"], name="name")


def _origin():
    return pd.DataFrame({"x": [1e-3, -1e-3], "y": [2e-3, -2e-3]}, index=NAMES)


def _momenta():
    return pd.DataFrame({"px": [1e-4, -1e-4], "py": [2e-4, -2e-4]}, index=NAMES)


def test_absolute_frame_restores_measured_positions_and_fitted_angles():
    frame = ReconstructionFrame(_origin(), fitted_momenta=_momenta())
    pd.testing.assert_frame_equal(frame.closed_orbit, pd.concat([_origin(), _momenta()], axis=1))


def test_dynamic_frame_zeroes_state_but_keeps_measured_origin():
    frame = ReconstructionFrame(_origin(), dynamic_planes=("x", "y"))
    assert (frame.closed_orbit == 0.0).all().all()
    pd.testing.assert_frame_equal(frame.orbit_zero, _origin())


def test_mixed_frame_restores_only_retained_plane():
    frame = ReconstructionFrame(_origin(), dynamic_planes=("y",), fitted_momenta=_momenta()[["px"]])
    assert (frame.closed_orbit[["y", "py"]] == 0.0).all().all()
    assert frame.closed_orbit["x"].equals(_origin()["x"])
    assert frame.closed_orbit["px"].equals(_momenta()["px"])


def test_retained_plane_requires_explicit_fitted_angle():
    with pytest.raises(ValueError, match="explicitly fitted"):
        ReconstructionFrame(_origin())


def test_duplicate_bpms_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ReconstructionFrame(pd.concat([_origin(), _origin().iloc[[0]]]), dynamic_planes=("x", "y"))


def test_legacy_api_is_absent():
    parameters = inspect.signature(calculate_pz).parameters
    assert "frame" in parameters and "measurement_pt_offset" in parameters
    assert "reference" not in parameters and "measurement_pt" not in parameters


def test_barrier_decision_remains_explicit():
    assert (
        inspect.signature(calculate_pz).parameters["barrier_s"].default is inspect.Parameter.empty
    )
