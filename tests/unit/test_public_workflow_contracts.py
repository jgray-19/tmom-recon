"""Fast contracts for the focused public reconstruction surface."""

from __future__ import annotations

import inspect

import pytest

import tmom_recon
from tmom_recon import OpticsInput, calculate_acd_pz, calculate_kicker_pz, calculate_pz

PUBLIC_API = {
    "ACDipoleConfig",
    "ACDipolePzGenerator",
    "Kick",
    "KickerConfig",
    "KickerPzGenerator",
    "ModelDetails",
    "ModelOpticsErrors",
    "OpticsInput",
    "PzGenerator",
    "ResolvedOptics",
    "build_twiss_from_measurements",
    "calculate_acd_pz",
    "calculate_kicker_pz",
    "calculate_pz",
    "estimate_closed_orbit_pt",
    "estimate_pt_from_orbit",
    "inject_noise_xy",
    "resolve_optics",
}


def test_top_level_public_api_is_focused() -> None:
    assert set(tmom_recon.__all__) == PUBLIC_API
    assert not hasattr(tmom_recon, "calculate_transverse_pz")
    assert not hasattr(tmom_recon, "calculate_transverse_pz_nbpm")
    assert not hasattr(tmom_recon, "calculate_ac_dipole_momentum")


def test_optics_input_defaults_every_category_to_model() -> None:
    optics = OpticsInput()
    assert {
        category: optics.source_for(category)
        for category in ("phase", "beta", "alpha", "dispersion")
    } == {
        "phase": "model",
        "beta": "model",
        "alpha": "model",
        "dispersion": "model",
    }


def test_measured_optics_requires_a_measurement_directory() -> None:
    with pytest.raises(ValueError, match="measurement_dir"):
        OpticsInput(sources={"phase": "measurement"})


def test_all_bpm_requires_an_explicit_barrier_decision() -> None:
    assert (
        inspect.signature(calculate_pz).parameters["barrier_s"].default is inspect.Parameter.empty
    )


def test_workflows_are_separate_public_operations() -> None:
    assert calculate_acd_pz is not calculate_pz
    assert calculate_kicker_pz is not calculate_pz


@pytest.mark.parametrize("function", [calculate_pz, calculate_acd_pz, calculate_kicker_pz])
def test_reconstruction_apis_require_explicit_orbit_inputs(function) -> None:
    parameters = inspect.signature(function).parameters
    assert "frame" not in parameters
    assert parameters["closed_orbit_at_zero"].default is inspect.Parameter.empty
    assert parameters["orbit_mode"].default is inspect.Parameter.empty


def test_frame_symbols_are_removed() -> None:
    for name in ("Frame", "DynamicFrame", "AbsoluteFrame"):
        assert name not in tmom_recon.__all__
        assert not hasattr(tmom_recon, name)


def test_matrix_requires_negative_zero_and_positive_momenta() -> None:
    from tests.contracts.conftest import machine_reconstruction_params

    params = machine_reconstruction_params(
        "psb", workflows=("all",), conditions=("clean",), frame_kinds=("dynamic",)
    )
    values = {param.values[0].delta_p for param in params}
    assert any(value < 0.0 for value in values)
    assert 0.0 in values
    assert any(value > 0.0 for value in values)
