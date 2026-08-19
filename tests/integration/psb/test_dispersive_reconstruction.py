"""PSB dispersive reconstruction contracts separated by pipeline stage."""

from __future__ import annotations

import pytest

from tests.reference_co import measured_zero_reference_for_simulation
from tests.support.assertions import rmse
from tests.support.measurements import (
    assert_dispersive_measurement_recovers_pt,
    run_dispersive_measurement,
)
from tests.support.model_details import model_details_for
from tests.support.truth import xsuite_to_ngtws

pytestmark = [pytest.mark.psb, pytest.mark.integration]


@pytest.mark.slow
@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_offmomentum_psb(tmp_path, delta_p, psb_tracking_setup):
    """Dispersive-measurement reconstruction for a PSB ring-3 AC-dipole excitation."""
    scenario = psb_tracking_setup(delta_p)
    nominal_model = model_details_for(scenario.machine.accelerator, pt=0.0)
    xsuite_measurement_tws = xsuite_to_ngtws(
        scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names
    )

    assert_dispersive_measurement_recovers_pt(
        scenario.measurement.data,
        xsuite_measurement_tws,
        scenario.measurement.truth,
        tmp_path / "dispersive_measurement_psb",
        scenario.measurement.pt,
        nominal_model,
        reference=measured_zero_reference_for_simulation(scenario.measurement.data),
        px_rmse_max=9e-7,
        py_rmse_max=9e-7,
    )


@pytest.mark.slow
@pytest.mark.parametrize("delta_p", [1e-3], ids=["off_momentum"])
def test_offmomentum_psb_pt_estimation(tmp_path, delta_p, psb_tracking_setup):
    """The dispersive measurement estimates the absolute PSB momentum."""
    scenario = psb_tracking_setup(delta_p)
    model = model_details_for(scenario.machine.accelerator, pt=0.0)
    result = run_dispersive_measurement(
        scenario.measurement.data,
        xsuite_to_ngtws(scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names),
        tmp_path / "dispersive_measurement_psb_pt",
        model,
        reference=measured_zero_reference_for_simulation(scenario.measurement.data),
    )

    assert result.attrs["PT_EST"] == pytest.approx(scenario.measurement.pt, abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("delta_p", [1e-3], ids=["off_momentum"])
def test_offmomentum_psb_reconstruction_with_known_pt(tmp_path, delta_p, psb_tracking_setup):
    """Momentum reconstruction remains accurate when the PT estimator is bypassed."""
    scenario = psb_tracking_setup(delta_p)
    model = model_details_for(scenario.machine.accelerator, pt=0.0)
    result = run_dispersive_measurement(
        scenario.measurement.data,
        xsuite_to_ngtws(scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names),
        tmp_path / "dispersive_measurement_psb_known_pt",
        model,
        reference=measured_zero_reference_for_simulation(scenario.measurement.data),
        measurement_pt=scenario.measurement.pt,
    )
    merged = scenario.measurement.truth.merge(
        result[["name", "turn", "px", "py"]], on=["name", "turn"]
    )

    assert rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy()) < 9e-7
    assert rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy()) < 9e-7
