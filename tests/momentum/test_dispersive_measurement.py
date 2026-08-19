"""Tests for dispersive measurement momentum reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
from pymadng_utils.accelerators import LHC

from tests.reference_co import measured_zero_reference_for_simulation
from tests.support.assertions import rmse
from tests.support.measurements import (
    assert_dispersive_measurement_recovers_pt,
    run_dispersive_measurement,
)
from tests.support.model_details import model_details_for
from tests.support.truth import simulated_nominal_reference_from_model, xsuite_to_ngtws

__test__ = False


@pytest.mark.slow
@pytest.mark.lhc
@pytest.mark.integration
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
@pytest.mark.parametrize("delta_p", [0.0, 4e-4])
def test_dispersive_measurement_recovers_pt(
    data_dir, seq_file, tmp_path, delta_p, acd_tracking_setup
):
    """Test that calculate_pz_measurement recovers the true pt from measurements."""
    setup = acd_tracking_setup(seq_file, data_dir, delta_p=delta_p)
    accelerator = LHC(
        beam=1,
        sequence_file=data_dir / "sequences" / seq_file,
        kinetic_energy=6800,
    )
    nominal_model = model_details_for(accelerator, pt=0.0)

    assert_dispersive_measurement_recovers_pt(
        setup.data,
        setup.measurement_twiss,
        setup.truth,
        tmp_path / "dispersive_measurement",
        accelerator.dp2pt(delta_p),
        # Plain dispersive reconstruction: the model is the nominal (on-momentum)
        # optics and the beam pt is estimated, so the dispersive orbit is not
        # double-counted.
        nominal_model,
        px_rmse_max=3.4e-7,
        py_rmse_max=2.8e-7,
        reference=simulated_nominal_reference_from_model(nominal_model, setup.data),
        reverse_meas_tws=False,  # Always working with B4
    )


@pytest.mark.slow
@pytest.mark.psb
@pytest.mark.integration
@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_offmomentum_psb(tmp_path, delta_p, psb_tracking_setup):
    """Dispersive-measurement reconstruction for a PSB ring-3 AC-dipole excitation.

    Mirrors :func:`test_dispersive_measurement_recovers_pt` but for the PSB, using
    the shared PSB tracking setup and the dispersive measurement pipeline (no
    AC-dipole cleaning of the reconstruction).
    """
    scenario = psb_tracking_setup(delta_p)
    nominal_model = model_details_for(scenario.machine.accelerator, pt=0.0)
    xsuite_measurement_tws = xsuite_to_ngtws(scenario.machine.xsuite_twiss)

    assert_dispersive_measurement_recovers_pt(
        scenario.measurement.data,
        xsuite_measurement_tws,
        scenario.measurement.truth,
        tmp_path / "dispersive_measurement_psb",
        scenario.measurement.pt,
        nominal_model,
        # This error-free setup has a zero nominal-RF orbit. The measured twiss
        # still represents the off-momentum beam optics.
        reference=measured_zero_reference_for_simulation(scenario.measurement.data),
        px_rmse_max=9e-7,
        py_rmse_max=9e-7,
    )


@pytest.mark.slow
@pytest.mark.psb
@pytest.mark.integration
@pytest.mark.parametrize("delta_p", [1e-3], ids=["off_momentum"])
def test_offmomentum_psb_pt_estimation(tmp_path, delta_p, psb_tracking_setup):
    """The dispersive measurement estimates the absolute PSB momentum."""
    scenario = psb_tracking_setup(delta_p)
    model = model_details_for(scenario.machine.accelerator, pt=0.0)
    result = run_dispersive_measurement(
        scenario.measurement.data,
        xsuite_to_ngtws(scenario.machine.xsuite_twiss),
        tmp_path / "dispersive_measurement_psb_pt",
        model,
        reference=measured_zero_reference_for_simulation(scenario.measurement.data),
    )

    assert result.attrs["PT_EST"] == pytest.approx(scenario.measurement.pt, abs=1e-5)


@pytest.mark.slow
@pytest.mark.psb
@pytest.mark.integration
@pytest.mark.parametrize("delta_p", [1e-3], ids=["off_momentum"])
def test_offmomentum_psb_reconstruction_with_known_pt(tmp_path, delta_p, psb_tracking_setup):
    """Momentum reconstruction remains accurate when the PT estimator is bypassed."""
    scenario = psb_tracking_setup(delta_p)
    model = model_details_for(scenario.machine.accelerator, pt=0.0)
    result = run_dispersive_measurement(
        scenario.measurement.data,
        xsuite_to_ngtws(scenario.machine.xsuite_twiss),
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


@pytest.mark.slow
@pytest.mark.psb
@pytest.mark.integration
@pytest.mark.crosscode
@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_psb_xsuite_madng_optics_agreement(delta_p, psb_tracking_setup):
    """The generated PSB model agrees across the tracking and reconstruction codes."""
    scenario = psb_tracking_setup(delta_p)
    xsuite_tws = xsuite_to_ngtws(scenario.machine.xsuite_twiss)
    madng_tws = scenario.machine.madng_twiss
    common = xsuite_tws.index.intersection(madng_tws.index)
    assert len(common) > 4

    for coordinate in ("x", "px", "y", "py"):
        difference = np.abs(
            xsuite_tws.loc[common, coordinate].to_numpy(float)
            - madng_tws.loc[common, coordinate].to_numpy(float)
        )
        assert difference.max() < 1e-5, (
            f"{coordinate} Xsuite/MAD-NG mismatch: {difference.max():.3e}"
        )

    for coordinate in ("beta11", "beta22", "alfa11", "alfa22", "dx", "dpx"):
        difference = np.abs(
            xsuite_tws.loc[common, coordinate].to_numpy(float)
            - madng_tws.loc[common, coordinate].to_numpy(float)
        )
        assert difference.max() < 1e-3, (
            f"{coordinate} Xsuite/MAD-NG mismatch: {difference.max():.3e}"
        )

    madng_qx = madng_tws.headers.get("q1", madng_tws.headers.get("Q1"))
    madng_qy = madng_tws.headers.get("q2", madng_tws.headers.get("Q2"))
    assert madng_qx is not None and madng_qy is not None
    assert scenario.machine.xsuite_twiss.qx == pytest.approx(madng_qx, abs=1e-3)
    assert scenario.machine.xsuite_twiss.qy == pytest.approx(madng_qy, abs=1e-3)
