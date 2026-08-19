"""Tests for dispersive measurement momentum reconstruction."""

from __future__ import annotations

import pytest
from pymadng_utils.accelerators import LHC

from tests.support.measurements import assert_dispersive_measurement_recovers_pt
from tests.support.model_details import model_details_for
from tests.support.truth import simulated_nominal_reference_from_model


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
