"""Tests for dispersive measurement momentum reconstruction."""

from __future__ import annotations

import pytest
from pymadng_utils.accelerators import LHC

from .momentum_test_utils import assert_dispersive_measurement_recovers_pt, model_details_for


@pytest.mark.slow
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

    assert_dispersive_measurement_recovers_pt(
        setup["tracking_df"],
        setup["tws"],
        setup["truth"],
        tmp_path / "dispersive_measurement",
        accelerator.dp2pt(delta_p),
        # Plain dispersive reconstruction: the model is the nominal (on-momentum)
        # optics and the beam pt is estimated, so the dispersive orbit is not
        # double-counted.
        model_details_for(accelerator, pt=0.0),
        px_rmse_max=3.4e-7,
        py_rmse_max=2.8e-7,
        reverse_meas_tws=False,  # Always working with B4
    )


@pytest.mark.slow
@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_offmomentum_psb(tmp_path, delta_p, psb_tracking_setup):
    """Dispersive-measurement reconstruction for a PSB ring-3 AC-dipole excitation.

    Mirrors :func:`test_dispersive_measurement_recovers_pt` but for the PSB, using
    the shared PSB tracking setup and the dispersive measurement pipeline (no
    AC-dipole cleaning of the reconstruction).
    """
    setup = psb_tracking_setup(delta_p)

    assert_dispersive_measurement_recovers_pt(
        setup["tracking_df"],
        setup["tws"],
        setup["truth"],
        tmp_path / "dispersive_measurement_psb",
        setup["model"].pt,
        model_details_for(setup["model"].accelerator, pt=0.0),
        px_rmse_max=9e-7,
        py_rmse_max=9e-7,
    )
