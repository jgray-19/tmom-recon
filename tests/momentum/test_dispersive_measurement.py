"""Tests for dispersive measurement momentum reconstruction."""

from __future__ import annotations

import pytest
from omc3.scripts.fake_measurement_from_model import generate as generate_fake_measurement
from pymadng_utils.madx import convert_tfs_to_madx

from tmom_recon import calculate_pz

from .momentum_test_utils import add_error_to_orbit_measurement, rmse


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
@pytest.mark.parametrize("delta_p", [0.0, 4e-4])
def test_dispersive_measurement_recovers_dpp(
    data_dir, seq_file, tmp_path, delta_p, acd_tracking_setup
):
    """Test that calculate_pz_measurement recovers the true DPP from measurements."""
    setup = acd_tracking_setup(seq_file, data_dir, delta_p=delta_p)
    tracking_df = setup["tracking_df"]
    ng_tws = setup["tws"]
    truth = setup["truth"]

    # Since the above gets only BPMs, we disable drift removal to keep names consistent
    # Drop mu1 and mu2 to avoid issues in conversion
    madx_tws = convert_tfs_to_madx(ng_tws.drop(columns=["mu1", "mu2"]), remove_drifts=False)

    # Generate fake measurements from the twiss
    meas_dir = tmp_path / "dispersive_measurement_uncertainties"
    generate_fake_measurement(
        twiss=madx_tws,
        outputdir=meas_dir,
        parameters=["BETX", "BETY", "DX", "DY", "PHASEX", "PHASEY", "X", "Y"],
    )

    # Add a nonzero orbit error
    add_error_to_orbit_measurement(meas_dir)

    # Call the measurement-based function
    # The function now handles closed orbit removal and px/py restoration internally
    result = calculate_pz(
        tracking_df.copy(deep=True),
        measurement_dir=str(meas_dir),
        model_tws=ng_tws,
        reverse_meas_tws=False,  # Always working with B4
        info=False,
    )

    # Check that DPP_EST is close to the true delta_p
    dpp_est = result.attrs["DPP_EST"]
    assert abs(dpp_est - delta_p) < 1e-5, f"DPP_EST {dpp_est:.2e} not close to true {delta_p:.2e}"

    # Also check that the result has the expected columns
    expected_cols = ["name", "turn", "x", "y", "px", "py"]
    assert all(col in result.columns for col in expected_cols)

    # Merge with truth and check RMSE
    merged = truth.merge(
        result[["name", "turn", "px", "py"]],
        on=["name", "turn"],
    )

    px_rmse = rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy())
    py_rmse = rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy())

    print(f"Dispersive measurement px RMSE: {px_rmse:.2e}, py RMSE: {py_rmse:.2e}")

    assert px_rmse < 3.4e-7, f"px RMSE {px_rmse:.2e} > 3.4e-7"
    assert py_rmse < 2.8e-7, f"py RMSE {py_rmse:.2e} > 2.7e-7"
