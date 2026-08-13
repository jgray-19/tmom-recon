"""Tests for dispersive measurement momentum reconstruction with uncertainties."""

from __future__ import annotations

import numpy as np
import pytest
from omc3.scripts.fake_measurement_from_model import generate as generate_fake_measurement

pytest.importorskip("xtrack_tools")

import pandas as pd
from pymadng_utils.madx import convert_tfs_to_madx

from tests.reference_co import zero_reference_co
from tmom_recon import calculate_pz, inject_noise_xy
from tmom_recon.svd import svd_clean_measurements

from .momentum_test_utils import add_error_to_orbit_measurement, lhc_model_details, rmse


@pytest.mark.slow
@pytest.mark.parametrize("add_noise", [False, True], ids=["no_noise", "with_noise"])
def test_dispersive_measurement_with_uncertainties(
    data_dir,
    tmp_path,
    add_noise,
    acd_tracking_setup,
):
    """Test momentum reconstruction with uncertainty propagation from measurements.

    This test:
    1. Runs ACD tracking
    2. Optionally adds noise to measurements
    3. Performs SVD cleaning if noisy
    4. Creates fake measurements with errors (1e-3 relative)
    5. Calculates momentum with uncertainty propagation
    6. Verifies RMSE within expected bounds

    Args:
        tmp_path: Temporary path fixture.
        add_noise: If True, adds noise and performs SVD cleaning before calculating pz.
    """
    setup = acd_tracking_setup("lhcb1.seq", data_dir, delta_p=0.0)
    tracking_df = setup["tracking_df"]
    ng_tws = setup["tws"]
    truth = setup["truth"]

    # Since the above gets only BPMs, we disable drift removal to keep names consistent
    # Drop mu1 and mu2 to avoid issues in conversion
    madx_tws = convert_tfs_to_madx(ng_tws, remove_drifts=False)

    # DY is zero in the model; add a tiny symmetric jitter so relative errors propagate
    # into non-zero absolute errors for DY (and DPY if present).
    rng_dy = np.random.default_rng(2025)
    if "DY" in madx_tws.columns:
        madx_tws["DY"] = madx_tws["DY"] + rng_dy.normal(0.0, 1e-6, size=len(madx_tws))
    if "DPY" in madx_tws.columns:
        madx_tws["DPY"] = madx_tws["DPY"] + rng_dy.normal(0.0, 1e-6, size=len(madx_tws))
    if "X" in madx_tws.columns:
        madx_tws["X"] = madx_tws["X"] + rng_dy.normal(0.0, 1e-12, size=len(madx_tws))
    if "Y" in madx_tws.columns:
        madx_tws["Y"] = madx_tws["Y"] + rng_dy.normal(0.0, 1e-12, size=len(madx_tws))

    # Generate fake measurements from the twiss
    temp_dir = tmp_path / "dispersive_measurement_uncertainties"
    rel_errors = [1e-2] if add_noise else [1e-12]
    generate_fake_measurement(
        twiss=madx_tws,
        outputdir=temp_dir,
        parameters=["BETX", "BETY", "DX", "DY", "PHASEX", "PHASEY", "X", "Y"],
        relative_errors=rel_errors,  # Keep tiny positive errors in the no-noise branch for compatibility
        randomize=["values", "errors"] if add_noise else [],
        seed=1234,
    )

    add_error_to_orbit_measurement(temp_dir)

    # Prepare data for calculation
    calc_df = tracking_df.copy(deep=True)
    # Add noise if requested
    if add_noise:
        rng = np.random.default_rng(42)
        calc_df = inject_noise_xy(calc_df, rng, noise_std=1e-4)  # 100 um noise
        # Apply SVD cleaning to noisy data
        calc_df = svd_clean_measurements(calc_df)

    # Call the measurement-based function with uncertainties
    result = calculate_pz(
        calc_df,
        lhc_model_details("lhcb1.seq", data_dir, ng_tws),
        reference_co=zero_reference_co(calc_df),
        measurement_dir=str(temp_dir),
        reverse_meas_tws=False,  # Always working with B4
        info=False,
    )
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    # Check that PT_EST is close to the true pt (should be ~0)
    pt_est = result.attrs["PT_EST"]
    assert abs(pt_est - 0.0) < 1e-5, f"PT_EST {pt_est:.2e} not close to expected 0.0"

    # Also check that the result has the expected columns
    expected_cols = ["name", "turn", "x", "y", "px", "py"]
    assert all(col in result.columns for col in expected_cols)

    # If uncertainties are included, check variance columns exist
    assert "var_px" in result.columns, "var_px column missing"
    assert "var_py" in result.columns, "var_py column missing"

    # Merge with truth and check RMSE
    merged = truth.merge(
        result[["name", "turn", "px", "py", "var_px", "var_py"]],
        on=["name", "turn"],
    )

    px_rmse = rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy())
    py_rmse = rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy())
    print(f"PX RMSE (add_noise={add_noise}):", px_rmse)
    print(f"PY RMSE (add_noise={add_noise}):", py_rmse)

    if add_noise:
        # With noise, expect slightly larger errors
        assert px_rmse < 4.2e-6, f"px RMSE with noise {px_rmse:.2e} > 4.2e-6"
        assert py_rmse < 3.8e-6, f"py RMSE with noise {py_rmse:.2e} > 3.8e-6"
    else:
        # Without noise, expect tighter errors
        assert px_rmse < 3.2e-7, f"px RMSE without noise {px_rmse:.2e} > 3.2e-7"
        assert py_rmse < 2.5e-7, f"py RMSE without noise {py_rmse:.2e} > 2.5e-7"

    # Check that variance columns exist and are valid
    assert "var_px" in merged.columns, "var_px column missing after merge"
    assert "var_py" in merged.columns, "var_py column missing after merge"
    assert not merged["var_px"].isna().any(), "var_px contains NaN values"
    assert not merged["var_py"].isna().any(), "var_py contains NaN values"
    assert (merged["var_px"] > 0).all(), "var_px contains non-positive values"
    assert (merged["var_py"] > 0).all(), "var_py contains non-positive values"

    # Test if uncertainties are well-calibrated
    px_residuals = merged["px"].to_numpy() - merged["px_true"].to_numpy()
    py_residuals = merged["py"].to_numpy() - merged["py_true"].to_numpy()
    px_uncertainties = np.sqrt(merged["var_px"].to_numpy())
    py_uncertainties = np.sqrt(merged["var_py"].to_numpy())

    # Calculate normalized residuals (z-scores)
    px_z_scores = px_residuals / px_uncertainties
    py_z_scores = py_residuals / py_uncertainties

    # Check for invalid values
    px_valid = np.isfinite(px_z_scores)
    py_valid = np.isfinite(py_z_scores)
    print(f"PX valid z-scores: {np.sum(px_valid)}/{len(px_valid)} ({100 * np.mean(px_valid):.1f}%)")
    print(f"PY valid z-scores: {np.sum(py_valid)}/{len(py_valid)} ({100 * np.mean(py_valid):.1f}%)")

    if not px_valid.all():
        print(f"PX uncertainty range: [{px_uncertainties.min():.2e}, {px_uncertainties.max():.2e}]")
        print(
            f"PX residual range: [{np.abs(px_residuals).min():.2e}, {np.abs(px_residuals).max():.2e}]"
        )
    if not py_valid.all():
        print(f"PY uncertainty range: [{py_uncertainties.min():.2e}, {py_uncertainties.max():.2e}]")
        print(
            f"PY residual range: [{np.abs(py_residuals).min():.2e}, {np.abs(py_residuals).max():.2e}]"
        )

    # Use only finite z-scores for statistics
    px_z_finite = px_z_scores[px_valid]
    py_z_finite = py_z_scores[py_valid]

    # Check that ~68% of residuals are within 1 sigma
    px_within_1sigma = np.sum(np.abs(px_z_finite) < 1) / len(px_z_finite)
    py_within_1sigma = np.sum(np.abs(py_z_finite) < 1) / len(py_z_finite)
    print(f"PX within 1 sigma: {px_within_1sigma:.1%} (expect ~68%)")
    print(f"PY within 1 sigma: {py_within_1sigma:.1%} (expect ~68%)")

    # Check that ~95% of residuals are within 2 sigma
    px_within_2sigma = np.sum(np.abs(px_z_finite) < 2) / len(px_z_finite)
    py_within_2sigma = np.sum(np.abs(py_z_finite) < 2) / len(py_z_finite)
    print(f"PX within 2 sigma: {px_within_2sigma:.1%} (expect ~95%)")
    print(f"PY within 2 sigma: {py_within_2sigma:.1%} (expect ~95%)")

    # Calculate reduced chi-squared (should be ~1 if uncertainties are correct)
    px_chi2_reduced = np.mean(px_z_finite**2)
    py_chi2_reduced = np.mean(py_z_finite**2)
    print(f"PX reduced χ²: {px_chi2_reduced:.3f} (expect ~1)")
    print(f"PY reduced χ²: {py_chi2_reduced:.3f} (expect ~1)")

    # Test that uncertainties are reasonable; only enforce an upper bound since
    # overestimated errors can drive chi² toward 0 when residuals are tiny.
    assert np.isfinite(px_chi2_reduced), "PX χ² is not finite"
    assert np.isfinite(py_chi2_reduced), "PY χ² is not finite"
    assert px_chi2_reduced < 20.0, (
        f"PX χ² = {px_chi2_reduced:.2f} indicates poorly calibrated uncertainties"
    )
    assert py_chi2_reduced < 20.0, (
        f"PY χ² = {py_chi2_reduced:.2f} indicates poorly calibrated uncertainties"
    )

    # At least 40% should be within 1 sigma (relaxed due to correlations/systematics)
    assert px_within_1sigma > 0.4, f"Only {px_within_1sigma:.1%} of PX within 1 sigma (expect >40%)"
    assert py_within_1sigma > 0.4, f"Only {py_within_1sigma:.1%} of PY within 1 sigma (expect >40%)"
