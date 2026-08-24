"""Measurement generation and reconstruction helpers for integration tests."""

from __future__ import annotations

import pandas as pd
import tfs
from omc3.optics_measurements.constants import ERR, EXT, ORBIT, ORBIT_NAME
from omc3.scripts.fake_measurement_from_model import generate as generate_fake_measurement
from pymadng_utils.madx import convert_tfs_to_madx

from tests.support.assertions import rmse
from tmom_recon import ACDipoleConfig, ModelDetails, ReconstructionFrame, calculate_pz


def add_error_to_orbit_measurement(fldr):
    """Inject a fixed orbit error into the fake measurement files."""
    for plane in ["x", "y"]:
        meas_file = fldr / f"{ORBIT_NAME}{plane}{EXT}"
        df = tfs.read(meas_file)
        df[f"{ERR}{ORBIT}{plane.upper()}"] = 1e-6
        tfs.write(meas_file, df)


def run_dispersive_measurement(
    tracking_df: pd.DataFrame,
    measurement_tws: pd.DataFrame,
    meas_dir,
    model_details: ModelDetails,
    *,
    reference: ReconstructionFrame,
    barrier_s: float | None,
    acd: ACDipoleConfig | None = None,
    reverse_meas_tws: bool = False,
    measurement_pt: float | None = None,
):
    """Generate a fake dispersive measurement and reconstruct the momenta."""
    madx_tws = convert_tfs_to_madx(measurement_tws, remove_drifts=False)
    generate_fake_measurement(
        twiss=madx_tws,
        outputdir=meas_dir,
        parameters=["BETX", "BETY", "DX", "DY", "PHASEX", "PHASEY", "X", "Y"],
    )
    add_error_to_orbit_measurement(meas_dir)
    result = calculate_pz(
        tracking_df.copy(deep=True),
        model_details,
        frame=reference,
        use_dispersion=True,
        measurement_dir=str(meas_dir),
        reverse_meas_tws=reverse_meas_tws,
        measurement_pt_offset=measurement_pt,
        barrier_s=barrier_s,
        acd=acd,
        info=False,
    )
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    return result


def assert_dispersive_measurement_recovers_pt(
    tracking_df: pd.DataFrame,
    measurement_tws: pd.DataFrame,
    meas_dir,
    expected_pt: float,
    model_details: ModelDetails,
    *,
    reference: ReconstructionFrame,
    barrier_s: float | None,
    acd: ACDipoleConfig | None = None,
    px_rmse_max: float,
    py_rmse_max: float,
    reverse_meas_tws: bool = False,
    measurement_pt: float | None = None,
):
    """Check estimated momentum and reconstructed transverse momentum."""
    result = run_dispersive_measurement(
        tracking_df,
        measurement_tws,
        meas_dir,
        model_details,
        reference=reference,
        reverse_meas_tws=reverse_meas_tws,
        measurement_pt=measurement_pt,
        barrier_s=barrier_s,
        acd=acd,
    )

    pt_est = result.attrs["PT_EST"]
    assert abs(pt_est - expected_pt) < 1e-5, (
        f"PT_EST {pt_est:.2e} not close to true {expected_pt:.2e}"
    )

    expected_cols = ["name", "turn", "x", "y", "px", "py"]
    assert all(col in result.columns for col in expected_cols)

    merged = tracking_df.merge(
        result[["name", "turn", "px", "py"]],
        on=["name", "turn"],
        suffixes=("_true", ""),
        validate="one_to_one",
        indicator=True,
    )
    px_rmse_value = rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy())
    py_rmse_value = rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy())
    print(f"Dispersive measurement px RMSE: {px_rmse_value:.2e}, py RMSE: {py_rmse_value:.2e}")
    assert px_rmse_value < px_rmse_max, f"px RMSE {px_rmse_value:.2e} > {px_rmse_max:.2e}"
    assert py_rmse_value < py_rmse_max, f"py RMSE {py_rmse_value:.2e} > {py_rmse_max:.2e}"
    return result


__all__ = [
    "add_error_to_orbit_measurement",
    "assert_dispersive_measurement_recovers_pt",
    "run_dispersive_measurement",
]
