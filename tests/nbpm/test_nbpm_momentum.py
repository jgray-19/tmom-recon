from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.reference_co import measured_zero_reference_for_simulation
from tests.support.assertions import rmse
from tests.support.lhc import get_twiss, lhc_model_details
from tmom_recon import (
    calculate_pz,
    calculate_transverse_pz_nbpm,
    inject_noise_xy,
)
from tmom_recon.svd import svd_clean_measurements

pytestmark = [pytest.mark.lhc, pytest.mark.integration]


def _select_local_bpm_window(
    tracking_df,
    tws_bpm,
    *,
    window_size: int = 24,
):
    bpm_mask = tws_bpm.index.str.match(r"^BPM.*\.B1$")
    tws_bpms = tws_bpm.loc[bpm_mask].copy(deep=True)
    mid_start = max(0, len(tws_bpms) // 2 - window_size // 2)
    selected_bpms = tws_bpms.index[mid_start : mid_start + window_size]
    return (
        tracking_df[tracking_df["name"].isin(selected_bpms)].copy(deep=True),
        tws_bpms.loc[selected_bpms].copy(deep=True),
        selected_bpms,
    )


@pytest.mark.slow
def test_nbpm_reconstruction_returns_valid_output_for_fixture_data(seq_b1, tracking_setup) -> None:
    tracking_df = tracking_setup(seq_b1, delta_p=0.0)
    tws = get_twiss(seq_b1, deltap=0.0)
    tracking_df, tws, selected_bpms = _select_local_bpm_window(tracking_df, tws)

    result = calculate_transverse_pz_nbpm(
        tracking_df.copy(deep=True),
        tws=tws,
        twiss_elements=tws,
        info=False,
        max_bpm_distance=11,
    )

    merged = tracking_df.merge(
        result[["name", "turn", "px", "py", "var_px", "var_py"]],
        on=["name", "turn"],
        suffixes=("_tracked", "_reconstructed"),
        validate="one_to_one",
        indicator=True,
    )
    assert (merged["_merge"] == "both").all()
    assert np.isfinite(merged["px_reconstructed"]).all()
    assert np.isfinite(merged["py_reconstructed"]).all()
    assert np.isfinite(merged["var_px_reconstructed"]).all()
    assert np.isfinite(merged["var_py_reconstructed"]).all()
    assert (merged["var_px_reconstructed"] > 0.0).all()
    assert (merged["var_py_reconstructed"] > 0.0).all()


@pytest.mark.slow
def test_nbpm_improves_noisy_local_window_over_two_bpm_baseline(seq_b1, tracking_setup) -> None:
    tracking_df = tracking_setup(seq_b1, delta_p=0.0)
    tws = get_twiss(seq_b1, deltap=0.0)
    tracking_df, tws, selected_bpms = _select_local_bpm_window(tracking_df, tws)

    noisy_df = tracking_df.copy(deep=True)
    noisy_df = inject_noise_xy(noisy_df, np.random.default_rng(42), noise_std=1e-4)

    baseline = calculate_pz(
        noisy_df.copy(deep=True),
        lhc_model_details(seq_b1, delta_p=0.0),
        frame=measured_zero_reference_for_simulation(noisy_df),
        barrier_s=None,
        info=False,
    )
    assert isinstance(baseline, pd.DataFrame)

    nbpm = calculate_transverse_pz_nbpm(
        noisy_df.copy(deep=True),
        tws=tws,
        twiss_elements=tws,
        info=False,
        max_bpm_distance=11,
    ).rename(columns={"px": "px_nbpm", "py": "py_nbpm"})

    merged = tracking_df.merge(
        baseline[["name", "turn", "px", "py"]],
        on=["name", "turn"],
        suffixes=("_true", "_base"),
        validate="one_to_one",
        indicator="base_merge",
    ).merge(
        nbpm[["name", "turn", "px_nbpm", "py_nbpm"]],
        on=["name", "turn"],
        validate="one_to_one",
        indicator="nbpm_merge",
    )

    px_rmse_base = rmse(merged["px_true"].to_numpy(), merged["px_base"].to_numpy())
    py_rmse_base = rmse(merged["py_true"].to_numpy(), merged["py_base"].to_numpy())
    px_rmse_nbpm = rmse(merged["px_true"].to_numpy(), merged["px_nbpm"].to_numpy())
    py_rmse_nbpm = rmse(merged["py_true"].to_numpy(), merged["py_nbpm"].to_numpy())

    assert px_rmse_nbpm <= px_rmse_base
    assert py_rmse_nbpm <= py_rmse_base

    cleaned_df = svd_clean_measurements(noisy_df)
    nbpm_cleaned = calculate_transverse_pz_nbpm(
        cleaned_df.copy(deep=True),
        tws=tws,
        twiss_elements=tws,
        info=False,
        max_bpm_distance=11,
    )

    merged_cleaned = tracking_df.merge(
        nbpm_cleaned[["name", "turn", "px", "py"]],
        on=["name", "turn"],
        suffixes=("_true", "_nbpm_cleaned"),
        validate="one_to_one",
        indicator=True,
    )
    px_rmse_nbpm_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_nbpm_cleaned"].to_numpy(),
    )
    py_rmse_nbpm_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_nbpm_cleaned"].to_numpy(),
    )

    assert px_rmse_nbpm_cleaned <= px_rmse_nbpm
    assert py_rmse_nbpm_cleaned <= py_rmse_nbpm
