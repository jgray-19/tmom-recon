from __future__ import annotations

import numpy as np
import pytest

from tests.momentum.momentum_test_utils import rmse
from tmom_recon import (
    calculate_pz,
    calculate_transverse_pz_nbpm,
    inject_noise_xy_inplace,
)
from tmom_recon.svd import svd_clean_measurements


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
def test_nbpm_reconstruction_returns_valid_output_for_fixture_data(
    data_dir, tracking_setup
) -> None:
    tracking_df, tws, truth = tracking_setup("lhcb1.seq", data_dir, delta_p=0.0)
    tracking_df, tws, selected_bpms = _select_local_bpm_window(tracking_df, tws)
    truth = truth[truth["name"].isin(selected_bpms)].copy(deep=True)

    result = calculate_transverse_pz_nbpm(
        tracking_df.copy(deep=True),
        tws=tws,
        twiss_elements=tws,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    )

    merged = truth.merge(
        result[["name", "turn", "px", "py", "var_px", "var_py"]], on=["name", "turn"]
    )
    assert len(merged) == len(truth)
    assert np.isfinite(merged["px"]).all()
    assert np.isfinite(merged["py"]).all()
    assert np.isfinite(merged["var_px"]).all()
    assert np.isfinite(merged["var_py"]).all()
    assert (merged["var_px"] > 0.0).all()
    assert (merged["var_py"] > 0.0).all()


@pytest.mark.slow
def test_nbpm_improves_noisy_local_window_over_two_bpm_baseline(data_dir, tracking_setup) -> None:
    tracking_df, tws, truth = tracking_setup("lhcb1.seq", data_dir, delta_p=0.0)
    tracking_df, tws, selected_bpms = _select_local_bpm_window(tracking_df, tws)
    truth = truth[truth["name"].isin(selected_bpms)].copy(deep=True)

    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(
        noisy_df,
        tracking_df,
        np.random.default_rng(42),
        noise_std=1e-4,
    )

    baseline = calculate_pz(
        noisy_df.copy(deep=True),
        model_tws=tws,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_base", "py": "py_base"})

    nbpm = calculate_transverse_pz_nbpm(
        noisy_df.copy(deep=True),
        tws=tws,
        twiss_elements=tws,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    ).rename(columns={"px": "px_nbpm", "py": "py_nbpm"})

    merged = truth.merge(
        baseline[["name", "turn", "px_base", "py_base"]],
        on=["name", "turn"],
    ).merge(
        nbpm[["name", "turn", "px_nbpm", "py_nbpm"]],
        on=["name", "turn"],
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
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    ).rename(columns={"px": "px_nbpm_cleaned", "py": "py_nbpm_cleaned"})

    merged_cleaned = truth.merge(
        nbpm_cleaned[["name", "turn", "px_nbpm_cleaned", "py_nbpm_cleaned"]],
        on=["name", "turn"],
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
