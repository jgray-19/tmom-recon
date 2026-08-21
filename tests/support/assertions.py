"""Domain-specific assertions and metrics for integration tests."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tmom_recon import inject_noise_xy
from tmom_recon.svd import svd_clean_measurements

LOGGER = logging.getLogger(__name__)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def merge_tracking_truth(tracking_df: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
    """Join reconstruction to truth without silently dropping BPM/turn rows."""
    keys = ["name", "turn"]
    truth = tracking_df[keys + ["px", "py"]].rename(columns={"px": "px_true", "py": "py_true"})
    merged = truth.merge(
        result[keys + ["px", "py"]],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] != "both", keys]
    if not missing.empty:
        examples = missing.head(8).to_dict(orient="records")
        raise AssertionError(
            f"Reconstruction omitted {len(missing)} tracked BPM/turn rows; examples: {examples}"
        )
    return merged.drop(columns="_merge")


def verify_pz_reconstruction(
    tracking_df,
    truth: pd.DataFrame,
    model_details,
    calculate_pz_func,
    px_nonoise_max: float,
    py_nonoise_max: float,
    px_noisy_min: float,
    px_noisy_max: float,
    py_noisy_min: float,
    py_noisy_max: float,
    px_cleaned_max: float,
    py_cleaned_max: float,
    rng_seed: int = 42,
    *,
    reference,
    barrier_s: float | None = None,
):
    """Verify momentum reconstruction with noise and SVD cleaning."""
    no_noise_result = calculate_pz_func(
        tracking_df.copy(deep=True),
        model_details,
        reference=reference,
        barrier_s=barrier_s,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(rng_seed)
    noisy_df = tracking_df.copy(deep=True)
    noisy_df = inject_noise_xy(noisy_df, rng, noise_std=1e-4)
    noisy_result = calculate_pz_func(
        noisy_df,
        model_details,
        reference=reference,
        barrier_s=barrier_s,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_noise_result = calculate_pz_func(
        cleaned_df,
        model_details,
        reference=reference,
        barrier_s=barrier_s,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_no_noise = truth.merge(
        no_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    assert len(merged_no_noise) == len(truth)
    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_nonoise = rmse(
        merged_no_noise["px_true"].to_numpy(), merged_no_noise["px_calc"].to_numpy()
    )
    py_rmse_nonoise = rmse(
        merged_no_noise["py_true"].to_numpy(), merged_no_noise["py_calc"].to_numpy()
    )
    px_rmse_noisy = rmse(merged_noisy["px_true"].to_numpy(), merged_noisy["px_calc"].to_numpy())
    py_rmse_noisy = rmse(merged_noisy["py_true"].to_numpy(), merged_noisy["py_calc"].to_numpy())
    px_rmse_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(), merged_cleaned["px_calc"].to_numpy()
    )
    py_rmse_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(), merged_cleaned["py_calc"].to_numpy()
    )

    LOGGER.info(
        "PX RMSE no noise: %.2e, noisy: %.2e, cleaned: %.2e",
        px_rmse_nonoise,
        px_rmse_noisy,
        px_rmse_cleaned,
    )
    LOGGER.info(
        "PY RMSE no noise: %.2e, noisy: %.2e, cleaned: %.2e",
        py_rmse_nonoise,
        py_rmse_noisy,
        py_rmse_cleaned,
    )

    assert px_rmse_nonoise < px_nonoise_max, (
        f"PX no-noise RMSE {px_rmse_nonoise:.2e} should be < {px_nonoise_max:.2e}"
    )
    assert py_rmse_nonoise < py_nonoise_max, (
        f"PY no-noise RMSE {py_rmse_nonoise:.2e} should be < {py_nonoise_max:.2e}"
    )
    assert px_noisy_min < px_rmse_noisy < px_noisy_max, (
        f"PX noisy RMSE {px_rmse_noisy:.2e} should be in ({px_noisy_min:.2e}, {px_noisy_max:.2e})"
    )
    assert py_noisy_min < py_rmse_noisy < py_noisy_max, (
        f"PY noisy RMSE {py_rmse_noisy:.2e} should be in ({py_noisy_min:.2e}, {py_noisy_max:.2e})"
    )
    assert px_rmse_cleaned < px_rmse_noisy, (
        f"PX cleaned {px_rmse_cleaned:.2e} should be < noisy {px_rmse_noisy:.2e}"
    )
    assert py_rmse_cleaned < py_rmse_noisy, (
        f"PY cleaned {py_rmse_cleaned:.2e} should be < noisy {py_rmse_noisy:.2e}"
    )
    assert px_rmse_cleaned < px_cleaned_max, (
        f"PX cleaned RMSE {px_rmse_cleaned:.2e} should be < {px_cleaned_max:.2e}"
    )
    assert py_rmse_cleaned < py_cleaned_max, (
        f"PY cleaned RMSE {py_rmse_cleaned:.2e} should be < {py_cleaned_max:.2e}"
    )


__all__ = ["merge_tracking_truth", "rmse", "verify_pz_reconstruction"]
