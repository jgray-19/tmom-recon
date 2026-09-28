"""Domain-specific assertions and metrics for integration tests."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

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


__all__ = ["merge_tracking_truth", "rmse"]
