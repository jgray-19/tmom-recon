"""Domain-specific assertions and metrics for integration tests."""

import numpy as np

from tests.momentum.momentum_test_utils import verify_pz_reconstruction


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


__all__ = ["rmse", "verify_pz_reconstruction"]
