"""O(n) diagonal of a banded SPD inverse (ACD cleaning variances) against the dense inverse."""

from __future__ import annotations

import numpy as np
import pytest

from tmom_recon.acd.cleaning import _banded_inverse_diagonal


def _upper_banded(a: np.ndarray, p: int) -> np.ndarray:
    ab = np.zeros((p + 1, len(a)))
    for d in range(p + 1):
        ab[p - d, d:] = np.diag(a, d)
    return ab


@pytest.mark.parametrize("n", [1, 2, 3, 50, 2000])
def test_matches_dense_inverse(n) -> None:
    """The cleaning system: inverse-variance weights plus a second-difference penalty."""
    rng = np.random.default_rng(n)
    a = np.diag(rng.uniform(0.1, 5.0, n))
    if n > 2:
        d2 = np.diff(np.eye(n), 2, axis=0)
        a = a + 0.7 * d2.T @ d2
    got = _banded_inverse_diagonal(_upper_banded(a, 2), bandwidth=2)
    np.testing.assert_allclose(got, np.diag(np.linalg.inv(a)), rtol=1e-12)


def test_not_positive_definite_raises() -> None:
    with pytest.raises(np.linalg.LinAlgError):
        _banded_inverse_diagonal(_upper_banded(-np.eye(4), 2), bandwidth=2)
