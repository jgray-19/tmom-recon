"""Independent contracts for measurement noise and SVD recovery."""

from __future__ import annotations

import numpy as np
import pytest

from tests.contracts.conftest import scenario_params, truth_and_reconstruction_for_plane
from tests.support.assertions import rmse
from tmom_recon import calculate_pz, inject_noise_xy
from tmom_recon.svd import svd_clean_measurements

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]


def _plane_error(contract_scenario, data, plane: str) -> float:
    """Reconstruct one supplied data frame and return its one-plane RMSE."""
    result = calculate_pz(
        data,
        contract_scenario.model_details,
        reference=contract_scenario.reference,
        measurement_pt=contract_scenario.pt,
        use_dispersion=False,
        barrier_s=contract_scenario.barrier_s,
        acd=contract_scenario.acd,
        info=False,
    )
    truth, reconstructed = truth_and_reconstruction_for_plane(contract_scenario.data, result, plane)
    return rmse(truth.to_numpy(), reconstructed.to_numpy())


@pytest.mark.parametrize("contract_scenario", scenario_params(0.0), indirect=True)
@pytest.mark.parametrize("plane", ["x", "y"])
def test_bpm_noise_degrades_transverse_reconstruction(contract_scenario, plane: str) -> None:
    """Failure means noise no longer reaches the reconstruction as measurement noise."""
    clean = _plane_error(contract_scenario, contract_scenario.data, plane)
    noisy_data = inject_noise_xy(contract_scenario.data.copy(deep=True), np.random.default_rng(42))
    noisy = _plane_error(contract_scenario, noisy_data, plane)
    assert noisy > clean, f"p{plane} noise RMSE {noisy:.3e} is not above clean {clean:.3e}"


@pytest.mark.parametrize("contract_scenario", scenario_params(0.0), indirect=True)
@pytest.mark.parametrize("plane", ["x", "y"])
def test_svd_cleaning_improves_noisy_transverse_reconstruction(
    contract_scenario, plane: str
) -> None:
    """Failure means SVD cleaning no longer improves the noisy measurement path."""
    noisy_data = inject_noise_xy(contract_scenario.data.copy(deep=True), np.random.default_rng(42))
    noisy = _plane_error(contract_scenario, noisy_data, plane)
    cleaned = _plane_error(contract_scenario, svd_clean_measurements(noisy_data), plane)
    assert cleaned < noisy, f"p{plane} cleaned RMSE {cleaned:.3e} is not below noisy {noisy:.3e}"
