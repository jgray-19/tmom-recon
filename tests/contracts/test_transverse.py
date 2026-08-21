"""Single-purpose contracts for the plain transverse reconstruction path."""

from __future__ import annotations

import logging

import pytest

from tests.contracts.conftest import (
    off_momentum_scenario_params,
    scenario_params,
    truth_and_reconstruction_for_plane,
)
from tests.support.assertions import rmse
from tmom_recon import calculate_pz

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]
LOGGER = logging.getLogger(__name__)

_RMSE_MAX = {
    "psb": {"x": 1.2e-7, "y": 1.2e-7},
    "lhcb1": {"x": 1.4e-7, "y": 8.7e-8},
    "b1_120cm_crossing": {"x": 6.6e-8, "y": 5.8e-8},
}


@pytest.mark.parametrize("contract_scenario", scenario_params(0.0), indirect=True)
@pytest.mark.parametrize("plane", ["x", "y"])
def test_clean_on_momentum_transverse_reconstruction(contract_scenario, plane: str) -> None:
    """Failure means the plain model/reference/neighbour momentum path is wrong."""
    result = calculate_pz(
        contract_scenario.data,
        contract_scenario.model_details,
        reference=contract_scenario.reference,
        measurement_pt=contract_scenario.pt,
        use_dispersion=False,
        barrier_s=contract_scenario.barrier_s,
        acd=contract_scenario.acd,
        info=False,
    )
    truth, reconstructed = truth_and_reconstruction_for_plane(contract_scenario.data, result, plane)
    error = rmse(truth.to_numpy(), reconstructed.to_numpy())
    limit = _RMSE_MAX[contract_scenario.machine][plane]
    LOGGER.debug(
        "transverse %s %s: rmse=%.12e limit=%.12e", contract_scenario.machine, plane, error, limit
    )
    assert error < limit, f"p{plane} RMSE {error:.3e} exceeds {limit:.3e}"


@pytest.mark.parametrize(
    "contract_scenario",
    off_momentum_scenario_params(),
    indirect=True,
)
@pytest.mark.parametrize("plane", ["x", "y"])
def test_corrected_lhc_off_momentum_orbit_reconstructs(contract_scenario, plane: str) -> None:
    """The corrected off-momentum LHC orbit keeps both transverse planes usable."""
    result = calculate_pz(
        contract_scenario.data,
        contract_scenario.model_details,
        reference=contract_scenario.reference,
        measurement_pt=contract_scenario.pt,
        # Off-momentum corrected orbits require the dispersive model columns;
        # keep the full machine parametrization valid rather than excluding PSB.
        use_dispersion=True,
        barrier_s=contract_scenario.barrier_s,
        info=False,
    )
    truth, reconstructed = truth_and_reconstruction_for_plane(contract_scenario.data, result, plane)
    # Reuse the established LHC clean transverse limits.
    error = rmse(truth.to_numpy(), reconstructed.to_numpy())
    LOGGER.debug(
        "corrected orbit %s %s: rmse=%.12e limit=%.12e",
        contract_scenario.machine,
        plane,
        error,
        _RMSE_MAX[contract_scenario.machine][plane],
    )
    assert error < _RMSE_MAX[contract_scenario.machine][plane]
