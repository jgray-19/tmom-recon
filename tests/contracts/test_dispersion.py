"""One-hypothesis contracts for the dispersive measurement pipeline."""

from __future__ import annotations

import logging

import pytest

from tests.contracts.conftest import (
    off_momentum_scenario_params,
    scenario_params,
    truth_and_reconstruction_for_plane,
)
from tests.support.assertions import rmse
from tests.support.measurements import run_dispersive_measurement

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]
LOGGER = logging.getLogger(__name__)

_MOMENTUM_RMSE_MAX = {
    "psb": {"x": 1.6e-7, "y": 1.6e-7},
    "lhcb1": {"x": 1.3e-7, "y": 9.0e-8},
    "b1_120cm_crossing": {"x": 6.6e-8, "y": 5.8e-8},
}


@pytest.mark.parametrize("contract_scenario", scenario_params(0.0), indirect=True)
def test_dispersive_measurement_estimates_zero_pt_on_momentum(contract_scenario, tmp_path) -> None:
    """The measured-optics estimator must also preserve the nominal condition."""
    result = run_dispersive_measurement(
        contract_scenario.data,
        contract_scenario.measurement_tws,
        tmp_path / "measurement-on-momentum",
        contract_scenario.nominal_details,
        frame=contract_scenario.reference,
        barrier_s=contract_scenario.barrier_s,
    )
    estimate = float(result.attrs["PT_EST"])
    LOGGER.debug("dispersion %s on-momentum: pt_est=%.12e", contract_scenario.machine, estimate)
    assert estimate == pytest.approx(0.0, abs=3.0e-6)


@pytest.mark.parametrize("contract_scenario", off_momentum_scenario_params(), indirect=True)
def test_dispersive_measurement_estimates_pt(contract_scenario, tmp_path) -> None:
    """Failure means the measured-optics dispersion path estimates the wrong pt."""
    result = run_dispersive_measurement(
        contract_scenario.data,
        contract_scenario.measurement_tws,
        tmp_path / "measurement",
        contract_scenario.nominal_details,
        frame=contract_scenario.reference,
        barrier_s=contract_scenario.barrier_s,
    )
    estimate = float(result.attrs["PT_EST"])
    LOGGER.debug(
        "dispersion %s: pt_est=%.12e truth=%.12e abs_error=%.12e",
        contract_scenario.machine,
        estimate,
        contract_scenario.pt,
        abs(estimate - contract_scenario.pt),
    )
    assert estimate == pytest.approx(contract_scenario.pt, abs=2.5e-6)


@pytest.mark.parametrize("contract_scenario", off_momentum_scenario_params(), indirect=True)
@pytest.mark.parametrize("plane", ["x", "y"])
def test_known_pt_dispersive_measurement_recovers_momentum(
    contract_scenario, plane: str, tmp_path
) -> None:
    """Failure means one-plane momentum restoration is wrong with known pt."""
    result = run_dispersive_measurement(
        contract_scenario.data,
        contract_scenario.measurement_tws,
        tmp_path / "measurement",
        contract_scenario.model_details,
        frame=contract_scenario.reference,
        measurement_pt_offset=contract_scenario.pt,
        barrier_s=contract_scenario.barrier_s,
        acd=contract_scenario.acd,
    )
    truth, reconstructed = truth_and_reconstruction_for_plane(contract_scenario.data, result, plane)
    error = rmse(truth.to_numpy(), reconstructed.to_numpy())
    limit = _MOMENTUM_RMSE_MAX[contract_scenario.machine][plane]
    LOGGER.debug(
        "dispersion %s %s: rmse=%.12e limit=%.12e", contract_scenario.machine, plane, error, limit
    )
    assert error < limit, f"p{plane} RMSE {error:.3e} exceeds {limit:.3e}"
