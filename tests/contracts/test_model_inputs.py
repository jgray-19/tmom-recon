"""Contracts for the MAD-NG reconstruction model input.

These tests intentionally do not compare to Xsuite Twiss.  They only verify
that the generated model provides the data consumed by tmom-recon.
"""

from __future__ import annotations

import pytest

from tests.contracts.conftest import scenario_params
from tmom_recon.model import resolve_model_details

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize("contract_scenario", scenario_params(0.0), indirect=True)
def test_generated_model_provides_all_reconstruction_columns(contract_scenario) -> None:
    """Failure means model generation, observation, or chromatic Twiss is invalid."""
    tws = resolve_model_details(contract_scenario.model_details).tws
    required = {
        "x",
        "px",
        "y",
        "py",
        "beta11",
        "beta22",
        "alfa11",
        "alfa22",
        "mu1",
        "mu2",
        "dx",
        "dpx",
        "dy",
        "dpy",
        "ddx",
        "ddpx",
        "ddy",
        "ddpy",
    }
    missing = required.difference(tws.columns)
    assert not missing, f"MAD-NG model is missing reconstruction columns: {sorted(missing)}"
