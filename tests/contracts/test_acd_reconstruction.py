"""Local AC-dipole override contract shared by all supported sequences."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.acd.acd_test_helpers import r_squared
from tests.contracts.conftest import acd_scenario_params
from tmom_recon import calculate_pz, inject_noise_xy

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]
LOGGER = logging.getLogger(__name__)

# These are the existing clean local-ACD BPM limits: PSB's direct test accepts
# 0.99, while the LHC direct test requires 0.9998 on momentum.
_BPM_R2_MIN = {"psb": 0.9990, "lhcb1": 0.99982, "b1_120cm_crossing": 0.99982}


@pytest.mark.parametrize("contract_scenario", acd_scenario_params(), indirect=True)
@pytest.mark.parametrize("plane", ["px", "py"])
def test_acd_override_recovers_each_adjacent_bpm(contract_scenario, plane: str) -> None:
    """Failure identifies ACD state transport or all-BPM override installation."""
    data = contract_scenario.data.copy(deep=True)
    if contract_scenario.noise_std:
        data = inject_noise_xy(data, np.random.default_rng(42), contract_scenario.noise_std)
    result = calculate_pz(
        data,
        contract_scenario.model_details,
        reference=contract_scenario.reference,
        measurement_pt=contract_scenario.pt,
        barrier_s=contract_scenario.barrier_s,
        acd=contract_scenario.acd,
        info=False,
    )
    acd_result = result.attrs["acd_result"]
    for side in ("upstream", "downstream"):
        bpm = str(acd_result.attrs[f"bpm_{side}"])
        truth = contract_scenario.data.loc[
            contract_scenario.data["name"].astype(str).str.upper() == bpm.upper(),
            ["turn", plane],
        ].rename(columns={plane: "truth"})
        reconstructed = result.loc[
            result["name"].astype(str).str.upper() == bpm.upper(), ["turn", plane]
        ]
        merged = truth.merge(reconstructed, on="turn", validate="one_to_one")
        assert len(merged) == len(truth), f"{side} BPM {bpm} was omitted from all-BPM output"
        assert np.isfinite(merged[["truth", plane]].to_numpy()).all(), (
            f"{side} BPM {bpm} has undefined {plane}"
        )
        r2 = r_squared(merged["truth"], merged[plane])
        LOGGER.debug(
            "acd %s dp=%s noise=%s %s: r2=%.12e floor=%.12e",
            contract_scenario.machine,
            contract_scenario.delta_p,
            contract_scenario.noise_std,
            plane,
            r2,
            _BPM_R2_MIN[contract_scenario.machine],
        )
        assert r2 > _BPM_R2_MIN[contract_scenario.machine], f"{side} BPM {bpm} {plane} R^2={r2:.6f}"
