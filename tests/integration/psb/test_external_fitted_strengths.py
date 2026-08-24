"""Compatibility contract for strengths supplied by an external optimiser.

This repository never fits magnet strengths.  It must nevertheless prove that
an externally supplied accurate estimate can be loaded into a fresh MAD-NG
model, used to derive the unmeasurable reference angles, and recover the
tracked momenta.  The mapping below is deliberately copied from the simulated
machine rather than obtained by any optimisation: it represents the successful
output contract of ``psb_md``/``sgd-magnet-tuner`` without importing either.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.psb_tracking import ACD_ELEMENT
from tests.support.acd_barrier import acd_barrier_s
from tests.support.assertions import merge_tracking_truth, rmse
from tests.support.scenarios import MATCHED_BEND_AND_QUAD_ERRORS
from tests.support.truth import simulated_reference_from_tracking_positions_and_model_angles
from tmom_recon import ModelDetails, calculate_pz

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.slow]


def _external_estimated_strengths(scenario) -> dict[str, float]:
    """Return an optimiser-shaped copy, never the tracking model itself."""
    strengths = {f"{name.upper()}.k0": value for name, value in scenario.bend_strengths.items()}
    strengths.update(
        {f"{name.upper()}.k1": value for name, value in scenario.quad_strengths.items()}
    )
    assert strengths, "The contract needs a non-empty externally supplied strengths mapping"
    return strengths


@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_external_estimated_strengths_and_mixed_reference_recover_momenta(
    delta_p: float, psb_scenarios
) -> None:
    scenario = psb_scenarios(
        delta_p=delta_p,
        # These are deliberately separate error families: the fitted model must
        # carry both orbit-generating bends and optics-changing quadrupoles.
        errors=MATCHED_BEND_AND_QUAD_ERRORS,
    )
    nominal_scenario = psb_scenarios(
        delta_p=0.0,
        errors=MATCHED_BEND_AND_QUAD_ERRORS,
    )
    estimated_strengths = _external_estimated_strengths(scenario)
    bpm_data = scenario.measurement.data.loc[
        scenario.measurement.data["name"].isin(scenario.measurement.bpm_names)
    ].copy()

    nominal_details = ModelDetails(
        accelerator=scenario.machine.accelerator,
        pt=0.0,
        magnet_strengths=estimated_strengths,
    )
    reference = simulated_reference_from_tracking_positions_and_model_angles(
        nominal_scenario.measurement.data,
        nominal_details,
        bpm_data,
    )
    reconstruction_details = ModelDetails(
        accelerator=scenario.machine.accelerator,
        pt=scenario.measurement.pt,
        magnet_strengths=estimated_strengths,
    )

    result = calculate_pz(
        bpm_data,
        reconstruction_details,
        frame=reference,
        measurement_pt_offset=scenario.measurement.pt,
        barrier_s=acd_barrier_s(scenario.machine.madng_model, ACD_ELEMENT),
        info=False,
    )
    assert isinstance(result, pd.DataFrame)
    merged = merge_tracking_truth(bpm_data, result)
    assert merged[["px", "py"]].notna().all().all(), "All physical BPM/turn rows must reconstruct"
    assert rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy()) < 2e-6
    assert rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy()) < 2e-6
