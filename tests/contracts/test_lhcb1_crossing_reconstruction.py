"""Crossing-orbit LHC B1 contracts; machine identity is not parametrised."""

import pytest

from tests.contracts.conftest import machine_reconstruction_params
from tests.contracts.test_reconstruction import (
    assert_all_bpm_with_matched_magnetic_errors,
    assert_model_pt_estimate_matches_tracked_offset,
    assert_reconstruction,
)

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow, pytest.mark.lhc]
PARAMS = machine_reconstruction_params("b1_120cm_crossing")
PT_PARAMS = machine_reconstruction_params(
    "b1_120cm_crossing", workflows=("all",), conditions=("clean",), frame_kinds=("dynamic",)
)

DRIVEN_PARAMS = machine_reconstruction_params("b1_120cm_crossing", workflows=("driven_all",))


@pytest.mark.parametrize("contract_scenario", PARAMS, indirect=True)
def test_reconstruction(contract_scenario):
    assert_reconstruction(contract_scenario)


@pytest.mark.parametrize("contract_scenario", PT_PARAMS, indirect=True)
def test_model_pt_estimate(contract_scenario):
    assert_model_pt_estimate_matches_tracked_offset(contract_scenario)


@pytest.mark.parametrize("contract_scenario", DRIVEN_PARAMS, indirect=True)
def test_driven_optics_reconstruction(contract_scenario):
    assert_reconstruction(contract_scenario)


def test_matched_magnetic_errors(data_dir, psb_scenarios, tmp_path_factory, xsuite_json_path):
    assert_all_bpm_with_matched_magnetic_errors(
        "b1_120cm_crossing", data_dir, psb_scenarios, tmp_path_factory, xsuite_json_path
    )
