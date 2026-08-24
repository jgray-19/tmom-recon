"""Contract for an optimiser-owned magnetic-strength hand-off.

tmom-recon must consume a supplied mapping; it must never run an optimiser.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest
from xtrack_tools.monitors import process_tracking_data
from xtrack_tools.tracking import run_tracking_without_ac_dipole

from tests.contracts.conftest import truth_and_reconstruction_for_plane
from tests.psb_tracking import ACD_ELEMENT
from tests.support.acd_barrier import acd_barrier_s
from tests.support.assertions import rmse
from tests.support.external_strengths import (
    ExternalStrengthFixture,
    load_external_strength_fixture,
)
from tests.support.lhc import lhc_acd_barrier_s, setup_xsuite_simulation
from tests.support.scenarios import MATCHED_BEND_AND_QUAD_ERRORS
from tests.support.truth import simulated_reference_from_tracking_positions_and_model_angles
from tmom_recon import ModelDetails, ReconstructionFrame, calculate_pz

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FittedStrengthCase:
    """Truth data plus a fresh model built from an external-strength mapping."""

    machine: str
    data: pd.DataFrame
    model_details: ModelDetails
    reference: ReconstructionFrame
    barrier_s: float
    external: ExternalStrengthFixture


def _runtime_strengths_for_psb(scenario) -> dict[str, float]:
    strengths = {f"{name.upper()}.k0": value for name, value in scenario.bend_strengths.items()}
    strengths.update(
        {f"{name.upper()}.k1": value for name, value in scenario.quad_strengths.items()}
    )
    return strengths


def _assert_fixture_matches_sequence(external: ExternalStrengthFixture, data_dir: Path) -> None:
    sequence = external.metadata["sequence"]
    if external.machine == "psb":
        source = data_dir / "acc-models-psb"
        digest = hashlib.sha256()
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    else:
        path = data_dir.parent.parent / sequence["path"]
        digest = hashlib.sha256(path.read_bytes())
    assert digest.hexdigest() == sequence["sha256"]


def _tracked_nominal_reference(line) -> pd.DataFrame:
    """Acquire a position reference from an explicit zero-coordinate track.

    Xsuite Twiss is deliberately not consulted: tracked coordinates are useful
    as simulation truth here, while Xsuite optics are not an authority for the
    reconstruction or its reference state.
    """
    particle_coords = {coordinate: [0.0] for coordinate in ("x", "px", "y", "py")}
    tracked_line = run_tracking_without_ac_dipole(
        line=line,
        tws=None,
        flattop_turns=8,
        bpm_pattern=r"(?i)bpm.*",
        particle_coords=particle_coords,
    )
    return process_tracking_data(
        tracked_line,
        ramp_turns=0,
        flattop_turns=8,
        add_variance_columns=False,
    )


@pytest.fixture(scope="module")
def fitted_strength_case(
    request: pytest.FixtureRequest,
    data_dir: Path,
    psb_scenarios,
    tmp_path_factory: pytest.TempPathFactory,
    xsuite_json_path,
) -> FittedStrengthCase:
    """Create one magnetic-error machine and load its frozen external mapping."""
    machine = request.param
    external = load_external_strength_fixture(data_dir / "external_strengths" / f"{machine}.json")
    assert external.machine == machine
    _assert_fixture_matches_sequence(external, data_dir)
    if machine == "psb":
        scenario = psb_scenarios(delta_p=1e-3, errors=MATCHED_BEND_AND_QUAD_ERRORS)
        nominal_scenario = psb_scenarios(delta_p=0.0, errors=MATCHED_BEND_AND_QUAD_ERRORS)
        runtime_strengths = _runtime_strengths_for_psb(scenario)
        assert runtime_strengths == pytest.approx(external.strengths, rel=0.0, abs=0.0)
        data = scenario.measurement.data.loc[
            scenario.measurement.data["name"].isin(scenario.measurement.bpm_names)
        ].copy()
        nominal = ModelDetails(
            accelerator=scenario.machine.accelerator,
            pt=0.0,
            magnet_strengths=dict(external.strengths),
        )
        model_details = replace(nominal, pt=scenario.measurement.pt)
        barrier_s = acd_barrier_s(scenario.machine.madng_model, ACD_ELEMENT)
        nominal_tracking_data = nominal_scenario.measurement.data
    else:
        sequence_file = data_dir / "sequences" / f"{machine}.seq"
        data, _, model_details, _, baseline_line = setup_xsuite_simulation(
            0.0,
            "all",
            12,
            xsuite_json_path(sequence_file.name),
            sequence_file,
            tmp_path_factory.mktemp(f"external-strengths-{machine}"),
            f"external_strengths_{machine}",
        )
        assert model_details.magnet_strengths, "tracking must yield a non-empty mapping"
        assert model_details.magnet_strengths == pytest.approx(external.strengths, rel=0.0, abs=0.0)
        model_details = replace(model_details, magnet_strengths=dict(external.strengths))
        nominal = replace(model_details, pt=0.0)
        barrier_s = lhc_acd_barrier_s(model_details.accelerator, model_details.pt)
        nominal_tracking_data = _tracked_nominal_reference(baseline_line)

    return FittedStrengthCase(
        machine=machine,
        data=data,
        model_details=model_details,
        reference=simulated_reference_from_tracking_positions_and_model_angles(
            nominal_tracking_data, nominal, data
        ),
        barrier_s=barrier_s,
        external=external,
    )


pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]

# Reuse established limits: PSB's externally fitted-strength contract and the
# existing LHC matched-magnet perturbation test. New contracts never loosen one.
_RMSE_MAX = {"psb": 1.2e-7, "lhcb1": 1.4e-7, "b1_120cm_crossing": 1.0e-7}


@pytest.mark.parametrize(
    "fitted_strength_case",
    [
        pytest.param("psb", id="psb", marks=pytest.mark.psb),
        pytest.param("lhcb1", id="lhcb1", marks=pytest.mark.lhc),
        pytest.param("b1_120cm_crossing", id="b1_120cm_crossing", marks=pytest.mark.lhc),
    ],
    indirect=True,
)
@pytest.mark.parametrize("plane", ["x", "y"])
def test_external_fitted_strengths_recover_momentum(fitted_strength_case, plane: str) -> None:
    """Failure means the fitted-strength/reference-angle hand-off is broken."""
    assert fitted_strength_case.external.strengths
    assert fitted_strength_case.model_details.magnet_strengths is not None
    assert (
        fitted_strength_case.model_details.magnet_strengths
        is not fitted_strength_case.external.strengths
    )
    assert (
        fitted_strength_case.model_details.magnet_strengths
        == fitted_strength_case.external.strengths
    )
    result = calculate_pz(
        fitted_strength_case.data,
        fitted_strength_case.model_details,
        frame=fitted_strength_case.reference,
        measurement_pt_offset=fitted_strength_case.model_details.pt,
        barrier_s=fitted_strength_case.barrier_s,
        info=False,
    )
    truth, reconstructed = truth_and_reconstruction_for_plane(
        fitted_strength_case.data, result, plane
    )
    error = rmse(truth.to_numpy(), reconstructed.to_numpy())
    limit = _RMSE_MAX[fitted_strength_case.machine]
    LOGGER.debug(
        "external strengths %s %s: rmse=%.12e limit=%.12e",
        fitted_strength_case.machine,
        plane,
        error,
        limit,
    )
    assert error < limit, f"p{plane} RMSE {error:.3e} exceeds {limit:.3e}"
