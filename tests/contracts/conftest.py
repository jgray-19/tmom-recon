"""Shared machine scenarios for the canonical diagnostic contracts.

Tracking coordinates are truth only.  Every reconstruction input is produced
by MAD-NG through :class:`~tmom_recon.ModelDetails` or the measurement helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pymadng_utils.mad import AcceleratorMadInterface

from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES
from tests.support.acd_barrier import acd_barrier_s
from tests.support.lhc import (
    AC_DIPOLE_DRIVEN_TUNES,
    AC_DIPOLE_MARKER,
    get_twiss,
    lhc_acd_barrier_s,
    lhc_model_details,
)
from tests.support.truth import model_details_for, simulated_mixed_reference_from_model
from tmom_recon import ACDipoleConfig, ModelDetails, MomentumReference


@dataclass(frozen=True)
class ContractScenario:
    """One tracked machine condition with MAD-NG-only reconstruction inputs."""

    machine: str
    sequence_file: Path
    delta_p: float
    noise_std: float
    pt: float
    data: pd.DataFrame
    tracking_line: Any
    measurement_tws: pd.DataFrame
    model_details: ModelDetails
    nominal_details: ModelDetails
    reference: MomentumReference
    barrier_s: float | None
    acd: ACDipoleConfig


_MACHINES = ("psb", "lhcb1", "b1_120cm_crossing")


def _psb_measurement_twiss(accelerator, delta_p: float) -> pd.DataFrame:
    """Recreate the independent MAD-NG input used by the passing PSB contract."""
    interface = AcceleratorMadInterface(accelerator)
    interface.observe()
    return interface.run_twiss(deltap=delta_p, coupling=True)


@pytest.fixture()
def contract_scenario(
    request: pytest.FixtureRequest, data_dir, psb_tracking_setup, acd_tracking_setup
):
    """Build a named on/off-momentum scenario for one contract test.

    Parameters are ``(machine, delta_p)``.  Keeping the condition in the test
    id makes an integration failure actionable without inspecting test code.
    """
    machine, delta_p, *noise = request.param
    noise_std = float(noise[0]) if noise else 0.0
    delta_p = float(delta_p)

    if machine == "psb":
        setup = psb_tracking_setup(delta_p)
        # Use exactly the BPMs selected by pymadng-utils' accelerator pattern;
        # tracking also records ACD state markers and the non-BPM BPMT monitor.
        data = setup.measurement.data.loc[
            setup.measurement.data["name"].isin(setup.measurement.bpm_names)
        ].copy(deep=True)
        nominal_details = model_details_for(setup.machine.accelerator, pt=0.0)
        model_details = model_details_for(setup.machine.accelerator, pt=setup.measurement.pt)
        return ContractScenario(
            machine=machine,
            sequence_file=Path(setup.machine.accelerator.sequence_file),
            delta_p=delta_p,
            noise_std=noise_std,
            pt=setup.measurement.pt,
            data=data,
            tracking_line=setup.machine.xsuite_line,
            measurement_tws=_psb_measurement_twiss(setup.machine.accelerator, delta_p),
            model_details=model_details,
            nominal_details=nominal_details,
            reference=simulated_mixed_reference_from_model(nominal_details, data),
            barrier_s=acd_barrier_s(setup.machine.madng_model, ACD_ELEMENT),
            acd=ACDipoleConfig(ac_dipole_marker=ACD_ELEMENT, driven_tunes=DRIVEN_TUNES),
        )

    sequence_file = data_dir / "sequences" / f"{machine}.seq"
    # Xsuite lines consume hundreds of MiB.  Only the location contract uses
    # one; every other contract needs the tracked dataframe and MAD-NG inputs.
    include_line = request.node.path.name == "test_acd_location.py"
    setup = acd_tracking_setup(sequence_file, delta_p=delta_p, include_line=include_line)
    data = setup.data.copy(deep=True)
    nominal_details = lhc_model_details(sequence_file, delta_p=0.0)
    model_details = lhc_model_details(sequence_file, delta_p=delta_p)
    return ContractScenario(
        machine=machine,
        sequence_file=sequence_file,
        delta_p=delta_p,
        noise_std=noise_std,
        pt=model_details.pt,
        data=data,
        tracking_line=setup.baseline_line,
        measurement_tws=get_twiss(sequence_file, deltap=delta_p),
        model_details=model_details,
        nominal_details=nominal_details,
        reference=simulated_mixed_reference_from_model(nominal_details, data),
        barrier_s=lhc_acd_barrier_s(model_details.accelerator, model_details.pt),
        acd=ACDipoleConfig(
            ac_dipole_marker=AC_DIPOLE_MARKER,
            driven_tunes=AC_DIPOLE_DRIVEN_TUNES,
        ),
    )


def scenario_params(*delta_ps: float) -> list[Any]:
    """Return explicit ids for every requested machine/momentum condition."""
    return [
        pytest.param(
            (machine, delta_p),
            id=f"{machine}-dp_{delta_p:+.0e}",
            marks=pytest.mark.psb if machine == "psb" else pytest.mark.lhc,
        )
        for machine in _MACHINES
        for delta_p in delta_ps
    ]


def acd_scenario_params() -> list[Any]:
    """Return the clean/noisy on/off-momentum ACD matrix."""
    values = {"psb": 1e-3, "lhcb1": 4e-4, "b1_120cm_crossing": 4e-4}
    result = []
    for machine in _MACHINES:
        for delta_p in (0.0, values[machine]):
            for noise_std in (0.0, 1e-5):
                result.append(
                    pytest.param(
                        (machine, delta_p, noise_std),
                        id=f"{machine}-dp_{delta_p:+.0e}-noise_{noise_std:.0e}",
                        marks=pytest.mark.psb if machine == "psb" else pytest.mark.lhc,
                    )
                )
    return result


def off_momentum_scenario_params() -> list[Any]:
    """Return the established, resolvable off-momentum condition per machine."""
    return [
        pytest.param(("psb", 1e-3), id="psb-off_momentum", marks=pytest.mark.psb),
        pytest.param(("lhcb1", 4e-4), id="lhcb1-off_momentum", marks=pytest.mark.lhc),
        pytest.param(
            ("b1_120cm_crossing", 4e-4),
            id="b1_120cm_crossing-off_momentum",
            marks=pytest.mark.lhc,
        ),
    ]


def truth_and_reconstruction_for_plane(
    tracking_data: pd.DataFrame, result: pd.DataFrame, plane: str
) -> tuple[pd.Series, pd.Series]:
    """Return fully aligned truth/reconstruction values for one momentum plane."""
    from tests.support.assertions import merge_tracking_truth

    merged = merge_tracking_truth(tracking_data, result)
    truth = merged[f"p{plane}_true"]
    reconstructed = merged[f"p{plane}"]
    missing = ~(truth.notna() & reconstructed.notna())
    assert not missing.any(), f"p{plane} has {int(missing.sum())} undefined reconstruction rows"
    return truth, reconstructed


__all__ = [
    "ContractScenario",
    "acd_scenario_params",
    "off_momentum_scenario_params",
    "scenario_params",
    "truth_and_reconstruction_for_plane",
]
