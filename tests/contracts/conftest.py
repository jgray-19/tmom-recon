"""Shared machine scenarios for the canonical diagnostic contracts.

Tracking coordinates are truth only.  Every reconstruction input is produced
by MAD-NG through :class:`~tmom_recon.ModelDetails` or the measurement helper.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest
from pymadng_utils.mad import AcceleratorMadInterface
from pymadng_utils.model_creator.madng_utils import update_model_with_madng

from tests.psb_tracking import (
    ACD_ELEMENT,
    DRIVEN_TUNES,
    build_psb_tracking_setup,
)
from tests.support.acd_barrier import acd_barrier_s
from tests.support.fake_measurement import Acquisition, run_fake_measurement
from tests.support.lhc import (
    AC_DIPOLE_DRIVEN_TUNES,
    AC_DIPOLE_MARKER,
    get_twiss,
    lhc_acd_barrier_s,
    lhc_model_details,
)
from tmom_recon import ACDipoleConfig, ModelDetails, OpticsInput


@dataclass(frozen=True)
class ContractScenario:
    """One tracked machine condition with MAD-NG-only reconstruction inputs."""

    machine: str
    delta_p: float
    noise_std: float
    pt: float
    data: pd.DataFrame
    model_details: ModelDetails
    reference: pd.DataFrame
    orbit_mode: str
    measured_orbit_zero: pd.DataFrame
    optics: OpticsInput
    barrier_s: float | None
    acd: ACDipoleConfig
    marker_truth: pd.DataFrame
    workflow: str = "all"
    condition: str = "clean"


@dataclass(frozen=True)
class ScenarioSpec:
    """One explicit end-to-end test cell.

    The fixture intentionally does not turn unsupported tracking/measurement
    setup into a skip: construction failures are test failures.
    """

    machine: str
    workflow: str
    delta_p: float
    condition: str
    optics_origin: str = "model"
    frame_kind: str = "dynamic"

    @property
    def id(self) -> str:
        return (
            f"{self.workflow}-{self.machine}-dp_{self.delta_p:+.0e}-"
            f"{self.condition}-{self.optics_origin}-{self.frame_kind}"
        )


_MACHINES = ("psb", "lhcb1", "b1_120cm_crossing")
_OFF_MOMENTUM = {"psb": 1e-3, "lhcb1": 4e-4, "b1_120cm_crossing": 4e-4}


def _psb_measurement_twiss(accelerator, delta_p: float) -> pd.DataFrame:
    """Recreate the independent MAD-NG input used by the passing PSB contract."""
    interface = AcceleratorMadInterface(accelerator)
    interface.observe()
    return interface.run_twiss(deltap=delta_p, coupling=True)


def _measured_orbit_zero(data: pd.DataFrame) -> pd.DataFrame:
    """The per-BPM turn mean of an on-momentum acquisition.

    This is *not* the closed orbit (see :func:`_orbit_zero`), and reconstruction
    must not use it. The momentum estimator differences two measured orbits,
    so it needs a reference acquired the same way as the data it is given:
    otherwise the driven-mean bias cancels on neither side and, for the LHC's
    100-turn window, moves the estimate by more than the tolerance.
    """
    return data.groupby("name", sort=False, observed=False)[["x", "y"]].mean()


def _orbit_zero(twiss: pd.DataFrame) -> pd.DataFrame:
    """The setting-zero orbit a frame subtracts: the on-momentum closed orbit.

    A turn mean of *driven* data is not that orbit. Over the 100 flattop turns
    these fixtures track, the AC-dipole oscillation does not average away: for
    LHC B1 the mean differs from the closed orbit by 5.3e-5 m rms (4.0e-4 m
    peak), which enters the reconstruction as a static per-BPM 1.5e-6 rad error
    -- an order of magnitude above the reconstruction budget itself.
    """
    return twiss[["x", "y"]]


@pytest.fixture()
def contract_scenario(
    request: pytest.FixtureRequest, data_dir, psb_tracking_setup, acd_tracking_setup
):
    """Build a named on/off-momentum scenario for one contract test.

    Parameters are ``(machine, delta_p)``.  Keeping the condition in the test
    id makes an integration failure actionable without inspecting test code.
    """
    spec = request.param if isinstance(request.param, ScenarioSpec) else None
    machine, delta_p, *noise = (spec.machine, spec.delta_p) if spec is not None else request.param
    noise_std = (
        1e-5
        if spec is not None and spec.condition == "bpm_noise"
        else float(noise[0])
        if noise
        else 0.0
    )
    delta_p = float(delta_p)

    if machine == "psb":
        setup = psb_tracking_setup(delta_p)
        # Use exactly the BPMs selected by pymadng-utils' accelerator pattern;
        # tracking also records ACD state markers and the non-BPM BPMT monitor.
        tracking_data = setup.measurement.data.copy(deep=True)
        data = tracking_data.loc[
            setup.measurement.data["name"].isin(setup.measurement.bpm_names)
        ].copy(deep=True)
        model_details = ModelDetails(accelerator=setup.machine.accelerator, pt=setup.measurement.pt)
        psb_zero_twiss = _psb_measurement_twiss(setup.machine.accelerator, 0.0)
        psb_zero = psb_tracking_setup(0.0).measurement
        optics = OpticsInput()
        if spec is not None and spec.optics_origin == "measurement":
            measurement_dirs = request.getfixturevalue("psb_fake_measurement_dirs")
            # "all" reconstructs on the free optics; "acd" and "driven_all" both
            # put the reconstruction on the driven optics, so both must read the
            # driven measurement.
            driven = spec.workflow in ("acd", "driven_all")
            optics = OpticsInput(
                # This cell's own momentum: the phase in each directory was
                # measured against a model of the lattice at that momentum.
                measurement_dir=measurement_dirs["acd" if driven else "all"][setup.measurement.pt],
                # Free all-BPM reconstruction retains the free model phase: even
                # a perfect synthetic HIO phase carries estimator uncertainty and
                # must not weaken the existing exact-model contract. The driven
                # workflows consume the measured driven phase they actually need.
                sources=(
                    {"phase": "measurement", "dispersion": "measurement"}
                    if driven
                    else {"dispersion": "measurement"}
                ),
            )
        return ContractScenario(
            machine=machine,
            delta_p=delta_p,
            noise_std=noise_std,
            pt=setup.measurement.pt,
            data=data,
            model_details=model_details,
            reference=_orbit_zero(psb_zero_twiss),
            orbit_mode=spec.frame_kind if spec is not None else "dynamic",
            measured_orbit_zero=_measured_orbit_zero(
                psb_zero.data.loc[psb_zero.data["name"].isin(psb_zero.bpm_names)]
            ),
            optics=optics,
            barrier_s=acd_barrier_s(setup.machine.madng_model, ACD_ELEMENT),
            acd=ACDipoleConfig(ac_dipole_marker=ACD_ELEMENT, driven_tunes=DRIVEN_TUNES),
            marker_truth=tracking_data,
            workflow=spec.workflow if spec is not None else "all",
            condition=spec.condition if spec is not None else "clean",
        )

    sequence_file = data_dir / "sequences" / f"{machine}.seq"
    setup = acd_tracking_setup(sequence_file, delta_p=delta_p, state_markers=True)
    zero_twiss = get_twiss(sequence_file, deltap=0.0)
    zero = acd_tracking_setup(sequence_file, delta_p=0.0)
    tracking_data = setup.data.copy(deep=True)
    # The ACD state markers are tracking truth, not BPM measurements.
    data = tracking_data.loc[
        ~tracking_data["name"].astype(str).str.endswith(("_BEFORE", "_AFTER"))
    ].copy(deep=True)
    model_details = lhc_model_details(sequence_file, delta_p=delta_p)
    return ContractScenario(
        machine=machine,
        delta_p=delta_p,
        noise_std=noise_std,
        pt=model_details.pt,
        data=data,
        model_details=model_details,
        reference=_orbit_zero(zero_twiss),
        orbit_mode=spec.frame_kind if spec is not None else "dynamic",
        measured_orbit_zero=_measured_orbit_zero(zero.data),
        optics=OpticsInput(),
        barrier_s=lhc_acd_barrier_s(model_details.accelerator, model_details.pt),
        acd=ACDipoleConfig(
            ac_dipole_marker=AC_DIPOLE_MARKER,
            driven_tunes=AC_DIPOLE_DRIVEN_TUNES,
        ),
        marker_truth=tracking_data,
        workflow=spec.workflow if spec is not None else "all",
        condition=spec.condition if spec is not None else "clean",
    )


def _madng_natural_tunes(twiss: Any) -> tuple[float, float]:
    """Fractional natural tunes of a MAD-NG twiss, at whatever momentum it holds."""
    headers = {str(key).lower(): value for key, value in twiss.headers.items()}
    return float(headers["q1"]) % 1, float(headers["q2"]) % 1


def _psb_momentum_model_dir(
    *,
    psb_model_dir,
    accelerator,
    natural_tunes: tuple[float, float],
    delta_p: float,
    tmp_path_factory,
):
    """A model dir whose twiss is exported at *delta_p*, like psb_md's phase model.

    ``psb_md.acd_workflow`` builds one runtime model per orbit group at that
    group's ``dpp`` before running phase analysis, so omc3 measures each momentum
    against a model of the lattice at that momentum. A single pt=0 model shared
    across momenta puts the measured phase and its own reference at different
    expansion points.

    ``tunes`` is the lattice's *own* off-momentum tune, so the creator's tune
    match is a no-op. Passing the on-momentum values instead would steer the
    quadrupoles to undo the chromatic tune shift and fabricate a lattice error
    that the machine does not have.
    """
    target = tmp_path_factory.mktemp(f"psb-model-dp{delta_p:+.0e}".replace(".", "p"))
    shutil.copytree(psb_model_dir, target, dirs_exist_ok=True)
    update_model_with_madng(
        accelerator,
        target,
        tunes=[*natural_tunes],
        drv_tunes=list(DRIVEN_TUNES),
        deltap=delta_p,
        convert_to_madx=True,
    )
    return target


@pytest.fixture(scope="session")
def psb_fake_measurement_dirs(psb_model_dir, tmp_path_factory):
    """Per-momentum HIO runs, mirroring psb_md's per-orbit phase analysis.

    Keyed by ``pt`` so each contract cell reads the phase measured at its own
    momentum. See :mod:`tests.support.fake_measurement` for why pooling them
    into one optics run is wrong.
    """
    acquisitions = []
    for delta_p in (-_OFF_MOMENTUM["psb"], 0.0, _OFF_MOMENTUM["psb"]):
        # Match the real psb_md analysis window. A 1k-turn diagnostic trace is
        # enough for model-optics reconstruction but not for precision phase
        # measurement by Harpy.
        setup = build_psb_tracking_setup(psb_model_dir, delta_p, flattop_turns=8000)
        data = setup.measurement.data.loc[
            setup.measurement.data["name"].isin(setup.measurement.bpm_names)
        ].copy()
        natural_tunes = _madng_natural_tunes(setup.machine.madng_twiss)
        acquisitions.append(
            Acquisition(
                data=data,
                pt=setup.measurement.pt,
                model_dir=_psb_momentum_model_dir(
                    psb_model_dir=psb_model_dir,
                    accelerator=setup.machine.accelerator,
                    natural_tunes=natural_tunes,
                    delta_p=delta_p,
                    tmp_path_factory=tmp_path_factory,
                ),
                natural_tunes=natural_tunes,
                # The AC dipole drives at a fixed frequency, so the driven tune
                # does not follow the momentum the way the natural one does.
                driven_tunes=DRIVEN_TUNES,
            )
        )
    return run_fake_measurement(
        acquisitions,
        output_dir=tmp_path_factory.mktemp("psb-fake-measurement"),
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


def machine_reconstruction_params(
    machine: str,
    *,
    workflows: tuple[str, ...] = ("all", "acd"),
    conditions: tuple[str, ...] = ("clean", "bpm_noise"),
    optics_origins: tuple[str, ...] = ("model",),
    frame_kinds: tuple[str, ...] = ("dynamic", "absolute"),
) -> list[Any]:
    """Return one machine's matrix without parametrising over machine type."""
    if machine not in _MACHINES:
        raise ValueError(f"Unsupported machine {machine!r}")
    return [
        pytest.param(
            ScenarioSpec(machine, workflow, delta_p, condition, optics_origin, frame_kind),
            id=ScenarioSpec(machine, workflow, delta_p, condition, optics_origin, frame_kind).id,
        )
        for workflow in workflows
        for delta_p in (-_OFF_MOMENTUM[machine], 0.0, _OFF_MOMENTUM[machine])
        for condition in conditions
        for optics_origin in optics_origins
        for frame_kind in frame_kinds
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
    "ScenarioSpec",
    "acd_scenario_params",
    "machine_reconstruction_params",
    "off_momentum_scenario_params",
    "scenario_params",
    "truth_and_reconstruction_for_plane",
]
