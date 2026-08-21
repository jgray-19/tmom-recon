"""Typed PSB simulation scenarios shared by integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tmom_recon.acd.madng_driver import ACDipoleMadDriver


@dataclass(frozen=True)
class SimulatedMachine:
    """The accelerator objects and optics used to create or reconstruct data."""

    accelerator: Any
    xsuite_line: Any
    madng_model: ACDipoleMadDriver
    madng_twiss: Any


@dataclass(frozen=True)
class SimulatedMeasurement:
    """Tracked observations, optics, and their known momentum truth."""

    data: Any
    delta_p: float
    pt: float
    bpm_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PSBScenario:
    """Explicit source-of-truth boundary for one PSB tracking experiment."""

    machine: SimulatedMachine
    measurement: SimulatedMeasurement
    bend_strengths: dict[str, float]
    quad_strengths: dict[str, float]

    def copy(self) -> PSBScenario:
        """Copy mutable tabular data while retaining expensive model objects."""
        return PSBScenario(
            machine=SimulatedMachine(
                accelerator=self.machine.accelerator,
                xsuite_line=self.machine.xsuite_line,
                madng_model=self.machine.madng_model,
                madng_twiss=self.machine.madng_twiss.copy(deep=True),
            ),
            measurement=SimulatedMeasurement(
                data=self.measurement.data.copy(deep=True),
                delta_p=self.measurement.delta_p,
                pt=self.measurement.pt,
                bpm_names=self.measurement.bpm_names,
            ),
            bend_strengths=dict(self.bend_strengths),
            quad_strengths=dict(self.quad_strengths),
        )
