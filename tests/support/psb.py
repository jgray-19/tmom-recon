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
    xsuite_twiss: Any
    madng_twiss: Any


@dataclass(frozen=True)
class SimulatedMeasurement:
    """Tracked observations and their known momentum truth."""

    data: Any
    truth: Any
    delta_p: float
    pt: float


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
                xsuite_twiss=_copy_if_possible(self.machine.xsuite_twiss),
                madng_twiss=self.machine.madng_twiss.copy(deep=True),
            ),
            measurement=SimulatedMeasurement(
                data=self.measurement.data.copy(deep=True),
                truth=self.measurement.truth.copy(deep=True),
                delta_p=self.measurement.delta_p,
                pt=self.measurement.pt,
            ),
            bend_strengths=dict(self.bend_strengths),
            quad_strengths=dict(self.quad_strengths),
        )


def _copy_if_possible(value: Any):
    copy_method = getattr(value, "copy", None)
    if callable(copy_method):
        try:
            return copy_method(deep=True)
        except TypeError:
            return copy_method()
    return value
