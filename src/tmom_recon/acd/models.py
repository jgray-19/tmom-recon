from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ACDipoleBPMSelection:
    upstream: str
    downstream: str


@dataclass(frozen=True)
class ACDipoleBPMWindow:
    upstream: tuple[str, ...]
    downstream: tuple[str, ...]

    @property
    def primary(self) -> ACDipoleBPMSelection:
        return ACDipoleBPMSelection(
            upstream=self.upstream[0],
            downstream=self.downstream[0],
        )


@dataclass(frozen=True)
class ACDipoleHarmonicFit:
    tune: float
    amplitude: float
    phase: float
    offset: float
    fitted: np.ndarray


@dataclass(frozen=True)
class ACDipoleStateSeries:
    x: np.ndarray
    px: np.ndarray
    y: np.ndarray
    py: np.ndarray


@dataclass(frozen=True)
class ACDipoleStateEstimate:
    state: ACDipoleStateSeries
    var_x: np.ndarray
    var_px: np.ndarray
    var_y: np.ndarray
    var_py: np.ndarray
