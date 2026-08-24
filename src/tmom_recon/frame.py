"""The measured orbit-zero coordinate frame used by momentum reconstruction.

The frame owns the ordering that callers previously had to reproduce by hand:
subtract the one measured orbit at setting zero in dynamic planes, estimate the
momentum from those coordinates, reconstruct with dispersion, and restore only
the closed-orbit components retained by the chosen frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

Plane = Literal["x", "y"]

__all__ = ["ReconstructionFrame"]


def _planes(values: Iterable[str]) -> tuple[Plane, ...]:
    requested = {str(value).lower() for value in values}
    unknown = requested - {"x", "y"}
    if unknown:
        raise ValueError(f"Unknown dynamic plane(s): {sorted(unknown)}")
    return tuple(plane for plane in ("x", "y") if plane in requested)


def _normalise(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    result = frame.copy()
    if "name" in result.columns:
        result = result.set_index("name")
    result.index = pd.Index(result.index.astype(str), name="name")
    if result.index.has_duplicates:
        raise ValueError(f"{label} contains duplicate BPM names")
    return result.sort_index()


@dataclass(frozen=True)
class ReconstructionFrame:
    """A reconstruction coordinate frame anchored to one measured orbit zero.

    ``orbit_zero`` supplies measured BPM positions. ``dynamic_planes`` are
    translated by those positions before any momentum estimate. Retained planes
    stay absolute and therefore require their fitted closed-orbit momentum in
    ``fitted_momenta``. Momentum values passed to reconstruction are offsets from
    this frame; the frame origin is always exactly zero.
    """

    orbit_zero: pd.DataFrame
    dynamic_planes: tuple[Plane, ...] = ()
    fitted_momenta: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        origin = _normalise(self.orbit_zero, label="orbit_zero")
        missing = [column for column in ("x", "y") if column not in origin]
        if missing:
            raise ValueError(f"orbit_zero is missing position column(s) {missing}")
        origin = origin[["x", "y"]].astype(float)
        if not np.isfinite(origin.to_numpy()).all():
            raise ValueError("orbit_zero contains non-finite positions")
        dynamic = _planes(self.dynamic_planes)

        momenta = None
        retained = tuple(plane for plane in ("x", "y") if plane not in dynamic)
        if retained:
            if self.fitted_momenta is None:
                raise ValueError(
                    "Retained closed-orbit planes require explicitly fitted "
                    "reference momenta; no model or zero-angle fallback is allowed."
                )
            momenta = _normalise(self.fitted_momenta, label="fitted_momenta")
            if not momenta.index.equals(origin.index):
                raise ValueError("fitted_momenta and orbit_zero must cover the same BPMs")
            required = [f"p{plane}" for plane in retained]
            missing = [column for column in required if column not in momenta]
            if missing:
                raise ValueError(f"fitted_momenta is missing column(s) {missing}")
            momenta = momenta[required].astype(float)
            if not np.isfinite(momenta.to_numpy()).all():
                raise ValueError("fitted_momenta contains non-finite values")

        object.__setattr__(self, "orbit_zero", origin)
        object.__setattr__(self, "dynamic_planes", dynamic)
        object.__setattr__(self, "fitted_momenta", momenta)

    @property
    def closed_orbit(self) -> pd.DataFrame:
        """Reference state subtracted/restored by the all-BPM reconstruction."""
        state = self.orbit_zero.copy()
        for plane in self.dynamic_planes:
            state[plane] = 0.0
        if self.fitted_momenta is not None:
            for column in self.fitted_momenta:
                state[column] = self.fitted_momenta[column]
        for plane in self.dynamic_planes:
            state[f"p{plane}"] = 0.0
        return state

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Translate raw BPM data into this frame before momentum estimation."""
        result = data.copy(deep=True)
        names = result["name"].astype(str)
        missing = pd.Index(names.unique()).difference(self.orbit_zero.index)
        if len(missing):
            raise ValueError(f"orbit_zero is missing BPM(s) {sorted(missing.tolist())}")
        for plane in self.dynamic_planes:
            # fmt: off
            result[plane] = (
                result[plane].astype(float)
                - names.map(self.orbit_zero[plane]).astype(float)
            )
            # fmt: on
        return result
