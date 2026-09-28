"""Internal closed-orbit reference handling for reconstruction workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OrbitMode = Literal["dynamic", "absolute"]
_STATE = ("x", "px", "y", "py")


def normalize_closed_orbit(
    closed_orbit_at_zero: pd.DataFrame, bpm_names: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Validate and align measured zero-momentum BPM positions."""
    orbit = pd.DataFrame(closed_orbit_at_zero).copy()
    orbit.columns = orbit.columns.astype(str).str.lower()
    if "name" not in orbit:
        if str(orbit.index.name).lower() != "name":
            raise ValueError("closed_orbit_at_zero must have a 'name' column or index")
        orbit["name"] = orbit.index.astype(str)
    names = orbit["name"].astype(str).str.upper()
    duplicates = names[names.duplicated(keep=False)].unique().tolist()
    if duplicates:
        raise ValueError(
            "closed_orbit_at_zero contains duplicate case-insensitive BPM names: "
            f"{sorted(duplicates)}"
        )
    missing_columns = [column for column in ("x", "y") if column not in orbit]
    if missing_columns:
        raise ValueError(f"closed_orbit_at_zero is missing required column(s) {missing_columns}")
    if not np.isfinite(orbit[["x", "y"]].to_numpy(dtype=float)).all():
        raise ValueError("closed_orbit_at_zero contains non-finite x/y values")
    orbit = orbit.assign(name=names).set_index("name")[["x", "y"]].astype(float)
    required = pd.Index([str(name).upper() for name in bpm_names])
    missing = required[~required.isin(orbit.index)].unique().tolist()
    if missing:
        raise ValueError(f"closed_orbit_at_zero is missing reconstructed BPM(s) {sorted(missing)}")
    return orbit.loc[required.unique()].copy()


def validate_orbit_mode(orbit_mode: str) -> OrbitMode:
    if orbit_mode not in ("dynamic", "absolute"):
        raise ValueError("orbit_mode must be 'dynamic' or 'absolute'")
    return orbit_mode


@dataclass(frozen=True)
class OrbitReference:
    measured: pd.DataFrame
    restored: pd.DataFrame

    def subtract(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        names = result["name"].astype(str).str.upper()
        result[["x", "y"]] -= self.measured.loc[names, ["x", "y"]].to_numpy()
        return result

    def restore(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        names = result["name"].astype(str).str.upper()
        result[list(_STATE)] += self.restored.loc[names, list(_STATE)].to_numpy()
        return result


def build_orbit_reference(
    closed_orbit_at_zero: pd.DataFrame,
    orbit_mode: str,
    zero_twiss: pd.DataFrame,
    bpm_names: list[str] | tuple[str, ...],
) -> OrbitReference:
    """Build the mode-specific state restored after zero-orbit subtraction."""
    mode = validate_orbit_mode(orbit_mode)
    measured = normalize_closed_orbit(closed_orbit_at_zero, bpm_names)
    model = pd.DataFrame(zero_twiss).copy()
    model.columns = model.columns.astype(str).str.lower()
    model.index = model.index.astype(str).str.upper()
    missing_columns = [column for column in _STATE if column not in model]
    if missing_columns:
        raise ValueError(f"zero-momentum model twiss is missing {missing_columns}")
    required = pd.Index([str(name).upper() for name in bpm_names]).unique()
    missing = required[~required.isin(model.index)].tolist()
    if missing:
        raise ValueError(f"zero-momentum model twiss is missing BPM(s) {sorted(missing)}")
    restored = model.loc[:, list(_STATE)].astype(float).copy()
    if mode == "absolute":
        restored.loc[required, ["x", "y"]] = measured.loc[required, ["x", "y"]]
    return OrbitReference(measured=measured, restored=restored)
