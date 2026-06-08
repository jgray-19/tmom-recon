from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .reconstruction import calculate_ac_dipole_momentum

if TYPE_CHECKING:
    import pandas as pd

    from .madng_driver import ACDipoleMadDriver


@dataclass(frozen=True)
class ACDipoleConfig:
    """Configuration for reusing AC-dipole-cleaned BPM momenta.

    Attributes:
        ac_dipole_marker: Lattice element name at which the kick is modeled.
        model: MAD-NG-backed transport driver used to move states between the
            BPMs and the marker.
        dpx_tune: Horizontal driven tune used as the harmonic fit seed.
        dpy_tune: Vertical driven tune used as the harmonic fit seed.
        bpm_upstream: Optional explicit upstream BPM name. If omitted, the
            closest upstream BPM around the marker is selected automatically.
        bpm_downstream: Optional explicit downstream BPM name. If omitted, the
            closest downstream BPM around the marker is selected automatically.
        smooth_lambda: Regularization strength for the marker-side momentum
            smoothing solve.
        tune_knobs_file: Optional knobs file applied when constructing the MAD
            driver.
        corrector_knobs_file: Optional orbit-correction knobs file applied when
            constructing the MAD driver.
    """

    ac_dipole_marker: str
    model: ACDipoleMadDriver
    dpx_tune: float
    dpy_tune: float
    bpm_upstream: str | None = None
    bpm_downstream: str | None = None
    smooth_lambda: float = 1.0
    tune_knobs_file: Path | None = None
    corrector_knobs_file: Path | None = None


def run_ac_dipole_reconstruction(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    config: ACDipoleConfig,
) -> pd.DataFrame:
    """Run AC-dipole reconstruction once on a measurement frame."""

    data_for_acd = data.copy(deep=True)
    if "var_x" not in data_for_acd.columns:
        data_for_acd["var_x"] = 1.0
    if "var_y" not in data_for_acd.columns:
        data_for_acd["var_y"] = 1.0

    return calculate_ac_dipole_momentum(
        data_for_acd,
        tws,
        ac_dipole_marker=config.ac_dipole_marker,
        model=config.model,
        dpx_tune=config.dpx_tune,
        dpy_tune=config.dpy_tune,
        bpm_upstream=config.bpm_upstream,
        bpm_downstream=config.bpm_downstream,
        smooth_lambda=config.smooth_lambda,
        inject_noise=False,
    )


def _get_acd_bpm_name(acd_result: pd.DataFrame, attr_name: str, column_name: str) -> str:
    return str(acd_result.attrs.get(attr_name, acd_result[column_name].iloc[0]))


def _apply_cleaned_bpm_override(
    result: pd.DataFrame,
    acd_result: pd.DataFrame,
    *,
    bpm_name: str,
    px_col: str,
    py_col: str,
) -> None:
    mask = result["name"].astype(str) == bpm_name
    if not mask.any():
        return

    side = acd_result[["turn", px_col, py_col]].rename(columns={px_col: "px", py_col: "py"})
    side = side.set_index("turn")
    turns = result.loc[mask, "turn"].to_numpy()
    result.loc[mask, "px"] = side.reindex(turns)["px"].to_numpy(dtype=float)
    result.loc[mask, "py"] = side.reindex(turns)["py"].to_numpy(dtype=float)


def apply_precomputed_ac_dipole_bpm_overrides_inplace(
    result: pd.DataFrame,
    acd_result: pd.DataFrame,
    config: ACDipoleConfig | None = None,
) -> pd.DataFrame:
    """Patch px/py for the BPMs around the AC dipole using cleaned ACD estimates.

    The function replaces ``px``/``py`` values in ``result`` at the selected
    upstream/downstream BPMs by matching on ``(name, turn)``.
    """
    bpm_upstream = _get_acd_bpm_name(acd_result, "bpm_upstream", "bpm_upstream")
    bpm_downstream = _get_acd_bpm_name(acd_result, "bpm_downstream", "bpm_downstream")

    # Persist resolved AC-dipole selection metadata for downstream consumers.
    result.attrs["ac_dipole_marker"] = (
        config.ac_dipole_marker if config is not None else acd_result.attrs.get("acd_marker")
    )
    result.attrs["ac_dipole_bpm_upstream"] = bpm_upstream
    result.attrs["ac_dipole_bpm_downstream"] = bpm_downstream
    result.attrs["ac_dipole_smooth_lambda"] = float(
        config.smooth_lambda
        if config is not None
        else acd_result.attrs.get("smooth_lambda", np.nan)
    )

    _apply_cleaned_bpm_override(
        result,
        acd_result,
        bpm_name=bpm_upstream,
        px_col="px_bpm_upstream_cleaned",
        py_col="py_bpm_upstream_cleaned",
    )
    _apply_cleaned_bpm_override(
        result,
        acd_result,
        bpm_name=bpm_downstream,
        px_col="px_bpm_downstream_cleaned",
        py_col="py_bpm_downstream_cleaned",
    )
    return acd_result


def apply_ac_dipole_bpm_overrides_inplace(
    result: pd.DataFrame,
    data: pd.DataFrame,
    tws: pd.DataFrame,
    config: ACDipoleConfig,
    *,
    acd_result: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run ACD reconstruction if needed, then apply its adjacent-BPM overrides."""
    if acd_result is None:
        acd_result = run_ac_dipole_reconstruction(data, tws, config)
    return apply_precomputed_ac_dipole_bpm_overrides_inplace(
        result=result,
        acd_result=acd_result,
        config=config,
    )
