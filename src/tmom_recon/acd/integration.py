"""Higher-level integration helpers for the AC-dipole reconstruction pipeline.

Provides :class:`ACDipoleConfig` for bundling reconstruction parameters and
convenience functions for running reconstruction and applying cleaned BPM
momentum overrides to an existing result DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .reconstruction import SUMMARY_ATTR_NAME, calculate_ac_dipole_momentum

if TYPE_CHECKING:
    import pandas as pd

    from .madng_driver import ACDipoleMadDriver


@dataclass(frozen=True)
class ACDipoleConfig:
    """Configuration bundle for AC-dipole reconstruction.

    Attributes:
        ac_dipole_marker: Lattice element name at which the kick is modelled.
        model: MAD-NG-backed transport driver used to move states between BPMs
            and the marker.
        dpx_tune: Horizontal driven tune used as the harmonic fit seed.
        dpy_tune: Vertical driven tune used as the harmonic fit seed.
        bpm_upstream: Optional explicit upstream BPM name. If omitted, the
            closest upstream BPM around the marker is selected automatically.
        bpm_downstream: Optional explicit downstream BPM name. If omitted,
            the closest downstream BPM is selected automatically.
        smooth_lambda: Regularisation strength for the marker-side momentum
            smoothing solve.
    """

    ac_dipole_marker: str
    model: ACDipoleMadDriver
    dpx_tune: float
    dpy_tune: float
    bpm_upstream: str | None = None
    bpm_downstream: str | None = None
    smooth_lambda: float = 1.0


def ensure_position_variances(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *data* with default unit ``var_x`` / ``var_y`` if absent."""
    data_for_acd = data.copy(deep=True)
    if "var_x" not in data_for_acd.columns:
        data_for_acd["var_x"] = 1.0
    if "var_y" not in data_for_acd.columns:
        data_for_acd["var_y"] = 1.0
    return data_for_acd


def run_ac_dipole_reconstruction(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    config: ACDipoleConfig,
    *,
    resolved_tws: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run AC-dipole reconstruction once on a measurement DataFrame.

    Adds default unit variances for ``var_x`` / ``var_y`` if they are absent,
    then delegates to :func:`calculate_ac_dipole_momentum` with noise injection
    disabled.

    Args:
        data: Turn-by-turn BPM measurement DataFrame with columns
            ``name, turn, x, y``.
        tws: Twiss DataFrame indexed by element name.
        config: Reconstruction configuration.

    Returns:
        A :class:`tfs.TfsDataFrame` with the reconstruction output — see
        :func:`calculate_ac_dipole_momentum` for the column and header layout.
    """
    data_for_acd = ensure_position_variances(data)
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
        resolved_tws=resolved_tws,
    )


def _summary_rows(acd_result: pd.DataFrame) -> pd.DataFrame:
    """Return the wide per-turn summary from an ACD reconstruction result.

    The long-form result from :func:`calculate_ac_dipole_momentum` carries its
    wide per-turn summary (including the ``*_cleaned`` momentum columns) in
    ``attrs["summary"]``. Older callers passed the summary frame directly, so
    fall back to filtering a ``"row_type"`` column when no summary attr exists.

    Args:
        acd_result: ACD reconstruction output (long-form with a ``"summary"``
            attr, or a bare summary DataFrame).

    Returns:
        The wide per-turn summary DataFrame.
    """
    summary = acd_result.attrs.get(SUMMARY_ATTR_NAME)
    if summary is not None:
        return summary
    if "row_type" not in acd_result.columns:
        return acd_result
    return acd_result.loc[acd_result["row_type"].fillna("summary") == "summary"].copy(deep=True)


def _apply_cleaned_bpm_override(
    result: pd.DataFrame,
    acd_result: pd.DataFrame,
    *,
    bpm_name: str,
    px_col: str,
    py_col: str,
) -> None:
    """Patch ``px``/``py`` in *result* for a single BPM using cleaned ACD values.

    Matches on turn number; rows with no matching ACD turn are left unchanged.

    Args:
        result: DataFrame to patch in-place (must have ``"name"`` and ``"turn"``
            columns).
        acd_result: ACD reconstruction output DataFrame.
        bpm_name: Name of the BPM to patch.
        px_col: Column in *acd_result* carrying the cleaned horizontal momentum.
        py_col: Column in *acd_result* carrying the cleaned vertical momentum.
    """
    mask = result["name"].astype(str) == bpm_name
    if not mask.any():
        return
    side = (
        _summary_rows(acd_result)[["turn", px_col, py_col]]
        .rename(columns={px_col: "px", py_col: "py"})
        .set_index("turn")
    )
    turns = result.loc[mask, "turn"].to_numpy()
    result.loc[mask, "px"] = side.reindex(turns)["px"].to_numpy(dtype=float)
    result.loc[mask, "py"] = side.reindex(turns)["py"].to_numpy(dtype=float)


def apply_precomputed_ac_dipole_bpm_overrides_inplace(
    result: pd.DataFrame,
    acd_result: pd.DataFrame,
    config: ACDipoleConfig | None = None,
) -> pd.DataFrame:
    """Patch ``px``/``py`` for the BPMs adjacent to the AC dipole using cleaned estimates.

    Replaces ``px``/``py`` values in *result* at the selected upstream/downstream
    BPMs by matching on ``(name, turn)``, then records resolution metadata in
    ``result.attrs``.

    Args:
        result: BPM-level momentum DataFrame to patch in-place.
        acd_result: Pre-computed ACD reconstruction output from
            :func:`run_ac_dipole_reconstruction`.
        config: Optional :class:`ACDipoleConfig` used to fill ``result.attrs``
            with the AC-dipole marker name and smooth lambda.

    Returns:
        *acd_result* (unchanged), so callers can chain further ACD inspection.
    """
    bpm_upstream = acd_result.attrs["bpm_upstream"]
    bpm_downstream = acd_result.attrs["bpm_downstream"]

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
    """Run ACD reconstruction if needed, then apply adjacent-BPM momentum overrides.

    If *acd_result* is already available (e.g. from a previous call) it is
    reused directly, skipping the reconstruction step.

    Args:
        result: BPM-level momentum DataFrame to patch in-place.
        data: Turn-by-turn BPM measurement DataFrame passed to
            :func:`run_ac_dipole_reconstruction` if *acd_result* is ``None``.
        tws: Twiss DataFrame passed to :func:`run_ac_dipole_reconstruction`.
        config: Reconstruction configuration.
        acd_result: Optional pre-computed ACD result. If ``None``, reconstruction
            is run automatically.

    Returns:
        *acd_result* (the reconstruction output), so callers can inspect it.
    """
    if acd_result is None:
        acd_result = run_ac_dipole_reconstruction(data, tws, config)
    return apply_precomputed_ac_dipole_bpm_overrides_inplace(
        result=result,
        acd_result=acd_result,
        config=config,
    )
