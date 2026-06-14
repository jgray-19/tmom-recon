"""AC-dipole fit pipeline and public entry point."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from tmom_recon.data.config import POSITION_STD_DEV
from tmom_recon.lattice.core import (
    get_rng,
    inject_noise_xy,
    remove_closed_orbit,
    validate_input,
)
from tmom_recon.lattice.names import normalise_measurement_names, normalise_twiss_index
from tmom_recon.optics import AMPLITUDE_COLUMNS, DISPERSION_COLUMNS, PHASE_COLUMNS

from .bpm_reconstruction import (
    _normalise_supplied_tune,
    _restore_reference_momenta,
    prepare_direct_bpm_reconstruction,
)
from .cleaning import _clean_ac_dipole_states, _combine_state_tables
from .models import (
    ACDipoleBPMSelection,
    ACDipoleBPMWindow,
    ACDipoleFitResult,
    ACDipoleHarmonicFit,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)
from .selection import resolve_name, select_ac_dipole_bpm_window
from .transport import (
    build_tracked_state_table,
    reconstruct_bpm_momentum_from_cleaned_acd,
)

if TYPE_CHECKING:
    from .madng_driver import ACDipoleMadDriver

LOGGER = logging.getLogger(__name__)

SUMMARY_ATTR_NAME = "summary"

# Optics columns taken from a resolved twiss (see tmom_recon.optics.resolve_optics)
# when one is supplied; error/variance columns are copied alongside.
ACD_TWISS_OVERRIDE_COLUMNS = frozenset(PHASE_COLUMNS + AMPLITUDE_COLUMNS + DISPERSION_COLUMNS)
UPSTREAM_BPM_ROW_TYPE = "upstream_bpm"
MARKER_BEFORE_ROW_TYPE = "marker_before"
MARKER_AFTER_ROW_TYPE = "marker_after"
DOWNSTREAM_BPM_ROW_TYPE = "downstream_bpm"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_harmonic_fit_quality(
    *,
    label: str,
    turns: np.ndarray,
    observed: np.ndarray,
    fitted: np.ndarray,
) -> float:
    """Log RMSE, MAE, NRMSE, and R² for a harmonic fit.

    Args:
        label: Coordinate label used in log messages (e.g. ``"dpx"``).
        turns: Turn numbers (used only for the count in the log message).
        observed: Raw observed kick values per turn.
        fitted: Fitted kick values per turn.
    """
    observed_arr = np.asarray(observed, dtype=float)
    fitted_arr = np.asarray(fitted, dtype=float)
    valid = np.isfinite(observed_arr) & np.isfinite(fitted_arr)
    if not np.any(valid):
        LOGGER.info("AC-dipole %s fit quality: no valid samples", label)
        return float("nan")

    obs = observed_arr[valid]
    fit = fitted_arr[valid]
    residual = obs - fit
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))

    obs_centered = obs - np.mean(obs)
    sst = float(np.dot(obs_centered, obs_centered))
    r_squared = float("nan") if sst <= 0.0 else 1.0 - float(np.dot(residual, residual)) / sst

    peak_to_peak = float(np.ptp(obs))
    nrmse = float("nan") if peak_to_peak <= 0.0 else rmse / peak_to_peak

    LOGGER.info(
        "AC-dipole %s fit quality over %d turns: RMSE=%.3e rad, MAE=%.3e rad, NRMSE=%.3e, R^2=%.6f",
        label,
        int(np.count_nonzero(valid)),
        rmse,
        mae,
        nrmse,
        r_squared,
    )
    return r_squared


def _log_harmonic_fit_parameters(
    *, label: str, reference_tune: float, fit: ACDipoleHarmonicFit
) -> None:
    """Log the parameters of a fitted harmonic.

    Args:
        label: Coordinate label (e.g. ``"dpx"``).
        reference_tune: The driven tune used as the fit seed.
        fit: The fitted :class:`ACDipoleHarmonicFit`.
    """
    LOGGER.info(
        "AC-dipole %s harmonic fit: reference tune=%.6f, fitted tune=%.6f, "
        "amplitude=%.3e rad, phase=%.6f rad, offset=%.3e rad",
        label,
        reference_tune,
        fit.tune,
        fit.amplitude,
        fit.phase,
        fit.offset,
    )


# ---------------------------------------------------------------------------
# Summary DataFrame assembly
# ---------------------------------------------------------------------------


def _merge_primary_bpm_results(
    upstream_frame: pd.DataFrame,
    downstream_frame: pd.DataFrame,
    *,
    selection: ACDipoleBPMSelection,
    marker_name: str,
) -> pd.DataFrame:
    """Merge upstream and downstream per-turn momentum estimates into one DataFrame.

    Args:
        upstream_frame: Reconstructed-momentum DataFrame for the primary upstream BPM.
        downstream_frame: Reconstructed-momentum DataFrame for the primary downstream BPM.
        selection: The primary upstream/downstream BPM pair.
        marker_name: Name of the AC-dipole marker element.

    Returns:
        DataFrame joined on ``turn`` with BPM metadata and per-plane momentum columns.
    """
    result = upstream_frame[["turn", "px", "py"]].rename(
        columns={"px": "px_bpm_upstream", "py": "py_bpm_upstream"}
    )
    result["bpm_upstream"] = selection.upstream
    result["bpm_downstream"] = selection.downstream
    result["acd_marker"] = marker_name
    result["acd_element"] = marker_name
    downstream_out = downstream_frame[["turn", "px", "py"]].rename(
        columns={"px": "px_bpm_downstream", "py": "py_bpm_downstream"}
    )
    return result.merge(downstream_out, on="turn", how="inner")


def _build_state_rows(
    *,
    row_type: str,
    name: str,
    turns: np.ndarray,
    x: np.ndarray | pd.Series,
    px: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    py: np.ndarray | pd.Series,
    var_x: np.ndarray | pd.Series | float | None = None,
    var_px: np.ndarray | pd.Series | float | None = None,
    var_y: np.ndarray | pd.Series | float | None = None,
    var_py: np.ndarray | pd.Series | float | None = None,
) -> pd.DataFrame:
    n_turns = len(turns)

    def _variance_array(value):
        if value is None:
            return np.full(n_turns, np.nan, dtype=float)
        if np.isscalar(value):
            return np.full(n_turns, float(value), dtype=float)  # ty:ignore[invalid-argument-type]
        return np.asarray(value, dtype=float)

    return pd.DataFrame(
        {
            "row_type": row_type,
            "name": name,
            "turn": turns.astype(int, copy=False),
            "x": np.asarray(x, dtype=float),
            "px": np.asarray(px, dtype=float),
            "y": np.asarray(y, dtype=float),
            "py": np.asarray(py, dtype=float),
            "var_x": _variance_array(var_x),
            "var_px": _variance_array(var_px),
            "var_y": _variance_array(var_y),
            "var_py": _variance_array(var_py),
        }
    )


def _build_marker_state_rows(
    *,
    turns: np.ndarray,
    marker_name: str,
    model: ACDipoleMadDriver,
    before_state: ACDipoleStateEstimate,
    after_state: ACDipoleStateEstimate,
) -> pd.DataFrame:
    before_name = model.accelerator.acd_marker_name("before")
    after_name = model.accelerator.acd_marker_name("after")

    return pd.concat(
        [
            _build_state_rows(
                row_type=MARKER_BEFORE_ROW_TYPE,
                name=before_name,
                turns=turns,
                x=before_state.state.x,
                px=before_state.state.px,
                y=before_state.state.y,
                py=before_state.state.py,
                var_x=before_state.var_x,
                var_px=before_state.var_px,
                var_y=before_state.var_y,
                var_py=before_state.var_py,
            ),
            _build_state_rows(
                row_type=MARKER_AFTER_ROW_TYPE,
                name=after_name,
                turns=turns,
                x=after_state.state.x,
                px=after_state.state.px,
                y=after_state.state.y,
                py=after_state.state.py,
                var_x=after_state.var_x,
                var_px=after_state.var_px,
                var_y=after_state.var_y,
                var_py=after_state.var_py,
            ),
        ],
        ignore_index=True,
    )


def _build_output_metadata(
    *,
    marker_name: str,
    window: ACDipoleBPMWindow,
    dpx_fit: ACDipoleHarmonicFit,
    dpy_fit: ACDipoleHarmonicFit,
    pt_used: float,
) -> dict[str, object]:
    """Build the single-source metadata dict used for both TFS headers and attrs.

    Args:
        marker_name: Name of the AC-dipole marker element.
        window: The selected BPM window.
        dpx_fit: Horizontal kick fit.
        dpy_fit: Vertical kick fit.
        pt_used: The MAD-NG pt value that was used.

    Returns:
        Dict suitable for use as both TFS headers and ``result.attrs``.
    """
    return {
        "acd_marker": marker_name,
        "acd_element": marker_name,
        "bpm_upstream": window.primary.upstream,
        "bpm_downstream": window.primary.downstream,
        "bpms_upstream_used": ",".join(window.upstream),
        "bpms_downstream_used": ",".join(window.downstream),
        "dpx_tune": dpx_fit.tune,
        "dpx_amplitude": dpx_fit.amplitude,
        "dpx_phase": dpx_fit.phase,
        "dpx_offset": dpx_fit.offset,
        "dpy_tune": dpy_fit.tune,
        "dpy_amplitude": dpy_fit.amplitude,
        "dpy_phase": dpy_fit.phase,
        "dpy_offset": dpy_fit.offset,
        "pt_used": pt_used,
        "kick_model_type": "constant_envelope",
    }


def _assemble_result_dataframe(
    result: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    fit: ACDipoleFitResult,
    selection: ACDipoleBPMSelection,
    marker_name: str,
    pt_est: float,
) -> pd.DataFrame:
    """Populate all output columns on the per-turn summary DataFrame.

    Args:
        result: Base DataFrame from :func:`_merge_primary_bpm_results`.
        tws: Twiss DataFrame (used to restore closed-orbit reference momenta).
        fit: The full fit result from :func:`_fit_ac_dipole_from_frames`.
        selection: The primary upstream/downstream BPM pair.
        marker_name: Name of the AC-dipole marker element.
        pt_est: Estimated MAD-NG pt.

    Returns:
        *result* with all ACD output columns populated.
    """
    for side, bpm_name in (("upstream", selection.upstream), ("downstream", selection.downstream)):
        for plane in ("px", "py"):
            col = f"{plane}_bpm_{side}"
            result[col] = _restore_reference_momenta(
                result[col].to_numpy(dtype=float), tws, bpm_name, plane
            )

    for side, state in (
        ("upstream", fit.raw_upstream.state),
        ("downstream", fit.raw_downstream.state),
    ):
        result[f"x_acd_{side}"] = state.x
        result[f"y_acd_{side}"] = state.y
        for plane in ("px", "py"):
            result[f"{plane}_acd_{side}"] = _restore_reference_momenta(
                getattr(state, plane), tws, marker_name, plane
            )

    result["dpx_rad"] = fit.dpx_raw
    result["dpy_rad"] = fit.dpy_raw
    result["dpx"] = fit.dpx_raw
    result["dpy"] = fit.dpy_raw
    result["dpx_fit_rad"] = fit.dpx_fit.fitted
    result["dpy_fit_rad"] = fit.dpy_fit.fitted

    result["x_acd_cleaned"] = fit.cleaned_upstream.state.x
    result["y_acd_cleaned"] = fit.cleaned_upstream.state.y
    for side, state in (
        ("upstream", fit.cleaned_upstream.state),
        ("downstream", fit.cleaned_downstream.state),
    ):
        result[f"x_acd_{side}_cleaned"] = state.x
        result[f"y_acd_{side}_cleaned"] = state.y
        for plane in ("px", "py"):
            result[f"{plane}_acd_{side}_cleaned"] = _restore_reference_momenta(
                getattr(state, plane), tws, marker_name, plane
            )

    result["pt_used"] = pt_est
    return result


# ---------------------------------------------------------------------------
# Core fitting pipeline
# ---------------------------------------------------------------------------


def _fit_ac_dipole_from_frames(
    *,
    selection: ACDipoleBPMSelection,
    window: ACDipoleBPMWindow,
    marker_name: str,
    model: ACDipoleMadDriver,
    smooth_lambda: float,
    dpx_tune: float,
    dpy_tune: float,
    upstream_frames: dict[str, pd.DataFrame],
    downstream_frames: dict[str, pd.DataFrame],
) -> ACDipoleFitResult:
    """Run the full AC-dipole fit and cleaning pipeline.

    Args:
        selection: Primary upstream/downstream BPM pair.
        window: Full BPM window used for multi-BPM averaging.
        marker_name: Name of the AC-dipole marker element.
        model: MAD-NG driver used for state transport.
        smooth_lambda: Second-difference regularisation strength.
        dpx_tune: Fractional horizontal driven tune.
        dpy_tune: Fractional vertical driven tune.
        upstream_frames: Per-BPM reconstructed-momentum DataFrames for upstream BPMs.
        downstream_frames: Per-BPM reconstructed-momentum DataFrames for downstream BPMs.

    Returns:
        An :class:`ACDipoleFitResult` with all intermediate and final quantities.
    """
    summary = _merge_primary_bpm_results(
        upstream_frames[selection.upstream],
        downstream_frames[selection.downstream],
        selection=selection,
        marker_name=marker_name,
    )
    turns = summary["turn"].to_numpy(dtype=float)

    upstream_tables = [
        build_tracked_state_table(
            upstream_frames[bpm_name],
            model,
            source_name=bpm_name,
            marker_name=marker_name,
            direction=1,
        )
        for bpm_name in window.upstream
    ]
    downstream_tables = [
        build_tracked_state_table(
            downstream_frames[bpm_name],
            model,
            source_name=bpm_name,
            marker_name=marker_name,
            direction=-1,
        )
        for bpm_name in window.downstream
    ]

    raw_upstream = _combine_state_tables(turns, upstream_tables)
    raw_downstream = _combine_state_tables(turns, downstream_tables)
    dpx_raw = raw_downstream.state.px - raw_upstream.state.px
    dpy_raw = raw_downstream.state.py - raw_upstream.state.py

    cleaning = _clean_ac_dipole_states(
        turns,
        upstream_tables,
        downstream_tables,
        dpx_tune=dpx_tune,
        dpy_tune=dpy_tune,
        smooth_lambda=smooth_lambda,
    )

    r2s = {}
    for label, tune, fit, raw in (
        ("dpx", dpx_tune, cleaning.dpx_fit, dpx_raw),
        ("dpy", dpy_tune, cleaning.dpy_fit, dpy_raw),
    ):
        _log_harmonic_fit_parameters(label=label, reference_tune=tune, fit=fit)
        r2s[label] = _log_harmonic_fit_quality(
            label=label, turns=turns, observed=raw, fitted=fit.fitted
        )

    return ACDipoleFitResult(
        summary=summary,
        turns=turns,
        raw_upstream=raw_upstream,
        raw_downstream=raw_downstream,
        cleaned_upstream=cleaning.upstream,
        cleaned_downstream=cleaning.downstream,
        dpx_raw=dpx_raw,
        dpy_raw=dpy_raw,
        dpx_fit=cleaning.dpx_fit,
        dpy_fit=cleaning.dpy_fit,
        dpx_r2=r2s["dpx"],
        dpy_r2=r2s["dpy"],
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PreparedACDInputs:
    """Optics-independent inputs for AC-dipole reconstruction.

    Produced once by :func:`prepare_ac_dipole_inputs` and consumed (possibly
    many times, with different model twisses) by
    :func:`reconstruct_from_prepared`. Holding the frozen, noise-injected,
    name-normalised, window-filtered measurement data here means a re-run for a
    new optics never re-validates, re-injects noise, or re-selects BPMs.

    Note:
        ``data`` still carries the closed orbit; it is removed per
        reconstruction because the closed-orbit reference depends on the twiss.

    Attributes:
        data: Frozen measurement data, filtered to the window BPMs.
        model: MAD-NG driver used for state transport.
        marker_name: Resolved AC-dipole marker element name.
        window: Selected upstream/downstream BPM window.
        selection: Primary upstream/downstream BPM pair (``window.primary``).
        lattice_names: Lattice element names (used to normalise twiss indices).
        lattice_bpm_order: Window BPMs in lattice order.
        dpx_tune_frac, dpy_tune_frac: Driven tunes folded into ``(0, 0.5)``.
        smooth_lambda: Second-difference regularisation strength.
        use_immediate_neighbors: Use immediate lattice neighbors for BPM
            momentum reconstruction instead of pi/2-phase neighbors.
    """

    data: pd.DataFrame
    model: ACDipoleMadDriver
    marker_name: str
    window: ACDipoleBPMWindow
    selection: ACDipoleBPMSelection
    lattice_names: list[str]
    lattice_bpm_order: list[str]
    dpx_tune_frac: float
    dpy_tune_frac: float
    smooth_lambda: float
    use_immediate_neighbors: bool


def prepare_ac_dipole_inputs(
    orig_data: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    ac_dipole_marker: str,
    model: ACDipoleMadDriver,
    dpx_tune: float,
    dpy_tune: float,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
    smooth_lambda: float = 1,
    inject_noise: bool | float = True,
    rng: np.random.Generator | None = None,
    use_immediate_neighbors_for_bpms: bool = False,
) -> PreparedACDInputs:
    """Run the optics-independent half of the AC-dipole reconstruction once.

    Validates and copies *orig_data*, injects measurement noise a single time,
    normalises BPM/twiss names, selects the AC-dipole BPM window, filters the
    data to that window, and folds the driven tunes into ``(0, 0.5)``. The
    result feeds :func:`reconstruct_from_prepared`, which may be called
    repeatedly with different model twisses.

    Args:
        orig_data: Turn-by-turn BPM measurement DataFrame (``name, turn, x, y``
            and optionally ``var_x, var_y``).
        tws: A model twiss used only to determine which measured BPMs are present
            in the optics (the BPM set is assumed stable across updates).
        ac_dipole_marker: Lattice element name at which the kick is modelled.
        model: MAD-NG driver providing the lattice element ordering.
        dpx_tune: Horizontal driven tune (folded into ``(0, 0.5)``).
        dpy_tune: Vertical driven tune.
        bpm_upstream: Optional explicit upstream BPM name.
        bpm_downstream: Optional explicit downstream BPM name.
        smooth_lambda: Second-difference regularisation strength.
        inject_noise: If ``True``, inject Gaussian position noise at the standard
            BPM resolution. Pass a float for a custom std dev [m]; ``False``
            disables it.
        rng: Optional NumPy random generator for reproducible noise injection.
        use_immediate_neighbors_for_bpms: Use immediate lattice neighbors instead
            of pi/2-phase neighbors for BPM momentum reconstruction.

    Returns:
        A :class:`PreparedACDInputs` bundle.
    """
    validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")

    rng = get_rng(rng)
    if inject_noise is not False:
        noise_std = POSITION_STD_DEV if inject_noise is True else float(inject_noise)
        data = inject_noise_xy(data, rng, noise_std=noise_std)

    lattice_names = [str(name) for name in model.twiss_elements.index]
    marker_name = resolve_name(ac_dipole_marker, lattice_names)
    data = normalise_measurement_names(data, lattice_names)
    tws = normalise_twiss_index(tws, lattice_names)

    measured_bpm_names = [str(name) for name in data["name"].unique()]
    available_bpm_names = [name for name in measured_bpm_names if name in set(tws.index)]
    window = select_ac_dipole_bpm_window(
        model.twiss_elements,
        marker_name,
        available_bpm_names,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
    )
    LOGGER.info(
        "Reconstructing AC-dipole kick at %s using upstream BPMs %s and downstream BPMs %s",
        marker_name,
        window.upstream,
        window.downstream,
    )

    data = data[data["name"].isin(available_bpm_names)].copy(deep=True)
    lattice_bpm_order = [
        str(name) for name in model.twiss_elements.index if str(name) in set(available_bpm_names)
    ]

    return PreparedACDInputs(
        data=data,
        model=model,
        marker_name=marker_name,
        window=window,
        selection=window.primary,
        lattice_names=lattice_names,
        lattice_bpm_order=lattice_bpm_order,
        dpx_tune_frac=_normalise_supplied_tune("dpx", float(dpx_tune)),
        dpy_tune_frac=_normalise_supplied_tune("dpy", float(dpy_tune)),
        smooth_lambda=smooth_lambda,
        use_immediate_neighbors=use_immediate_neighbors_for_bpms,
    )


def reconstruct_from_prepared(
    prepared: PreparedACDInputs,
    tws: pd.DataFrame,
    *,
    resolved_tws: pd.DataFrame | None = None,
) -> tfs.TfsDataFrame:
    """Reconstruct AC-dipole kicks for a given model twiss from prepared inputs.

    The optics-dependent half of the pipeline: removes the closed orbit,
    reconstructs BPM momenta, transports states to and from the AC-dipole marker
    (re-tracking through MAD-NG, so it stays correct when the lattice magnets
    have changed), and assembles the output. Safe to call repeatedly with
    different *tws* / *resolved_tws* for the same *prepared* data.

    Args:
        prepared: Output of :func:`prepare_ac_dipole_inputs`.
        tws: Model twiss for this reconstruction (optics + tune headers ``q1`` /
            ``q2``). Used for closed-orbit removal and state transport.
        resolved_tws: Optional resolved twiss from
            :func:`tmom_recon.optics.resolve_optics`. When provided, its optics,
            uncertainty and variance columns (and tune headers) override the
            model values for BPM-pair selection and the initial ``px``/``py``
            estimate.

    Returns:
        A :class:`tfs.TfsDataFrame` with four long-form state row groups
        (upstream BPM, marker before, marker after, downstream BPM) and the wide
        per-turn summary in ``attrs["summary"]``.
    """
    model = prepared.model
    marker_name = prepared.marker_name
    window = prepared.window
    selection = prepared.selection
    smooth_lambda = prepared.smooth_lambda
    dpx_tune_frac = prepared.dpx_tune_frac
    dpy_tune_frac = prepared.dpy_tune_frac

    data = prepared.data.copy(deep=True)
    tws = normalise_twiss_index(tws, prepared.lattice_names)

    tws_bpm = tws.reindex(prepared.lattice_bpm_order).copy(deep=True)
    data = remove_closed_orbit(data, tws)

    if resolved_tws is not None:
        resolved = normalise_twiss_index(resolved_tws, prepared.lattice_names)
        common = tws_bpm.index.intersection(resolved.index)
        override_cols = [
            col
            for col in resolved.columns
            if col in ACD_TWISS_OVERRIDE_COLUMNS or col.endswith("_err") or col.endswith("_var")
        ]
        for col in override_cols:
            tws_bpm.loc[common, col] = resolved.loc[common, col]
        resolved_headers = dict(getattr(resolved, "headers", {}) or {})
        headers = getattr(tws_bpm, "headers", None)
        if isinstance(headers, dict):
            for key in ("q1", "q2", "mu1_total_var", "mu2_total_var"):
                if key in resolved_headers:
                    headers[key] = resolved_headers[key]

    bpm_order = [str(name) for name in tws_bpm.index]
    bpm_index = {name: idx for idx, name in enumerate(bpm_order)}
    upstream_frames, downstream_frames = prepare_direct_bpm_reconstruction(
        data,
        tws_bpm,
        window=window,
        bpm_index=bpm_index,
        pt_est=model.pt,
        use_immediate_neighbors=prepared.use_immediate_neighbors,
    )

    fit: ACDipoleFitResult = _fit_ac_dipole_from_frames(
        selection=selection,
        window=window,
        marker_name=marker_name,
        model=model,
        smooth_lambda=smooth_lambda,
        dpx_tune=dpx_tune_frac,
        dpy_tune=dpy_tune_frac,
        upstream_frames=upstream_frames,
        downstream_frames=downstream_frames,
    )

    result = _assemble_result_dataframe(
        fit.summary.copy(deep=True),
        tws,
        fit=fit,
        selection=selection,
        marker_name=marker_name,
        pt_est=model.pt,
    )

    px_up_clean, py_up_clean = reconstruct_bpm_momentum_from_cleaned_acd(
        tws,
        model,
        bpm_name=selection.upstream,
        cleaned_acd=fit.cleaned_upstream,
        marker_name=marker_name,
        direction=-1,
    )
    px_dn_clean, py_dn_clean = reconstruct_bpm_momentum_from_cleaned_acd(
        tws,
        model,
        bpm_name=selection.downstream,
        cleaned_acd=fit.cleaned_downstream,
        marker_name=marker_name,
        direction=1,
    )
    result["px_bpm_upstream_cleaned"] = px_up_clean
    result["py_bpm_upstream_cleaned"] = py_up_clean
    result["px_bpm_downstream_cleaned"] = px_dn_clean
    result["py_bpm_downstream_cleaned"] = py_dn_clean

    before_state = ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            fit.cleaned_upstream.state.x,
            _restore_reference_momenta(fit.cleaned_upstream.state.px, tws, marker_name, "px"),
            fit.cleaned_upstream.state.y,
            _restore_reference_momenta(fit.cleaned_upstream.state.py, tws, marker_name, "py"),
            fit.cleaned_upstream.state.t,
            fit.cleaned_upstream.state.pt,
        ),
        var_x=fit.cleaned_upstream.var_x,
        var_px=fit.cleaned_upstream.var_px,
        var_y=fit.cleaned_upstream.var_y,
        var_py=fit.cleaned_upstream.var_py,
    )
    after_state = ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            fit.cleaned_downstream.state.x,
            _restore_reference_momenta(fit.cleaned_downstream.state.px, tws, marker_name, "px"),
            fit.cleaned_downstream.state.y,
            _restore_reference_momenta(fit.cleaned_downstream.state.py, tws, marker_name, "py"),
            fit.cleaned_downstream.state.t,
            fit.cleaned_downstream.state.pt,
        ),
        var_x=fit.cleaned_downstream.var_x,
        var_px=fit.cleaned_downstream.var_px,
        var_y=fit.cleaned_downstream.var_y,
        var_py=fit.cleaned_downstream.var_py,
    )
    marker_rows = _build_marker_state_rows(
        turns=fit.turns,
        marker_name=marker_name,
        model=model,
        before_state=before_state,
        after_state=after_state,
    )

    metadata = _build_output_metadata(
        marker_name=marker_name,
        window=window,
        dpx_fit=fit.dpx_fit,
        dpy_fit=fit.dpy_fit,
        pt_used=model.pt,
    )

    tfs_headers = {f"ACD_{k.removeprefix('acd_').upper()}": v for k, v in metadata.items()}

    state_rows = pd.concat(
        [
            _build_state_rows(
                row_type=UPSTREAM_BPM_ROW_TYPE,
                name=selection.upstream,
                turns=fit.turns,
                x=upstream_frames[selection.upstream]["x"],
                px=px_up_clean,
                y=upstream_frames[selection.upstream]["y"],
                py=py_up_clean,
                var_x=upstream_frames[selection.upstream].get("var_x"),
                var_px=upstream_frames[selection.upstream].get("var_px"),
                var_y=upstream_frames[selection.upstream].get("var_y"),
                var_py=upstream_frames[selection.upstream].get("var_py"),
            ),
            marker_rows,
            _build_state_rows(
                row_type=DOWNSTREAM_BPM_ROW_TYPE,
                name=selection.downstream,
                turns=fit.turns,
                x=downstream_frames[selection.downstream]["x"],
                px=px_dn_clean,
                y=downstream_frames[selection.downstream]["y"],
                py=py_dn_clean,
                var_x=downstream_frames[selection.downstream].get("var_x"),
                var_px=downstream_frames[selection.downstream].get("var_px"),
                var_y=downstream_frames[selection.downstream].get("var_y"),
                var_py=downstream_frames[selection.downstream].get("var_py"),
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    if state_rows.duplicated(["name", "turn"]).any():
        duplicates = state_rows.loc[
            state_rows.duplicated(["name", "turn"], keep=False), ["name", "turn"]
        ]
        raise ValueError(f"AC-dipole output contains duplicate (name, turn) rows:\n{duplicates}")

    result_out = tfs.TfsDataFrame(
        state_rows.sort_values(["turn", "row_type", "name"]).reset_index(drop=True),
        headers=tfs_headers,
    )

    result_out.attrs.update(metadata)
    result_out.attrs[SUMMARY_ATTR_NAME] = result
    result_out.attrs["acd_marker_name"] = marker_name
    result_out.attrs["smooth_lambda"] = smooth_lambda
    result_out.attrs["qx_drive"] = dpx_tune_frac
    result_out.attrs["qy_drive"] = dpy_tune_frac
    result_out.attrs["dpx_fit"] = dataclasses.asdict(fit.dpx_fit)
    result_out.attrs["dpy_fit"] = dataclasses.asdict(fit.dpy_fit)
    result_out.attrs["dpx_r2"] = fit.dpx_r2
    result_out.attrs["dpy_r2"] = fit.dpy_r2

    return result_out


def calculate_ac_dipole_momentum(
    orig_data: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    ac_dipole_marker: str,
    model: ACDipoleMadDriver,
    dpx_tune: float,
    dpy_tune: float,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
    smooth_lambda: float = 1,
    inject_noise: bool | float = True,
    rng: np.random.Generator | None = None,
    use_immediate_neighbors_for_bpms: bool = False,
    resolved_tws: pd.DataFrame | None = None,
) -> tfs.TfsDataFrame:
    """Reconstruct AC-dipole kicks and constrained BPM momenta in one pass.

    Thin wrapper that runs :func:`prepare_ac_dipole_inputs` once and then
    :func:`reconstruct_from_prepared`. Use those two directly when the same
    measurement data must be reconstructed repeatedly for different model
    twisses (see :class:`tmom_recon.reconstruction.ACDipolePzGenerator`).

    Args:
        orig_data: Turn-by-turn BPM measurement DataFrame with columns
            ``name, turn, x, y`` and optionally ``var_x, var_y``.
        tws: Model Twiss DataFrame indexed by element name with optics columns
            and tune headers ``q1`` / ``q2``. Used for closed-orbit removal,
            state transport, and as fallback optics when no measurement folder
            is given.
        ac_dipole_marker: Lattice element name at which the kick is modelled.
        model: MAD-NG driver used for state transport and Jacobian computation.
        dpx_tune: Horizontal driven tune (integer or fractional; fractional part
            is used and folded into ``(0, 0.5)``).
        dpy_tune: Vertical driven tune.
        bpm_upstream: Optional explicit upstream BPM name. Closest upstream BPM
            is selected automatically when omitted.
        bpm_downstream: Optional explicit downstream BPM name.
        smooth_lambda: Second-difference regularisation strength for the
            marker-side momentum solve.
        inject_noise: If ``True``, inject Gaussian position noise at the standard
            BPM resolution. Pass a float to use a custom noise std dev [m].
            ``False`` disables noise injection.
        rng: Optional NumPy random generator for reproducible noise injection.
        use_immediate_neighbors_for_bpms: If ``True``, use immediate lattice
            neighbors instead of pi/2-phase neighbors for BPM momentum
            reconstruction.
        resolved_tws: Optional resolved twiss from
            :func:`tmom_recon.optics.resolve_optics`. When provided, its optics,
            uncertainty and variance columns (and tune headers) override the
            model values for BPM-pair selection and the initial ``px``/``py``
            estimate.

    Returns:
        A :class:`tfs.TfsDataFrame` with four long-form state row groups:
        upstream BPM, marker before, marker after, and downstream BPM.
        The wide per-turn summary is stored in ``attrs["summary"]``.
    """
    prepared = prepare_ac_dipole_inputs(
        orig_data,
        tws,
        ac_dipole_marker=ac_dipole_marker,
        model=model,
        dpx_tune=dpx_tune,
        dpy_tune=dpy_tune,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        smooth_lambda=smooth_lambda,
        inject_noise=inject_noise,
        rng=rng,
        use_immediate_neighbors_for_bpms=use_immediate_neighbors_for_bpms,
    )
    return reconstruct_from_prepared(prepared, tws, resolved_tws=resolved_tws)


__all__ = [
    "PreparedACDInputs",
    "calculate_ac_dipole_momentum",
    "prepare_ac_dipole_inputs",
    "reconstruct_from_prepared",
]
