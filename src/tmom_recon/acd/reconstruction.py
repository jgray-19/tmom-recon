"""AC-dipole fit pipeline and public entry point."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from tmom_recon.lattice.core import (
    remove_closed_orbit,
    validate_input,
)
from tmom_recon.lattice.names import normalise_measurement_names
from tmom_recon.optics import (
    ALPHA_COLUMNS,
    BETA_COLUMNS,
    DISPERSION_COLUMNS,
    PHASE_COLUMNS,
    SECOND_ORDER_DISPERSION_COLUMNS,
)

from .bpm_reconstruction import (
    _normalise_supplied_tune,
    prepare_direct_bpm_reconstruction,
)
from .cleaning import _clean_ac_dipole_states, _combine_state_tables
from .models import (
    ACDipoleBPMSelection,
    ACDipoleBPMWindow,
    ACDipoleFitResult,
    ACDipoleHarmonicFit,
    ACDipoleSide,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)
from .selection import resolve_name, select_ac_dipole_bpm_window
from .transport import (
    build_tracked_state_table,
    transport_marker_state_to_bpm,
)

if TYPE_CHECKING:
    from .madng_driver import ACDipoleMadDriver

LOGGER = logging.getLogger(__name__)

SUMMARY_ATTR_NAME = "summary"

# Optics columns taken from a resolved twiss (see tmom_recon.optics.resolve_optics)
# when one is supplied; error/variance columns are copied alongside.
ACD_TWISS_OVERRIDE_COLUMNS = frozenset(
    PHASE_COLUMNS
    + BETA_COLUMNS
    + ALPHA_COLUMNS
    + DISPERSION_COLUMNS
    + SECOND_ORDER_DISPERSION_COLUMNS
)


def _normalise_observed_twiss(tws: pd.DataFrame) -> pd.DataFrame:
    """Return an observed twiss with upper-case string element names."""
    out = tws.copy(deep=True)
    out.index = out.index.astype(str).str.upper()
    return out


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_harmonic_fit_quality(
    *,
    label: str,
    turns: np.ndarray,
    observed: np.ndarray,
    fitted: np.ndarray,
    variances: np.ndarray | None = None,
) -> float:
    """Log RMSE, MAE, NRMSE, and weighted R² for a harmonic fit.

    Args:
        label: Coordinate label used in log messages (e.g. ``"dpx"``).
        turns: Turn numbers (used only for the count in the log message).
        observed: Raw observed kick values per turn.
        fitted: Fitted kick values per turn.
        variances: Optional per-turn variances. When supplied, the primary R²
            uses the same inverse-variance weights as the harmonic fit.
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
    unweighted_r_squared = (
        float("nan") if sst <= 0.0 else 1.0 - float(np.dot(residual, residual)) / sst
    )

    r_squared = unweighted_r_squared
    if variances is not None:
        variance_arr = np.asarray(variances, dtype=float)[valid]
        valid_weights = np.isfinite(variance_arr) & (variance_arr > 0.0)
        if np.any(valid_weights):
            obs_w = obs[valid_weights]
            fit_w = fit[valid_weights]
            weights = 1.0 / variance_arr[valid_weights]
            mean_w = float(np.average(obs_w, weights=weights))
            weighted_sst = float(np.sum(weights * (obs_w - mean_w) ** 2))
            weighted_sse = float(np.sum(weights * (obs_w - fit_w) ** 2))
            r_squared = float("nan") if weighted_sst <= 0.0 else 1.0 - weighted_sse / weighted_sst

    peak_to_peak = float(np.ptp(obs))
    nrmse = float("nan") if peak_to_peak <= 0.0 else rmse / peak_to_peak

    LOGGER.info(
        "AC-dipole %s fit quality over %d turns: RMSE=%.3e rad, MAE=%.3e rad, "
        "NRMSE=%.3e, weighted R^2=%.6f, unweighted R^2=%.6f",
        label,
        int(np.count_nonzero(valid)),
        rmse,
        mae,
        nrmse,
        r_squared,
        unweighted_r_squared,
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
) -> pd.DataFrame:
    """Merge the primary upstream/downstream per-turn momentum estimates on ``turn``.

    Identity metadata (marker name, primary BPM names) is *not* copied per row
    here; it lives once in the output headers/``attrs`` (see
    :func:`_build_output_metadata`).

    Args:
        upstream_frame: Reconstructed-momentum DataFrame for the primary upstream BPM.
        downstream_frame: Reconstructed-momentum DataFrame for the primary downstream BPM.

    Returns:
        DataFrame joined on ``turn`` with the per-plane BPM momentum columns.
    """
    upstream_out = upstream_frame[["turn", "px", "py"]].rename(
        columns={"px": "px_bpm_upstream", "py": "py_bpm_upstream"}
    )
    downstream_out = downstream_frame[["turn", "px", "py"]].rename(
        columns={"px": "px_bpm_downstream", "py": "py_bpm_downstream"}
    )
    return upstream_out.merge(downstream_out, on="turn", how="inner")


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
    model: ACDipoleMadDriver,
    marker_states: list[tuple[ACDipoleSide, ACDipoleStateEstimate]],
) -> pd.DataFrame:
    """Build marker-side output rows for the supplied state estimates.

    Args:
        turns: Turn numbers array.
        model: MAD-NG driver, used to resolve the marker element name per side.
        marker_states: ``(side, marker_state)`` pairs; each side supplies the row
            type and marker end. The estimate may be raw or cleaned.

    Returns:
        Concatenated marker state rows, one group per side.
    """
    return pd.concat(
        [
            _build_state_rows(
                row_type=side.marker_row_type,
                name=model.accelerator.acd_marker_name(side.marker_end),
                turns=turns,
                x=estimate.state.x,
                px=estimate.state.px,
                y=estimate.state.y,
                py=estimate.state.py,
                var_x=estimate.var_x,
                var_px=estimate.var_px,
                var_y=estimate.var_y,
                var_py=estimate.var_py,
            )
            for side, estimate in marker_states
        ],
        ignore_index=True,
    )


def _build_bpm_state_rows(
    side: ACDipoleSide,
    *,
    turns: np.ndarray,
    cleaned_bpm_state: ACDipoleStateSeries,
) -> pd.DataFrame:
    """Build the output rows for *side*'s primary BPM.

    Positions come from the measured/reconstructed primary frame; momenta are the
    cleaned marker state transported back to the BPM.

    Args:
        side: The side whose primary BPM is emitted.
        turns: Turn numbers array.
        cleaned_bpm_state: Cleaned state transported from the marker to the primary BPM.

    Returns:
        A state-row DataFrame for the primary BPM.
    """
    frame = side.primary_frame
    return _build_state_rows(
        row_type=side.bpm_row_type,
        name=side.primary,
        turns=turns,
        x=frame["x"],
        px=cleaned_bpm_state.px,
        y=frame["y"],
        py=cleaned_bpm_state.py,
        var_x=frame.get("var_x"),
        var_px=frame.get("var_px"),
        var_y=frame.get("var_y"),
        var_py=frame.get("var_py"),
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
    *,
    fit: ACDipoleFitResult,
    pt_est: float,
) -> pd.DataFrame:
    """Populate all output columns on the per-turn summary DataFrame.

    Args:
        result: Base DataFrame from :func:`_merge_primary_bpm_results`.
        fit: The full fit result from :func:`_fit_ac_dipole_from_frames`.
        pt_est: Estimated MAD-NG pt.

    Returns:
        *result* with all ACD output columns populated.
    """
    result["dpx_rad"] = fit.dpx_raw
    result["dpy_rad"] = fit.dpy_raw
    result["dpx_fit_rad"] = fit.dpx_fit.fitted - fit.dpx_fit.offset
    result["dpy_fit_rad"] = fit.dpy_fit.fitted - fit.dpy_fit.offset
    result["pt_used"] = pt_est
    return result


# ---------------------------------------------------------------------------
# Core fitting pipeline
# ---------------------------------------------------------------------------


def _tracked_marker_tables(side: ACDipoleSide, model: ACDipoleMadDriver) -> list[pd.DataFrame]:
    """Transport every window BPM on *side* to the marker.

    Args:
        side: The side whose BPM frames are transported.
        model: MAD-NG driver used for state transport.

    Returns:
        One tracked-state table per window BPM.
    """
    return [
        build_tracked_state_table(frame, model, source_name=bpm_name, direction=side.direction)
        for bpm_name, frame in side.bpm_frames.items()
    ]


def _fit_ac_dipole_from_frames(
    *,
    upstream_side: ACDipoleSide,
    downstream_side: ACDipoleSide,
    model: ACDipoleMadDriver,
    smooth_lambda: float,
    dpx_tune: float,
    dpy_tune: float,
) -> ACDipoleFitResult:
    """Run the full AC-dipole fit and cleaning pipeline.

    Args:
        upstream_side: Upstream side (window BPM frames + transport conventions).
        downstream_side: Downstream side.
        model: MAD-NG driver used for state transport.
        smooth_lambda: Second-difference regularisation strength.
        dpx_tune: Fractional horizontal driven tune.
        dpy_tune: Fractional vertical driven tune.

    Returns:
        An :class:`ACDipoleFitResult` with all intermediate and final quantities.
    """
    summary = _merge_primary_bpm_results(
        upstream_side.primary_frame,
        downstream_side.primary_frame,
    )
    turns = summary["turn"].to_numpy(dtype=float)

    upstream_tables = _tracked_marker_tables(upstream_side, model)
    downstream_tables = _tracked_marker_tables(downstream_side, model)

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
    for label, tune, fit, raw, variance in (
        (
            "dpx",
            dpx_tune,
            cleaning.dpx_fit,
            dpx_raw,
            raw_upstream.var_px + raw_downstream.var_px,
        ),
        (
            "dpy",
            dpy_tune,
            cleaning.dpy_fit,
            dpy_raw,
            raw_upstream.var_py + raw_downstream.var_py,
        ),
    ):
        _log_harmonic_fit_parameters(label=label, reference_tune=tune, fit=fit)
        r2s[label] = _log_harmonic_fit_quality(
            label=label,
            turns=turns,
            observed=raw,
            fitted=fit.fitted,
            variances=variance,
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
        use_immediate_neighbors_for_bpms: Use immediate lattice neighbors instead
            of pi/2-phase neighbors for BPM momentum reconstruction.

    Returns:
        A :class:`PreparedACDInputs` bundle.
    """
    validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")

    lattice_names = [str(name) for name in model.twiss_elements.index]
    marker_name = resolve_name(ac_dipole_marker, lattice_names)
    data = normalise_measurement_names(data, lattice_names)

    tws = _normalise_observed_twiss(tws)
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


class ACDipoleStateConsistencyError(ValueError):
    """The reconstructed BPM state disagrees with the model's prediction.

    Its own type, so a batch caller can act on *this* verdict -- excluding the
    acquisition it came from -- without catching unrelated ``ValueError``s or
    matching on the message. It is a rejection of one measurement, not a bug:
    the usual cause is an acquisition whose driven amplitude is too close to the
    BPM noise floor, and the correct response is to drop that file, never to
    widen the tolerance.

    Attributes:
        bpm_name: BPM the check failed at.
        coord: Coordinate that failed (``"x"`` or ``"px"``).
        max_residual: Largest per-turn residual [m or rad].
        state_amplitude: Amplitude of the predicted state [m or rad].
        tolerance: Tolerance that was applied.
        records: Verdict record per coordinate checked before the rejection,
            in the same shape as the ``acd_state_consistency`` attribute an
            accepted reconstruction carries.
    """

    def __init__(
        self,
        message: str,
        *,
        bpm_name: str,
        coord: str,
        max_residual: float,
        state_amplitude: float,
        tolerance: float,
    ) -> None:
        super().__init__(message)
        self.bpm_name = bpm_name
        self.coord = coord
        self.max_residual = max_residual
        self.state_amplitude = state_amplitude
        self.tolerance = tolerance
        self.records: list[dict[str, object]] = []


def _check_bpm_state_consistency(
    frame: pd.DataFrame, bpm_name: str, predicted_state: ACDipoleStateSeries
) -> list[dict[str, object]]:
    """Check that the reconstructed BPM state is consistent with the predicted state.

    Returns one record per checked coordinate describing the verdict -- the
    residual, the tolerance applied and whether it passed -- so a caller can
    report *why* an acquisition was rejected, or plot the disagreement, instead
    of only learning that it was. The records are produced whether or not the
    check passes, and whether or not the escape hatch is set; raising is still
    the default behaviour on failure.

    Args:
        frame: Reconstructed-momentum DataFrame for a BPM.
        bpm_name: Name of the BPM.
        predicted_state: Predicted state at the BPM from the model.

    Raises:
        ValueError: If *bpm_name* has no data in *frame*.
        ACDipoleStateConsistencyError: If the reconstructed BPM state is not
            consistent with the predicted state.
    """
    bpm_frame = frame[frame["name"] == bpm_name]
    if bpm_frame.empty:
        raise ValueError(f"No data for BPM {bpm_name} in the reconstructed frame.")

    # Check that the reconstructed state matches the predicted state. The states
    # oscillate through zero, so a per-turn relative error is meaningless near the
    # zero-crossings; compare max absolute residuals against a tolerance. On
    # momentum a fixed 1e-4 m floor is comfortable: on the PSB ring-3 0 mm file the
    # measured guard residual is 5.8e-5 m in x (2.5e-5 in px). It is also
    # noise-robust -- injecting Gaussian reading noise at the realistic BPM floor
    # (static PSB table, sigma_x ~= 6e-5 m; no blank_acquisitions co-located) leaves
    # the x residual at 5.9e-5, and even 3x that noise (2e-4) only reaches 6.1e-5,
    # because the SVD cleaning and turn-averaging over the flat-top suppress
    # per-turn noise. So the floor's headroom is set by the model, not by noise.
    #
    # Off momentum the fixed 1e-4 m floor is over-tight. Studied on the three PSB
    # ring-3 files (0 / +6 mm / -6 mm radial steering): the driven state at the
    # primary BPM has amplitude ~4.6-7.0 mm, and the (already off-momentum-fitted)
    # closed-orbit model reproduces its mean off-momentum orbit only to a residual
    # of 1.0e-4 m (+6 mm) / 2.1e-4 m (-6 mm) in x -- a known, small model-vs-machine
    # orbit imperfection, not a bad reconstruction (px residual stays <=9e-6). The
    # -6 mm x residual therefore tripped the 1e-4 floor. Once the state amplitude
    # exceeds 1 mm, switch to a relative tolerance so the check scales with the
    # signal. 5% was initially chosen but left only 8% margin at -6 mm, and a
    # later run measured 2.318e-4 against the 2.30e-4 tolerance -- the model's
    # off-momentum orbit fidelity (~0.2 mm) genuinely sits that close to it. At
    # 10% the tolerance is 4.6-7.0e-4 m, which keeps ~2x headroom over the model
    # floor while still bounding gross reconstruction errors to <10% of the
    # driven amplitude.
    abs_floor = 1e-4  # Absolute tolerance floor for state consistency check
    rel_tolerance = 0.10  # Relative tolerance used once |state| exceeds 1 mm
    amplitude_threshold = 1e-3  # State amplitude above which the check goes relative
    records: list[dict[str, object]] = []
    for coord in ("x", "px"):
        reconstructed_value = bpm_frame[coord].values
        predicted_value = getattr(predicted_state, coord)
        max_residual = np.max(np.abs(reconstructed_value - predicted_value))
        state_amplitude = np.max(np.abs(predicted_value))
        if state_amplitude > amplitude_threshold:
            tolerance = max(abs_floor, rel_tolerance * state_amplitude)
        else:
            tolerance = abs_floor
        records.append(
            {
                "bpm": bpm_name,
                "coord": coord,
                "max_residual": float(max_residual),
                "rms_residual": float(
                    np.sqrt(np.mean((reconstructed_value - predicted_value) ** 2))
                ),
                "state_amplitude": float(state_amplitude),
                "tolerance": float(tolerance),
                "passed": bool(max_residual <= tolerance),
            }
        )
        if max_residual > tolerance:
            # Report the residual as a fraction of the state as well as in metres.
            # The absolute number alone is unreadable: below the 1 mm threshold the
            # floor is a *looser* test than the relative branch (at |state|=7e-4 the
            # 1e-4 floor is 14%, not 10%), so tripping it means the reconstruction is
            # genuinely worse than 10% of the driven signal -- not that the tolerance
            # is too tight. Note the fraction is frame-dependent: with the closed
            # orbit removed from the data (dynamic-part reconstruction) |state| is the
            # driven amplitude alone, so the same residual reads several times larger
            # than it would against an absolute state carrying the orbit.
            # DIAGNOSTIC ESCAPE HATCH. Set TMOM_RECON_IGNORE_ACD_STATE_GUARD=1 to
            # log the verdict instead of raising it. This exists so a *known
            # failing* reconstruction can still be plotted and looked at while
            # comparing cleaning methods -- it is never a way to obtain a
            # physics result. The guard's answer does not change; only whether
            # the acquisition is dropped.
            if os.environ.get("TMOM_RECON_IGNORE_ACD_STATE_GUARD") == "1":
                LOGGER.warning(
                    "ACD state-consistency guard IGNORED (diagnostic mode): %s at BPM %s "
                    "max|residual|=%.3e vs tolerance %.1e (%.1f%% of |state|=%.3e)",
                    coord,
                    bpm_name,
                    max_residual,
                    tolerance,
                    100 * max_residual / state_amplitude if state_amplitude else float("inf"),
                    state_amplitude,
                )
                continue
            error = ACDipoleStateConsistencyError(
                f"Reconstructed {coord} at BPM {bpm_name} does not match the predicted "
                f"value within tolerance {tolerance:.1e} (max|residual|={max_residual:.3e}, "
                f"|state|={state_amplitude:.3e}, "
                f"residual={100 * max_residual / state_amplitude if state_amplitude else float('inf'):.1f}% of |state|"
                f"{', below the 1 mm relative-branch threshold' if state_amplitude <= amplitude_threshold else ''})."
                " This rejects the acquisition, most often because its driven"
                " amplitude is too close to the BPM noise floor; drop the file"
                " rather than widening the tolerance.",
                bpm_name=bpm_name,
                coord=coord,
                max_residual=float(max_residual),
                state_amplitude=float(state_amplitude),
                tolerance=float(tolerance),
            )
            # The records travel with the rejection so a batch caller can build
            # the same table for rejected acquisitions as for accepted ones.
            error.records = records
            raise error
    return records


def reconstruct_from_prepared(
    prepared: PreparedACDInputs,
    tws: pd.DataFrame,
    *,
    closed_orbit_tws: pd.DataFrame,
    dispersion_tws: pd.DataFrame | None = None,
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
            ``q2``). Used for model optics and state transport.
        closed_orbit_tws: Undriven twiss carrying the state reference.
        dispersion_tws: Optional explicit dispersion source; defaults to
            ``closed_orbit_tws``.
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
    smooth_lambda = prepared.smooth_lambda
    dpx_tune_frac = prepared.dpx_tune_frac
    dpy_tune_frac = prepared.dpy_tune_frac

    data = prepared.data.copy(deep=True)
    tws_bpm = _normalise_observed_twiss(tws).reindex(prepared.lattice_bpm_order)

    # The closed-orbit reference already contains the model momentum offset.
    betatron_pt = 0.0

    co_bpm = _normalise_observed_twiss(closed_orbit_tws).reindex(prepared.lattice_bpm_order)
    disp_bpm = _normalise_observed_twiss(
        dispersion_tws if dispersion_tws is not None else closed_orbit_tws
    ).reindex(prepared.lattice_bpm_order)

    data = remove_closed_orbit(data, co_bpm)

    if resolved_tws is not None:
        resolved = _normalise_observed_twiss(resolved_tws)
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

    common = tws_bpm.index.intersection(disp_bpm.index)
    for col in DISPERSION_COLUMNS + SECOND_ORDER_DISPERSION_COLUMNS:
        if col in tws_bpm.columns and col in disp_bpm.columns:
            tws_bpm.loc[common, col] = disp_bpm.loc[common, col].to_numpy(dtype=float)

    bpm_order = [str(name) for name in tws_bpm.index]
    bpm_index = {name: idx for idx, name in enumerate(bpm_order)}
    upstream_frames, downstream_frames = prepare_direct_bpm_reconstruction(
        data,
        tws_bpm,
        window=window,
        bpm_index=bpm_index,
        pt_est=betatron_pt,
        use_immediate_neighbors=prepared.use_immediate_neighbors,
    )
    upstream_side = ACDipoleSide("upstream", "before", +1, upstream_frames)
    downstream_side = ACDipoleSide("downstream", "after", -1, downstream_frames)

    LOGGER.info(
        "Adding closed orbit back to reconstructed BPM momenta before tracking and fitting ACD"
    )
    for side in (upstream_side, downstream_side):
        for bpm_name, frame in side.bpm_frames.items():
            for plane in ("x", "px", "y", "py"):
                if plane in frame and plane in co_bpm.columns:
                    frame[plane] += co_bpm.loc[bpm_name, plane]

    fit = _fit_ac_dipole_from_frames(
        upstream_side=upstream_side,
        downstream_side=downstream_side,
        model=model,
        smooth_lambda=smooth_lambda,
        dpx_tune=dpx_tune_frac,
        dpy_tune=dpy_tune_frac,
    )

    result = _assemble_result_dataframe(
        fit.summary.copy(deep=True),
        fit=fit,
        pt_est=model.pt,
    )

    # Transport each side's cleaned marker state back to its primary BPM.
    cleaned_bpm_states: dict[str, ACDipoleStateSeries] = {}
    consistency_records: list[dict[str, object]] = []
    for side, cleaned_marker in (
        (upstream_side, fit.cleaned_upstream),
        (downstream_side, fit.cleaned_downstream),
    ):
        bpm_state = transport_marker_state_to_bpm(
            cleaned_marker.state,
            model,
            bpm_name=side.primary,
            marker_name=marker_name,
            direction=-side.direction,
        )
        consistency_records.extend(
            _check_bpm_state_consistency(side.primary_frame, side.primary, bpm_state)
        )
        cleaned_bpm_states[side.label] = bpm_state
        result[f"px_bpm_{side.label}_cleaned"] = bpm_state.px
        result[f"py_bpm_{side.label}_cleaned"] = bpm_state.py

    marker_rows = _build_marker_state_rows(
        turns=fit.turns,
        model=model,
        marker_states=[
            (upstream_side, fit.cleaned_upstream),
            (downstream_side, fit.cleaned_downstream),
        ],
    )
    raw_marker_rows = _build_marker_state_rows(
        turns=fit.turns,
        model=model,
        marker_states=[
            (upstream_side, fit.raw_upstream),
            (downstream_side, fit.raw_downstream),
        ],
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
            _build_bpm_state_rows(
                upstream_side,
                turns=fit.turns,
                cleaned_bpm_state=cleaned_bpm_states[upstream_side.label],
            ),
            marker_rows,
            _build_bpm_state_rows(
                downstream_side,
                turns=fit.turns,
                cleaned_bpm_state=cleaned_bpm_states[downstream_side.label],
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
    # Keep the unregularised marker estimates for diagnostics. The rows in the
    # result itself remain the cleaned states consumed by ACD-marker tracking.
    result_out.attrs["raw_marker_states"] = raw_marker_rows
    result_out.attrs["acd_state_consistency"] = consistency_records
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
    use_immediate_neighbors_for_bpms: bool = False,
    closed_orbit_tws: pd.DataFrame,
    dispersion_tws: pd.DataFrame | None = None,
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
            and tune headers ``q1`` / ``q2``.
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
        use_immediate_neighbors_for_bpms: If ``True``, use immediate lattice
            neighbors instead of pi/2-phase neighbors for BPM momentum
            reconstruction.
        closed_orbit_tws: Undriven twiss carrying the state reference.
        dispersion_tws: Optional explicit dispersion source.
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
        use_immediate_neighbors_for_bpms=use_immediate_neighbors_for_bpms,
    )
    return reconstruct_from_prepared(
        prepared,
        tws,
        closed_orbit_tws=closed_orbit_tws,
        dispersion_tws=dispersion_tws,
        resolved_tws=resolved_tws,
    )


__all__ = [
    "PreparedACDInputs",
    "calculate_ac_dipole_momentum",
    "prepare_ac_dipole_inputs",
    "reconstruct_from_prepared",
]
