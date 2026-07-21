"""AC-dipole fit pipeline and public entry point."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from tmom_recon.lattice.core import (
    remove_closed_orbit,
    validate_input,
)
from tmom_recon.lattice.names import normalise_measurement_names
from tmom_recon.optics import AMPLITUDE_COLUMNS, DISPERSION_COLUMNS, PHASE_COLUMNS
from tmom_recon.physics.closed_orbit import parse_plane_spec, warn_on_closed_orbit_mismatch

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
ACD_TWISS_OVERRIDE_COLUMNS = frozenset(PHASE_COLUMNS + AMPLITUDE_COLUMNS + DISPERSION_COLUMNS)


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
    """Build the marker-side output rows for each side's cleaned marker state.

    Args:
        turns: Turn numbers array.
        model: MAD-NG driver, used to resolve the marker element name per side.
        marker_states: ``(side, cleaned_marker_state)`` pairs; each side supplies
            the row type and marker end.

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


def _check_bpm_state_consistency(
    frame: pd.DataFrame, bpm_name: str, predicted_state: ACDipoleStateSeries
) -> None:
    """Check that the reconstructed BPM state is consistent with the predicted state.

    Args:
        frame: Reconstructed-momentum DataFrame for a BPM.
        bpm_name: Name of the BPM.
        predicted_state: Predicted state at the BPM from the model.

    Raises:
        ValueError: If the reconstructed BPM state is not consistent with the predicted state.
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
    for coord in ("x", "px"):
        reconstructed_value = bpm_frame[coord].values
        predicted_value = getattr(predicted_state, coord)
        max_residual = np.max(np.abs(reconstructed_value - predicted_value))
        state_amplitude = np.max(np.abs(predicted_value))
        if state_amplitude > amplitude_threshold:
            tolerance = max(abs_floor, rel_tolerance * state_amplitude)
        else:
            tolerance = abs_floor
        if max_residual > tolerance:
            raise ValueError(
                f"Reconstructed {coord} at BPM {bpm_name} does not match the predicted "
                f"value within tolerance {tolerance:.1e} (max|residual|={max_residual:.3e}, "
                f"|state|={state_amplitude:.3e})."
            )


def _data_mean_closed_orbit(
    data: pd.DataFrame, order: list[str], disp_bpm: pd.DataFrame, pt_est: float
) -> pd.DataFrame:
    """Return the per-BPM turn-mean position of *data*, indexed by *order*.

    The mean of an AC-dipole driven oscillation over the flat-top turns is the
    closed-orbit position at that BPM, so this is a data-only estimate of the
    closed orbit. The dispersive orbit ``pt * d`` is removed so the result is the
    same quantity as the ``dp/p=0`` twiss closed orbit.
    """
    means = data.groupby("name", observed=True)[["x", "y"]].mean()
    means.index = means.index.astype(str).str.upper()
    means = means.reindex(order)
    for plane, dispersion in (("x", "dx"), ("y", "dy")):
        if dispersion in disp_bpm.columns:
            means[plane] -= pt_est * disp_bpm[dispersion].to_numpy(dtype=float)
    return means


def _closed_orbit_momenta(
    co_bpm: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    *,
    planes: tuple[str, ...],
    window: ACDipoleBPMWindow,
    bpm_index: dict[str, int],
    use_immediate_neighbors: bool,
) -> pd.DataFrame:
    """Fill ``px``/``py`` in *co_bpm* from its own positions.

    The per-BPM mean orbit is a static orbit sampled at every BPM, so the same
    neighbour-pair machinery used for the turn-by-turn momenta recovers its
    angle. The orbit is fed in as two identical "turns" so that neighbour
    look-ups which wrap around the ring still find a partner row.

    The angle is inferred with the *model* optics, so this is first-order
    correct: good when the orbit is a perturbation of the model. Only the
    momenta of *planes* are touched; a plane taking its closed orbit from the
    twiss keeps the twiss angle.
    """
    frames = []
    for turn in (0, 1):
        frame = co_bpm[["x", "y"]].reset_index()
        frame.columns = ["name", "x", "y"]
        frame["turn"] = turn
        frame["var_x"] = 0.0
        frame["var_y"] = 0.0
        frames.append(frame)
    co_data = pd.concat(frames, ignore_index=True).dropna(subset=["x", "y"])

    upstream, downstream = prepare_direct_bpm_reconstruction(
        co_data,
        tws_bpm,
        window=window,
        bpm_index=bpm_index,
        pt_est=0.0,
        use_immediate_neighbors=use_immediate_neighbors,
    )
    momenta = tuple({"x": "px", "y": "py"}[plane] for plane in planes)
    co_bpm = co_bpm.copy()
    for momentum in momenta:
        co_bpm[momentum] = 0.0
    for name, frame in {**downstream, **upstream}.items():
        if name not in co_bpm.index or frame.empty:
            continue
        for momentum in momenta:
            if momentum not in frame.columns:
                continue
            # A neighbour look-up that wraps the ring shifts the partner turn by
            # +/-1, leaving one of the two synthetic turns without a partner and
            # hence NaN; take whichever row did resolve.
            values = frame[momentum].dropna()
            if not values.empty:
                co_bpm.loc[name, momentum] = float(values.iat[0])
    return co_bpm


def reconstruct_from_prepared(
    prepared: PreparedACDInputs,
    tws: pd.DataFrame,
    *,
    closed_orbit_tws: pd.DataFrame,
    dispersion_tws: pd.DataFrame | None = None,
    resolved_tws: pd.DataFrame | None = None,
    data_mean_closed_orbit_planes: str | tuple[str, ...] | None = None,
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
        resolved_tws: Optional resolved twiss from
            :func:`tmom_recon.optics.resolve_optics`. When provided, its optics,
            uncertainty and variance columns (and tune headers) override the
            model values for BPM-pair selection and the initial ``px``/``py``
            estimate.
        closed_orbit_tws: Twiss carrying the closed-orbit reference to subtract
            before the betatron reconstruction and restore before tracking.
        dispersion_tws: Optional twiss carrying the dispersion columns to use for
            off-momentum BPM reconstruction. If omitted, ``closed_orbit_tws`` is
            used.
        data_mean_closed_orbit_planes: Planes — ``"x"``, ``"y"``, ``"xy"``,
            ``"yx"``, or ``None``/``""`` — in which the closed orbit is taken
            from the per-BPM turn-mean of the data instead of
            ``closed_orbit_tws``. It is removed before the betatron
            reconstruction and restored afterwards, with its angle inferred from
            its own positions. Default takes the closed orbit from the twiss in
            both planes.

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

    co_bpm = _normalise_observed_twiss(closed_orbit_tws).reindex(prepared.lattice_bpm_order)
    disp_bpm = _normalise_observed_twiss(
        dispersion_tws if dispersion_tws is not None else closed_orbit_tws
    ).reindex(prepared.lattice_bpm_order)

    # By default the closed orbit comes straight from ``closed_orbit_tws``. Per
    # plane the user may instead take it from the data: the mean of an AC-dipole
    # driven oscillation over the flat-top turns is the closed-orbit position, so
    # the per-BPM turn-mean is a data-only estimate, used when the model twiss
    # does not represent the machine orbit. ``data`` still carries the closed
    # orbit at this point.
    override_planes = parse_plane_spec(
        data_mean_closed_orbit_planes, field="data_mean_closed_orbit_planes"
    )
    mean_co = _data_mean_closed_orbit(data, prepared.lattice_bpm_order, disp_bpm, model.pt)
    warn_on_closed_orbit_mismatch(
        co_bpm,
        mean_co,
        planes=tuple(p for p in ("x", "y") if p not in override_planes),
    )
    if override_planes:
        for plane in override_planes:
            co_bpm[plane] = mean_co[plane].to_numpy(dtype=float)
        # The data mean gives positions only; recover the matching angles from
        # those positions so the reference restored before tracking is a
        # consistent state rather than a position with a zero angle.
        co_bpm = _closed_orbit_momenta(
            co_bpm,
            tws_bpm,
            planes=override_planes,
            window=window,
            bpm_index={name: idx for idx, name in enumerate(prepared.lattice_bpm_order)},
            use_immediate_neighbors=prepared.use_immediate_neighbors,
        )
        LOGGER.info(
            "Using per-BPM data mean as closed-orbit reference for plane(s) %s",
            ", ".join(override_planes),
        )

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
    for col in DISPERSION_COLUMNS:
        if col in tws_bpm.columns and col in disp_bpm.columns:
            tws_bpm.loc[common, col] = disp_bpm.loc[common, col].to_numpy(dtype=float)

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
        _check_bpm_state_consistency(side.primary_frame, side.primary, bpm_state)
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
    data_mean_closed_orbit_planes: str | tuple[str, ...] | None = None,
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
        resolved_tws: Optional resolved twiss from
            :func:`tmom_recon.optics.resolve_optics`. When provided, its optics,
            uncertainty and variance columns (and tune headers) override the
            model values for BPM-pair selection and the initial ``px``/``py``
            estimate.
        closed_orbit_tws: Explicit closed-orbit reference.
        dispersion_tws: Optional explicit dispersion source.
        data_mean_closed_orbit_planes: Planes (``"x"``, ``"y"``, ``"xy"``,
            ``"yx"``, or ``None``) in which the closed orbit is taken from the
            per-BPM data mean instead of ``closed_orbit_tws``.

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
        data_mean_closed_orbit_planes=data_mean_closed_orbit_planes,
    )


__all__ = [
    "PreparedACDInputs",
    "calculate_ac_dipole_momentum",
    "prepare_ac_dipole_inputs",
    "reconstruct_from_prepared",
]
