"""Fast AC-dipole kick reconstruction from BPMs around the observed element."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs

from tmom_recon.data.config import POSITION_STD_DEV
from tmom_recon.data.schema import NEXT, PREV
from tmom_recon.lattice.core import (
    attach_lattice_columns,
    build_lattice_maps,
    get_rng,
    inject_noise_xy_inplace,
    remove_closed_orbit_inplace,
    validate_input,
)
from tmom_recon.physics.bpm_phases import next_bpm_to_pi_2, prev_bpm_to_pi_2
from tmom_recon.physics.momenta import momenta_from_next, momenta_from_prev

from .cleaning import _clean_ac_dipole_states, _combine_state_tables
from .models import (
    ACDipoleBPMSelection,
    ACDipoleBPMWindow,
    ACDipoleHarmonicFit,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)
from .selection import (
    resolve_name as _resolve_name,
)
from .selection import (
    select_ac_dipole_bpm_window,
    select_ac_dipole_bpms,
)

if TYPE_CHECKING:
    from .madng_driver import ACDipoleMadDriver

LOGGER = logging.getLogger(__name__)


def _get_tune(tws: pd.DataFrame, header_key: str) -> float:
    headers = dict(getattr(tws, "headers", {}) or {})
    if header_key in headers:
        return float(headers[header_key])
    value = getattr(tws, header_key, None)
    if value is not None and not callable(value):
        return float(value)
    raise KeyError(
        f"Twiss table is missing tune {header_key!r}; expected it in twiss headers or as an attribute."
    )


def _restore_reference_momenta(
    values: np.ndarray,
    tws: pd.DataFrame,
    element_name: str,
    column: str,
) -> np.ndarray:
    if column not in tws.columns or element_name not in tws.index:
        return values
    return values + float(tws.at[element_name, column])


def _estimate_dominant_tune(turns: np.ndarray, values: np.ndarray) -> float:
    turn_array = np.asarray(turns, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if turn_array.ndim != 1 or value_array.ndim != 1 or len(turn_array) != len(value_array):
        return 0.0
    if len(turn_array) < 4:
        return 0.0

    deltas = np.diff(turn_array)
    if np.any(deltas <= 0):
        return 0.0

    step = float(np.median(deltas))
    centered = value_array - np.mean(value_array)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=step)
    if len(freqs) <= 1:
        return 0.0

    peak_idx = int(np.argmax(np.abs(spectrum[1:])) + 1)
    tune_guess = float(freqs[peak_idx])
    if tune_guess <= 0.0:
        return 0.0

    span = max(float(turn_array[-1] - turn_array[0]), 1.0)
    tune_step = 1.0 / span
    tune_grid = np.linspace(
        max(1e-6, tune_guess - 2.0 * tune_step),
        min(0.5 - 1e-6, tune_guess + 2.0 * tune_step),
        257,
    )

    omega_grid = 2.0 * np.pi * tune_grid  # (257,)
    sin_t = np.sin(np.outer(omega_grid, turn_array))  # (257, T)
    cos_t = np.cos(np.outer(omega_grid, turn_array))  # (257, T)

    # Normal equations D^T D (257 × 3×3) without building (257, T, 3) intermediate.
    # D columns: [sin, cos, 1].  Unique symmetric entries:
    ss = (sin_t * sin_t).sum(axis=1)  # (257,)
    sc = (sin_t * cos_t).sum(axis=1)
    s1 = sin_t.sum(axis=1)
    cc = (cos_t * cos_t).sum(axis=1)
    c1 = cos_t.sum(axis=1)
    nn = float(len(turn_array))
    dtd = np.array([[ss, sc, s1], [sc, cc, c1], [s1, c1, np.full(len(tune_grid), nn)]]).transpose(
        2, 0, 1
    )  # (257, 3, 3)
    dty = np.column_stack(
        [
            sin_t @ value_array,
            cos_t @ value_array,
            np.full(len(tune_grid), value_array.sum()),
        ]
    )  # (257, 3)

    theta = np.linalg.solve(dtd, dty[:, :, None]).squeeze(-1)  # (257, 3)
    # SSE = ||y||^2 - y^T D theta  (no need to compute fitted explicitly)
    yty = float(np.dot(value_array, value_array))
    sse = yty - np.einsum("gi,gi->g", dty, theta)  # (257,)
    return float(tune_grid[int(np.argmin(sse))])


def _normalise_measurement_names(data: pd.DataFrame, tws_names: list[str]) -> pd.DataFrame:
    data = data.copy(deep=True)
    raw_name_map = {
        str(name): _resolve_name(str(name), tws_names) for name in data["name"].unique()
    }
    data["name"] = data["name"].astype(str).map(raw_name_map)
    return data


def _normalise_twiss_index(tws: pd.DataFrame, lattice_names: list[str]) -> pd.DataFrame:
    normalised = tws.copy(deep=True)
    normalised.index = [_resolve_name(str(name), lattice_names) for name in normalised.index]
    return normalised


def _require_neighbor_name(value: object, bpm_name: str, plane: str, direction: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(
            f"No {direction} pi/2 BPM was found for {plane}-plane reconstruction at {bpm_name!r}"
        )
    return str(value)


def _prepare_neighbor_tables(
    tws_bpm: pd.DataFrame,
    use_immediate_neighbors_for_bpms: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare neighbor BPM lookup tables.

    Args:
        tws_bpm: Twiss dataframe indexed by BPM names
        use_immediate_neighbors_for_bpms: If True, use the immediate neighboring BPMs in the lattice as the pi/2 BPMs for all BPMs, instead of the BPMs at the closest pi/2 phase advance. This can be useful for very sparse BPM configurations where the pi/2 BPMs are very far from the primary BPMs, but may lead to worse reconstruction quality if the immediate neighbors are not close to pi/2 phase advance.
    """
    q1 = _get_tune(tws_bpm, "q1")
    q2 = _get_tune(tws_bpm, "q2")

    prev_x = prev_bpm_to_pi_2(tws_bpm["mu1"], q1).rename(
        columns={"prev_bpm": PREV.bpm_x, "delta": PREV.delta_x}
    )
    prev_y = prev_bpm_to_pi_2(tws_bpm["mu2"], q2).rename(
        columns={"prev_bpm": PREV.bpm_y, "delta": PREV.delta_y}
    )
    next_x = next_bpm_to_pi_2(tws_bpm["mu1"], q1).rename(
        columns={"next_bpm": NEXT.bpm_x, "delta": NEXT.delta_x}
    )
    next_y = next_bpm_to_pi_2(tws_bpm["mu2"], q2).rename(
        columns={"next_bpm": NEXT.bpm_y, "delta": NEXT.delta_y}
    )

    if use_immediate_neighbors_for_bpms:
        # Use immediate lattice neighbors instead of π/2 phase neighbors
        bpm_order = pd.Series(tws_bpm.index.astype(str), index=tws_bpm.index)
        prev_names = bpm_order.shift(1).fillna(bpm_order.iloc[-1])
        next_names = bpm_order.shift(-1).fillna(bpm_order.iloc[0])
        prev_x[PREV.bpm_x] = prev_names
        prev_y[PREV.bpm_y] = prev_names
        next_x[NEXT.bpm_x] = next_names
        next_y[NEXT.bpm_y] = next_names

        # Recalculate delta values from the phase advances in mu1/mu2, not from s.
        # The momentum formulas expect delta = actual_phase_advance - 0.25.
        mu1 = pd.Series(tws_bpm["mu1"].to_numpy(dtype=float), index=tws_bpm.index)
        mu2 = pd.Series(tws_bpm["mu2"].to_numpy(dtype=float), index=tws_bpm.index)
        q1 = _get_tune(tws_bpm, "q1")
        q2 = _get_tune(tws_bpm, "q2")

        prev_phase_x = (mu1 - mu1.shift(1).fillna(mu1.iloc[-1]) + q1) % q1
        prev_phase_y = (mu2 - mu2.shift(1).fillna(mu2.iloc[-1]) + q2) % q2
        next_phase_x = (mu1.shift(-1).fillna(mu1.iloc[0]) - mu1 + q1) % q1
        next_phase_y = (mu2.shift(-1).fillna(mu2.iloc[0]) - mu2 + q2) % q2

        prev_x[PREV.delta_x] = prev_phase_x - 0.25
        prev_y[PREV.delta_y] = prev_phase_y - 0.25
        next_x[NEXT.delta_x] = next_phase_x - 0.25
        next_y[NEXT.delta_y] = next_phase_y - 0.25

    return prev_x, prev_y, next_x, next_y


def _merge_prev_neighbor_data(
    rows: pd.DataFrame,
    data: pd.DataFrame,
    bpm_index: dict[str, int],
) -> pd.DataFrame:
    current_idx = bpm_index[str(rows["name"].iat[0])]
    x_neighbor_idx = bpm_index[str(rows[PREV.bpm_x].iat[0])]
    y_neighbor_idx = bpm_index[str(rows[PREV.bpm_y].iat[0])]

    rows["turn_x_p"] = rows["turn"] - int(current_idx < x_neighbor_idx)
    rows["turn_y_p"] = rows["turn"] - int(current_idx < y_neighbor_idx)

    x_coords = data[["turn", "name", "x", "var_x"]].rename(
        columns={"turn": "turn_x_p", "name": PREV.bpm_x, "x": PREV.x, "var_x": PREV.var_x}
    )
    y_coords = data[["turn", "name", "y", "var_y"]].rename(
        columns={"turn": "turn_y_p", "name": PREV.bpm_y, "y": PREV.y, "var_y": PREV.var_y}
    )

    rows = rows.merge(x_coords, on=["turn_x_p", PREV.bpm_x], how="left")
    rows = rows.merge(y_coords, on=["turn_y_p", PREV.bpm_y], how="left")
    return rows.drop(columns=["turn_x_p", "turn_y_p"])


def _merge_next_neighbor_data(
    rows: pd.DataFrame,
    data: pd.DataFrame,
    bpm_index: dict[str, int],
) -> pd.DataFrame:
    current_idx = bpm_index[str(rows["name"].iat[0])]
    x_neighbor_idx = bpm_index[str(rows[NEXT.bpm_x].iat[0])]
    y_neighbor_idx = bpm_index[str(rows[NEXT.bpm_y].iat[0])]

    rows["turn_x_n"] = rows["turn"] + int(current_idx > x_neighbor_idx)
    rows["turn_y_n"] = rows["turn"] + int(current_idx > y_neighbor_idx)

    x_coords = data[["turn", "name", "x", "var_x"]].rename(
        columns={"turn": "turn_x_n", "name": NEXT.bpm_x, "x": NEXT.x, "var_x": NEXT.var_x}
    )
    y_coords = data[["turn", "name", "y", "var_y"]].rename(
        columns={"turn": "turn_y_n", "name": NEXT.bpm_y, "y": NEXT.y, "var_y": NEXT.var_y}
    )

    rows = rows.merge(x_coords, on=["turn_x_n", NEXT.bpm_x], how="left")
    rows = rows.merge(y_coords, on=["turn_y_n", NEXT.bpm_y], how="left")
    return rows.drop(columns=["turn_x_n", "turn_y_n"])


def _prepare_prev_reconstruction(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    bpm_name: str,
    prev_x: pd.DataFrame,
    prev_y: pd.DataFrame,
    bpm_index: dict[str, int],
) -> pd.DataFrame:
    maps = build_lattice_maps(tws_bpm)
    rows = data.loc[data["name"] == bpm_name, ["name", "turn", "x", "y", "var_x", "var_y"]].copy(
        deep=True
    )
    attach_lattice_columns(rows, maps)

    rows[PREV.bpm_x] = _require_neighbor_name(
        prev_x.at[bpm_name, PREV.bpm_x], bpm_name, "x", "previous"
    )
    rows[PREV.delta_x] = float(prev_x.at[bpm_name, PREV.delta_x])
    rows[PREV.bpm_y] = _require_neighbor_name(
        prev_y.at[bpm_name, PREV.bpm_y], bpm_name, "y", "previous"
    )
    rows[PREV.delta_y] = float(prev_y.at[bpm_name, PREV.delta_y])
    rows["sqrt_betax_p"] = rows[PREV.bpm_x].map(maps.sqrt_betax)
    rows["sqrt_betay_p"] = rows[PREV.bpm_y].map(maps.sqrt_betay)

    rows = _merge_prev_neighbor_data(rows, data, bpm_index)
    return momenta_from_prev(rows)


def _prepare_next_reconstruction(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    bpm_name: str,
    next_x: pd.DataFrame,
    next_y: pd.DataFrame,
    bpm_index: dict[str, int],
) -> pd.DataFrame:
    maps = build_lattice_maps(tws_bpm)
    rows = data.loc[data["name"] == bpm_name, ["name", "turn", "x", "y", "var_x", "var_y"]].copy(
        deep=True
    )
    attach_lattice_columns(rows, maps)

    rows[NEXT.bpm_x] = _require_neighbor_name(
        next_x.at[bpm_name, NEXT.bpm_x], bpm_name, "x", "next"
    )
    rows[NEXT.delta_x] = float(next_x.at[bpm_name, NEXT.delta_x])
    rows[NEXT.bpm_y] = _require_neighbor_name(
        next_y.at[bpm_name, NEXT.bpm_y], bpm_name, "y", "next"
    )
    rows[NEXT.delta_y] = float(next_y.at[bpm_name, NEXT.delta_y])
    rows["sqrt_betax_n"] = rows[NEXT.bpm_x].map(maps.sqrt_betax)
    rows["sqrt_betay_n"] = rows[NEXT.bpm_y].map(maps.sqrt_betay)

    rows = _merge_next_neighbor_data(rows, data, bpm_index)
    return momenta_from_next(rows)


def _prepare_direct_bpm_reconstruction(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    *,
    window: ACDipoleBPMWindow,
    bpm_index: dict[str, int],
    use_immediate_neighbors_for_bpms: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    prev_x, prev_y, next_x, next_y = _prepare_neighbor_tables(
        tws_bpm, use_immediate_neighbors_for_bpms=use_immediate_neighbors_for_bpms
    )
    upstream_frames = {
        bpm_name: _prepare_prev_reconstruction(
            data,
            tws_bpm,
            bpm_name,
            prev_x,
            prev_y,
            bpm_index,
        )
        for bpm_name in window.upstream
    }
    downstream_frames = {
        bpm_name: _prepare_next_reconstruction(
            data,
            tws_bpm,
            bpm_name,
            next_x,
            next_y,
            bpm_index,
        )
        for bpm_name in window.downstream
    }
    return upstream_frames, downstream_frames


def _fit_ac_dipole_from_frames(
    *,
    selection: ACDipoleBPMSelection,
    window: ACDipoleBPMWindow,
    marker_name: str,
    model: ACDipoleMadDriver,
    smooth_lambda: float,
    upstream_frames: dict[str, pd.DataFrame],
    downstream_frames: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    ACDipoleStateEstimate,
    ACDipoleStateEstimate,
    ACDipoleStateEstimate,
    ACDipoleStateEstimate,
    np.ndarray,
    np.ndarray,
    ACDipoleHarmonicFit,
    ACDipoleHarmonicFit,
]:
    primary_upstream = upstream_frames[selection.upstream]
    primary_downstream = downstream_frames[selection.downstream]
    result = _merge_primary_bpm_results(
        primary_upstream,
        primary_downstream,
        selection=selection,
        marker_name=marker_name,
    )
    turns = result["turn"].to_numpy(dtype=float)

    upstream_tables = [
        _build_tracked_state_table(
            upstream_frames[bpm_name],
            model,
            source_name=bpm_name,
            marker_name=marker_name,
            direction=1,
        )
        for bpm_name in window.upstream
    ]
    downstream_tables = [
        _build_tracked_state_table(
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
    cleaned_upstream, cleaned_downstream, dpx_fit, dpy_fit = _clean_ac_dipole_states(
        turns,
        upstream_tables,
        downstream_tables,
        dpx_tune=_estimate_dominant_tune(turns, dpx_raw),
        dpy_tune=_estimate_dominant_tune(turns, dpy_raw),
        smooth_lambda=smooth_lambda,
    )
    return (
        result,
        turns,
        raw_upstream,
        raw_downstream,
        cleaned_upstream,
        cleaned_downstream,
        dpx_raw,
        dpy_raw,
        dpx_fit,
        dpy_fit,
    )


def _transport_to_marker(
    frame: pd.DataFrame,
    model: ACDipoleMadDriver,
    *,
    source_name: str,
    marker_name: str,
    direction: int,
) -> ACDipoleStateSeries:
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +/- 1, got {direction}")

    source_state = frame[["x", "px", "y", "py"]].to_numpy(dtype=float)
    transfer_mat = model.transfer_matrix(source_name, marker_name, direction=direction)
    marker_state = source_state @ transfer_mat
    return ACDipoleStateSeries(
        marker_state[:, 0],
        marker_state[:, 1],
        marker_state[:, 2],
        marker_state[:, 3],
    )


def _build_tracked_state_table(
    frame: pd.DataFrame,
    model: ACDipoleMadDriver,
    *,
    source_name: str,
    marker_name: str,
    direction: int,
) -> pd.DataFrame:
    tracked_state = _transport_to_marker(
        frame,
        model,
        source_name=source_name,
        marker_name=marker_name,
        direction=direction,
    )
    tracked_table = frame[["turn", "var_x", "var_px", "var_y", "var_py"]].copy(deep=True)
    tracked_table["source_bpm"] = source_name
    tracked_table["x"] = tracked_state.x
    tracked_table["px"] = tracked_state.px
    tracked_table["y"] = tracked_state.y
    tracked_table["py"] = tracked_state.py
    return tracked_table.sort_values("turn").reset_index(drop=True)


def _blend_bpm_momentum_component(
    raw_values: np.ndarray,
    raw_variances: np.ndarray,
    raw_acd_values: np.ndarray,
    raw_acd_variances: np.ndarray,
    cleaned_acd_values: np.ndarray,
    cleaned_acd_variances: np.ndarray,
) -> np.ndarray:
    correction = np.asarray(cleaned_acd_values, dtype=float) - np.asarray(
        raw_acd_values, dtype=float
    )
    raw_var = np.clip(np.asarray(raw_variances, dtype=float), 0.0, None)
    correction_var = np.clip(
        np.asarray(raw_acd_variances, dtype=float) + np.asarray(cleaned_acd_variances, dtype=float),
        0.0,
        None,
    )
    denominator = 2.0 * raw_var + correction_var
    shrinkage = np.divide(
        raw_var,
        denominator,
        out=np.zeros_like(raw_var, dtype=float),
        where=denominator > 0.0,
    )
    return np.asarray(raw_values, dtype=float) + shrinkage * correction


def _blend_bpm_momentum_estimate(
    frame: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    bpm_name: str,
    raw_acd: ACDipoleStateEstimate,
    cleaned_acd: ACDipoleStateEstimate,
) -> tuple[np.ndarray, np.ndarray]:
    raw_px = _restore_reference_momenta(frame["px"].to_numpy(dtype=float), tws, bpm_name, "px")
    raw_py = _restore_reference_momenta(frame["py"].to_numpy(dtype=float), tws, bpm_name, "py")
    cleaned_px = _blend_bpm_momentum_component(
        raw_px,
        frame["var_px"].to_numpy(dtype=float),
        raw_acd.state.px,
        raw_acd.var_px,
        cleaned_acd.state.px,
        cleaned_acd.var_px,
    )
    cleaned_py = _blend_bpm_momentum_component(
        raw_py,
        frame["var_py"].to_numpy(dtype=float),
        raw_acd.state.py,
        raw_acd.var_py,
        cleaned_acd.state.py,
        cleaned_acd.var_py,
    )
    return cleaned_px, cleaned_py


def _merge_primary_bpm_results(
    upstream_frame: pd.DataFrame,
    downstream_frame: pd.DataFrame,
    *,
    selection: ACDipoleBPMSelection,
    marker_name: str,
) -> pd.DataFrame:
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


def _build_ac_dipole_headers(
    *,
    marker_name: str,
    window: ACDipoleBPMWindow,
    dpx_fit: ACDipoleHarmonicFit,
    dpy_fit: ACDipoleHarmonicFit,
) -> dict[str, object]:
    return {
        "ACD_MARKER": marker_name,
        "ACD_ELEMENT": marker_name,
        "ACD_BPM_UPSTREAM": window.primary.upstream,
        "ACD_BPM_DOWNSTREAM": window.primary.downstream,
        "ACD_BPMS_UPSTREAM_USED": ",".join(window.upstream),
        "ACD_BPMS_DOWNSTREAM_USED": ",".join(window.downstream),
        "ACD_DPX_TUNE": dpx_fit.tune,
        "ACD_DPX_AMPLITUDE": dpx_fit.amplitude,
        "ACD_DPX_PHASE": dpx_fit.phase,
        "ACD_DPX_OFFSET": dpx_fit.offset,
        "ACD_DPY_TUNE": dpy_fit.tune,
        "ACD_DPY_AMPLITUDE": dpy_fit.amplitude,
        "ACD_DPY_PHASE": dpy_fit.phase,
        "ACD_DPY_OFFSET": dpy_fit.offset,
    }


def calculate_ac_dipole_momentum(
    orig_data: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    ac_dipole_marker: str,
    model: ACDipoleMadDriver,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
    smooth_lambda: float = 1,
    inject_noise: bool | float = True,
    rng: np.random.Generator | None = None,
    use_immediate_neighbors_for_bpms: bool = False,
) -> pd.DataFrame:
    """Reconstruct AC-dipole kicks and constrained BPM momenta.

    The kick itself is fitted from the raw pre/post ACD momentum difference. The
    cleaned BPM momenta are then obtained by:
    1. tracking local BPM state estimates to the ACD from each selected BPM,
    2. solving a global weighted linear least-squares problem for the pre/post
         ACD momenta constrained by the fitted kick waveform, with tunable
         smoothness strength ``smooth_lambda``, and
    3. using the ACD-side momentum correction as a variance-weighted update to
       the raw BPM momentum estimates.
    """

    validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")

    rng = get_rng(rng)
    if inject_noise is not False:
        noise_std = POSITION_STD_DEV if inject_noise is True else float(inject_noise)
        inject_noise_xy_inplace(data, orig_data, rng, noise_std=noise_std)

    lattice_names = [str(name) for name in model.twiss_elements.index]
    marker_name = _resolve_name(ac_dipole_marker, lattice_names)
    data = _normalise_measurement_names(data, lattice_names)
    tws = _normalise_twiss_index(tws, lattice_names)

    measured_bpm_names = [str(name) for name in data["name"].unique()]
    available_bpm_names = [name for name in measured_bpm_names if name in set(tws.index)]
    window = select_ac_dipole_bpm_window(
        model.twiss_elements,
        marker_name,
        available_bpm_names,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
    )
    selection = window.primary
    LOGGER.info(
        "Reconstructing AC-dipole kick at %s using upstream BPMs %s and downstream BPMs %s",
        marker_name,
        window.upstream,
        window.downstream,
    )

    data = data[data["name"].isin(available_bpm_names)].copy(deep=True)
    lattice_bpm_order = [
        name for name in model.twiss_elements.index if str(name) in set(available_bpm_names)
    ]
    tws_bpm = tws.reindex(lattice_bpm_order).copy(deep=True)
    remove_closed_orbit_inplace(data, tws)

    bpm_order = [str(name) for name in tws_bpm.index]
    bpm_index = {str(name): idx for idx, name in enumerate(bpm_order)}
    upstream_frames, downstream_frames = _prepare_direct_bpm_reconstruction(
        data,
        tws_bpm,
        window=window,
        bpm_index=bpm_index,
        use_immediate_neighbors_for_bpms=use_immediate_neighbors_for_bpms,
    )
    (
        result,
        turns,
        raw_upstream,
        raw_downstream,
        cleaned_upstream,
        cleaned_downstream,
        dpx_raw,
        dpy_raw,
        dpx_fit,
        dpy_fit,
    ) = _fit_ac_dipole_from_frames(
        selection=selection,
        window=window,
        marker_name=marker_name,
        model=model,
        smooth_lambda=smooth_lambda,
        upstream_frames=upstream_frames,
        downstream_frames=downstream_frames,
    )
    primary_upstream = upstream_frames[selection.upstream]
    primary_downstream = downstream_frames[selection.downstream]

    px_bpm_upstream_cleaned, py_bpm_upstream_cleaned = _blend_bpm_momentum_estimate(
        primary_upstream,
        tws,
        bpm_name=selection.upstream,
        raw_acd=raw_upstream,
        cleaned_acd=cleaned_upstream,
    )
    px_bpm_downstream_cleaned, py_bpm_downstream_cleaned = _blend_bpm_momentum_estimate(
        primary_downstream,
        tws,
        bpm_name=selection.downstream,
        raw_acd=raw_downstream,
        cleaned_acd=cleaned_downstream,
    )

    bpm_elements = {
        "upstream": selection.upstream,
        "downstream": selection.downstream,
    }
    for side, element_name in bpm_elements.items():
        for plane in ("px", "py"):
            col = f"{plane}_bpm_{side}"
            result[col] = _restore_reference_momenta(
                result[col].to_numpy(dtype=float),
                tws,
                element_name,
                plane,
            )

    raw_states = {
        "upstream": raw_upstream.state,
        "downstream": raw_downstream.state,
    }
    for side, state in raw_states.items():
        result[f"x_acd_{side}"] = state.x
        result[f"y_acd_{side}"] = state.y
        for plane in ("px", "py"):
            result[f"{plane}_acd_{side}"] = _restore_reference_momenta(
                getattr(state, plane),
                tws,
                marker_name,
                plane,
            )
    result["dpx_rad"] = dpx_raw
    result["dpy_rad"] = dpy_raw
    result["dpx"] = result["dpx_rad"]
    result["dpy"] = result["dpy_rad"]
    result["dpx_fit_rad"] = dpx_fit.fitted
    result["dpy_fit_rad"] = dpy_fit.fitted
    cleaned_states = {
        "upstream": cleaned_upstream.state,
        "downstream": cleaned_downstream.state,
    }
    result["x_acd_cleaned"] = cleaned_states["upstream"].x
    result["y_acd_cleaned"] = cleaned_states["upstream"].y
    for side, state in cleaned_states.items():
        result[f"x_acd_{side}_cleaned"] = state.x
        result[f"y_acd_{side}_cleaned"] = state.y
        for plane in ("px", "py"):
            result[f"{plane}_acd_{side}_cleaned"] = _restore_reference_momenta(
                getattr(state, plane),
                tws,
                marker_name,
                plane,
            )

    cleaned_bpm = {
        "upstream": {
            "px": px_bpm_upstream_cleaned,
            "py": py_bpm_upstream_cleaned,
        },
        "downstream": {
            "px": px_bpm_downstream_cleaned,
            "py": py_bpm_downstream_cleaned,
        },
    }
    for side, planes in cleaned_bpm.items():
        for plane, values in planes.items():
            result[f"{plane}_bpm_{side}_cleaned"] = values

    dpx_fit_meta = {
        "tune": dpx_fit.tune,
        "amplitude": dpx_fit.amplitude,
        "phase": dpx_fit.phase,
        "offset": dpx_fit.offset,
    }
    dpy_fit_meta = {
        "tune": dpy_fit.tune,
        "amplitude": dpy_fit.amplitude,
        "phase": dpy_fit.phase,
        "offset": dpy_fit.offset,
    }
    headers = _build_ac_dipole_headers(
        marker_name=marker_name,
        window=window,
        dpx_fit=dpx_fit,
        dpy_fit=dpy_fit,
    )
    result_out = tfs.TfsDataFrame(
        result.sort_values("turn").reset_index(drop=True),
        headers=headers,
    )
    result_out.attrs["acd_marker"] = marker_name
    result_out.attrs["acd_element"] = marker_name
    result_out.attrs["bpm_upstream"] = selection.upstream
    result_out.attrs["bpm_downstream"] = selection.downstream
    result_out.attrs["bpms_upstream_used"] = window.upstream
    result_out.attrs["bpms_downstream_used"] = window.downstream
    result_out.attrs["smooth_lambda"] = smooth_lambda
    result_out.attrs["dpx_fit"] = dpx_fit_meta
    result_out.attrs["dpy_fit"] = dpy_fit_meta
    return result_out


__all__ = [
    "ACDipoleBPMSelection",
    "ACDipoleBPMWindow",
    "ACDipoleHarmonicFit",
    "calculate_ac_dipole_momentum",
    "select_ac_dipole_bpms",
    "select_ac_dipole_bpm_window",
]
