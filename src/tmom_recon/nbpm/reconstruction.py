from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import SupportsFloat, SupportsInt, cast

import numpy as np
import pandas as pd

from tmom_recon.acd.integration import (
    ACDipoleConfig,
    apply_ac_dipole_bpm_overrides,
    apply_precomputed_ac_dipole_bpm_overrides,
    run_ac_dipole_reconstruction,
)
from tmom_recon.data.config import POSITION_STD_DEV
from tmom_recon.lattice.core import (
    OUT_COLS,
    diagnostics,
    remove_closed_orbit,
    restore_closed_orbit_and_reference_momenta,
    validate_input,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationCube:
    x: np.ndarray
    y: np.ndarray
    var_x: np.ndarray
    var_y: np.ndarray
    row_bpm_idx: np.ndarray
    row_turn_idx: np.ndarray
    turn_values: np.ndarray


@dataclass(frozen=True)
class PairTemplate:
    plane: str
    branch: str
    eta: int
    i_bpm_name: str
    j_bpm_name: str
    i_bpm_idx: int
    j_bpm_idx: int
    phi_model_rad: float
    beta_i: float
    beta_j: float
    alpha_i: float
    d_i: float
    d_j: float
    dp_i: float
    turn_shift: int
    kick_sign: float


@dataclass(frozen=True)
class PairCandidate:
    branch: str
    eta: int
    j_bpm_name: str
    phi_model_rad: float
    crosses_excluded_element: bool = False


def _normalise_twiss(frame: pd.DataFrame) -> pd.DataFrame:
    tws = frame.copy(deep=True)
    if "name" in tws.columns:
        tws["name"] = tws["name"].astype(str).str.upper()
        tws = tws.set_index("name", drop=False)
    else:
        tws.index = tws.index.astype(str).str.upper()
        tws["name"] = tws.index
    return tws


def _resolve_name_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "name" in frame.columns:
        out = frame.copy(deep=True)
    else:
        out = (
            frame.reset_index()
            .rename(columns={frame.index.name or "index": "name"})
            .copy(deep=True)
        )
    out["name"] = out["name"].astype(str).str.upper()
    return out


def _get_column(frame: pd.DataFrame, *names: str, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index, dtype=float)


def _all_phase_candidates(
    mu: pd.Series,
    tune: float,
    *,
    target: float = 0.25,
    max_bpm_distance: int = 11,
    forward: bool,
) -> dict[str, list[tuple[str, float]]]:
    values = mu.to_numpy(dtype=float)
    names = mu.index.astype(str).tolist()
    n_bpms = len(mu)
    out: dict[str, list[tuple[str, float]]] = {}

    def _phase_distance(values_in: np.ndarray, center: float) -> np.ndarray:
        return np.abs((values_in - center + 0.5) % 1.0 - 0.5)

    for i, bpm_name in enumerate(names):
        if forward:
            diff = (values - values[i] + tune) % tune
            distances = (np.arange(n_bpms) - i) % n_bpms
        else:
            diff = (values[i] - values + tune) % tune
            distances = (i - np.arange(n_bpms)) % n_bpms
        diff[i] = np.nan

        local_mask = (distances <= max_bpm_distance) & (distances > 0)
        candidates = np.flatnonzero(local_mask)
        sort_key = sorted(
            (
                int(distances[idx]),
                float(_phase_distance(np.asarray([diff[idx]]), target)[0]),
                int(idx),
            )
            for idx in candidates
        )
        out[bpm_name] = [(names[idx], float(diff[idx] - target)) for _dist, _err, idx in sort_key]
    return out


def _distance_to_integer_pi(angle_rad: float) -> float:
    wrapped = np.mod(angle_rad, np.pi)
    return float(min(wrapped, np.pi - wrapped))


def _wrap_contains(start_s: float, end_s: float, value_s: float, circumference: float) -> bool:
    del circumference
    if np.isnan(start_s) or np.isnan(end_s) or np.isnan(value_s):
        return False
    if start_s <= end_s:
        return start_s < value_s < end_s
    return value_s > start_s or value_s < end_s


def _is_excluded_element_inside_pair(
    *,
    bpm_s: float,
    neighbor_s: float,
    branch: str,
    excluded_s: float | None,
    circumference: float,
) -> bool:
    if excluded_s is None:
        return False
    if branch == "next":
        return _wrap_contains(bpm_s, neighbor_s, excluded_s, circumference)
    return _wrap_contains(neighbor_s, bpm_s, excluded_s, circumference)


def _build_pair_catalog(
    tws_bpm: pd.DataFrame,
    *,
    twiss_elements: pd.DataFrame | None = None,
    plane: str,
    target_phase_turns: float = 0.25,
    max_bpm_distance: int = 11,
    min_abs_cos: float = 0.05,
    bad_phase_threshold_rad: float = 2.0 * np.pi * 1.0e-2,
    excluded_element_name: str | None = None,
    allow_excluded_crossing: bool = False,
) -> dict[str, list[PairCandidate]]:
    tws = _resolve_name_column(tws_bpm).set_index("name", drop=False)
    twiss_reference = _resolve_name_column(
        twiss_elements if twiss_elements is not None else tws
    ).set_index("name", drop=False)
    mu_col = "mu1" if plane == "x" else "mu2"
    tune_key = "q1" if plane == "x" else "q2"
    tune = float(tws.attrs.get(tune_key, 1.0))
    if hasattr(tws_bpm, "headers"):
        headers = dict(getattr(tws_bpm, "headers", {}) or {})
        tune = float(headers.get(tune_key, tune))

    excluded_s: float | None = None
    if excluded_element_name is not None:
        excluded_key = str(excluded_element_name).upper()
        if excluded_key not in twiss_reference.index:
            raise KeyError(f"Excluded element {excluded_element_name!r} not present in twiss table")
        excluded_s = float(twiss_reference.loc[excluded_key, "s"])

    circumference = float(_get_column(twiss_reference, "s").max())
    mu = tws[mu_col]
    prev_candidates = _all_phase_candidates(
        mu,
        tune,
        target=target_phase_turns,
        max_bpm_distance=max_bpm_distance,
        forward=False,
    )
    next_candidates = _all_phase_candidates(
        mu,
        tune,
        target=target_phase_turns,
        max_bpm_distance=max_bpm_distance,
        forward=True,
    )

    out: dict[str, list[PairCandidate]] = {}
    for bpm_name in tws.index:
        rows: list[PairCandidate] = []
        for branch, candidates in (
            ("prev", prev_candidates[bpm_name]),
            ("next", next_candidates[bpm_name]),
        ):
            eta = -1 if branch == "prev" else 1
            for neighbor_name, phase_turns in candidates:
                actual_phase_rad = 2.0 * np.pi * (phase_turns + target_phase_turns)
                phi_model_rad = 2.0 * np.pi * phase_turns
                bpm_s = float(tws.loc[bpm_name, "s"])
                neighbor_s = float(tws.loc[neighbor_name, "s"])
                if not np.isfinite(phi_model_rad):
                    continue
                if _distance_to_integer_pi(actual_phase_rad) < bad_phase_threshold_rad:
                    continue
                if abs(np.cos(phi_model_rad)) < min_abs_cos:
                    continue
                crosses_excluded_element = _is_excluded_element_inside_pair(
                    bpm_s=bpm_s,
                    neighbor_s=neighbor_s,
                    branch=branch,
                    excluded_s=excluded_s,
                    circumference=circumference,
                )
                if crosses_excluded_element and not allow_excluded_crossing:
                    continue
                rows.append(
                    PairCandidate(
                        branch=branch,
                        eta=eta,
                        j_bpm_name=neighbor_name,
                        phi_model_rad=phi_model_rad,
                        crosses_excluded_element=crosses_excluded_element,
                    )
                )
        out[str(bpm_name)] = rows
    return out


def _build_observation_cube(
    data: pd.DataFrame,
    bpm_names: list[str],
) -> ObservationCube:
    """Materialize BPM observations into dense [bpm, turn] arrays.

    The batched N-BPM path repeatedly accesses neighboring BPM values at the
    same or adjacent turns. Converting the input frame into dense arrays once
    avoids per-row dictionary lookups in the hot path.
    """
    turn_values = np.sort(np.unique(data["turn"].to_numpy(dtype=int, copy=True)))
    bpm_to_idx = {name: idx for idx, name in enumerate(bpm_names)}
    turn_to_idx = {int(turn): idx for idx, turn in enumerate(turn_values.tolist())}

    row_bpm_idx = np.array([bpm_to_idx[str(name)] for name in data["name"].astype(str)], dtype=int)
    row_turn_idx = np.array(
        [turn_to_idx[int(turn)] for turn in data["turn"].to_numpy(dtype=int)], dtype=int
    )

    shape = (len(bpm_names), len(turn_values))
    x = np.full(shape, np.nan, dtype=float)
    y = np.full(shape, np.nan, dtype=float)
    var_x = np.full(shape, np.nan, dtype=float)
    var_y = np.full(shape, np.nan, dtype=float)

    x[row_bpm_idx, row_turn_idx] = data["x"].to_numpy(dtype=float)
    y[row_bpm_idx, row_turn_idx] = data["y"].to_numpy(dtype=float)
    var_x[row_bpm_idx, row_turn_idx] = data["var_x"].to_numpy(dtype=float)
    var_y[row_bpm_idx, row_turn_idx] = data["var_y"].to_numpy(dtype=float)
    return ObservationCube(
        x=x,
        y=y,
        var_x=var_x,
        var_y=var_y,
        row_bpm_idx=row_bpm_idx,
        row_turn_idx=row_turn_idx,
        turn_values=turn_values,
    )


def _sanitize_measurement_variances(data: pd.DataFrame, variance_floor: float) -> pd.DataFrame:
    """Clamp measurement variances to a finite, non-zero floor."""
    out = data.copy(deep=True)
    for col in ("var_x", "var_y"):
        values = np.full(len(data), variance_floor, dtype=float)
        if col in data.columns:
            raw_values = data[col].to_numpy(dtype=float)
            finite_mask = np.isfinite(raw_values)
            values[finite_mask] = np.maximum(raw_values[finite_mask], variance_floor)
        out[col] = values
    return out


def _as_int(value: object) -> int:
    return int(cast(SupportsInt, value))


def _as_float(value: object) -> float:
    return float(cast(SupportsFloat, value))


def _build_pair_templates(
    tws_bpm: pd.DataFrame,
    *,
    twiss_elements: pd.DataFrame | None = None,
    plane: str,
    max_bpm_distance: int,
    min_abs_cos: float,
    bad_phase_threshold_rad: float,
    excluded_element_name: str | None = None,
    allow_excluded_crossing: bool = False,
) -> dict[str, list[PairTemplate]]:
    catalog = _build_pair_catalog(
        tws_bpm,
        twiss_elements=twiss_elements,
        plane=plane,
        max_bpm_distance=max_bpm_distance,
        min_abs_cos=min_abs_cos,
        bad_phase_threshold_rad=bad_phase_threshold_rad,
        excluded_element_name=excluded_element_name,
        allow_excluded_crossing=allow_excluded_crossing,
    )
    bpm_names = tws_bpm.index.astype(str).tolist()
    bpm_index = {name: idx for idx, name in enumerate(bpm_names)}
    beta_col = "beta11" if plane == "x" else "beta22"
    alpha_col = "alfa11" if plane == "x" else "alfa22"
    disp_col = "dx" if plane == "x" else "dy"
    ddisp_col = "dpx" if plane == "x" else "dpy"
    templates: dict[str, list[PairTemplate]] = {}
    for bpm_name, candidates in catalog.items():
        current_row = tws_bpm.loc[bpm_name]
        current_idx = bpm_index[bpm_name]
        bpm_templates: list[PairTemplate] = []
        for candidate in candidates:
            neighbor_name = candidate.j_bpm_name.upper()
            neighbor_row = tws_bpm.loc[neighbor_name]
            neighbor_idx = bpm_index[neighbor_name]
            turn_shift = 0
            if candidate.branch == "prev" and current_idx < neighbor_idx:
                turn_shift = -1
            elif candidate.branch == "next" and current_idx > neighbor_idx:
                turn_shift = 1
            bpm_templates.append(
                PairTemplate(
                    plane=plane,
                    branch=candidate.branch,
                    eta=_as_int(candidate.eta),
                    i_bpm_name=bpm_name,
                    j_bpm_name=neighbor_name,
                    i_bpm_idx=current_idx,
                    j_bpm_idx=neighbor_idx,
                    phi_model_rad=_as_float(candidate.phi_model_rad),
                    beta_i=float(current_row[beta_col]),
                    beta_j=float(neighbor_row[beta_col]),
                    alpha_i=float(current_row[alpha_col]),
                    d_i=float(current_row.get(disp_col, 0.0)),
                    d_j=float(neighbor_row.get(disp_col, 0.0)),
                    dp_i=float(current_row.get(ddisp_col, 0.0)),
                    turn_shift=turn_shift,
                    kick_sign=(
                        -1.0
                        if candidate.crosses_excluded_element and candidate.branch == "next"
                        else 1.0
                        if candidate.crosses_excluded_element and candidate.branch == "prev"
                        else 0.0
                    ),
                )
            )
        templates[bpm_name] = bpm_templates
    return templates


def _combine_momentum_blue(
    pair_momenta: np.ndarray,
    covariance: np.ndarray,
) -> tuple[float, float]:
    ones = np.ones(len(pair_momenta), dtype=float)
    try:
        solved_ones = np.linalg.solve(covariance, ones)
    except np.linalg.LinAlgError:
        solved_ones = np.linalg.pinv(covariance, hermitian=True) @ ones
    denom = float(ones @ solved_ones)
    if denom <= 0.0:
        return np.nan, np.nan
    weights = solved_ones / denom
    return float(weights @ pair_momenta), float(1.0 / denom)


def _reconstruct_plane_cube(
    *,
    bpm_names: list[str],
    observation_cube: ObservationCube,
    pair_templates: dict[str, list[PairTemplate]],
    plane: str,
    delta: float = 0.0,
    kick_by_turn: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one plane by directly combining per-pair estimates."""
    values = observation_cube.x if plane == "x" else observation_cube.y
    variances = observation_cube.var_x if plane == "x" else observation_cube.var_y
    kicks = (
        np.zeros(observation_cube.turn_values.shape[0], dtype=float)
        if kick_by_turn is None
        else np.asarray(kick_by_turn, dtype=float)
    )
    n_bpms, n_turns = values.shape
    momentum = np.full((n_bpms, n_turns), np.nan, dtype=float)
    momentum_var = np.full((n_bpms, n_turns), np.nan, dtype=float)

    for bpm_idx, bpm_name in enumerate(bpm_names):
        templates = pair_templates.get(bpm_name, [])
        if not templates:
            continue
        for turn_idx in range(n_turns):
            current_var = float(variances[bpm_idx, turn_idx])
            if not np.isfinite(current_var) or current_var < 0.0:
                continue
            current_u = float(values[bpm_idx, turn_idx])

            pair_values: list[float] = []
            coeff_i_values: list[float] = []
            coeff_j_values: list[float] = []
            neighbor_group_keys: list[tuple[int, int]] = []
            neighbor_variances: dict[tuple[int, int], float] = {}

            for template in templates:
                neighbor_turn_idx = turn_idx + template.turn_shift
                if neighbor_turn_idx < 0 or neighbor_turn_idx >= n_turns:
                    continue
                neighbor_idx = template.j_bpm_idx
                neighbor_var = float(variances[neighbor_idx, neighbor_turn_idx])
                if not np.isfinite(neighbor_var) or neighbor_var < 0.0:
                    continue
                neighbor_u = float(values[neighbor_idx, neighbor_turn_idx])
                sec_phi = 1.0 / np.cos(template.phi_model_rad)
                tan_phi = np.tan(template.phi_model_rad)
                coeff_i = (
                    template.eta * (tan_phi - template.eta * template.alpha_i) / template.beta_i
                )
                coeff_j = template.eta * sec_phi / np.sqrt(template.beta_i * template.beta_j)
                coeff_delta = (
                    template.eta
                    * (
                        -template.d_j * sec_phi / np.sqrt(template.beta_i * template.beta_j)
                        - template.d_i
                        * (tan_phi - template.eta * template.alpha_i)
                        / template.beta_i
                    )
                    + template.dp_i
                )
                pair_values.append(
                    coeff_i * current_u
                    + coeff_j * neighbor_u
                    + coeff_delta * float(delta)
                    + template.kick_sign * kicks[turn_idx]
                )
                coeff_i_values.append(coeff_i)
                coeff_j_values.append(coeff_j)
                neighbor_key = (neighbor_idx, neighbor_turn_idx)
                neighbor_group_keys.append(neighbor_key)
                neighbor_variances[neighbor_key] = neighbor_var

            if not pair_values:
                continue

            coeff_i_arr = np.asarray(coeff_i_values, dtype=float)
            coeff_j_arr = np.asarray(coeff_j_values, dtype=float)
            pair_arr = np.asarray(pair_values, dtype=float)
            covariance = np.outer(coeff_i_arr, coeff_i_arr) * current_var

            for neighbor_key, neighbor_var in neighbor_variances.items():
                active = np.array(
                    [idx for idx, key in enumerate(neighbor_group_keys) if key == neighbor_key],
                    dtype=int,
                )
                coeff_group = coeff_j_arr[active]
                covariance[np.ix_(active, active)] += (
                    np.outer(coeff_group, coeff_group) * neighbor_var
                )

            momentum[bpm_idx, turn_idx], momentum_var[bpm_idx, turn_idx] = _combine_momentum_blue(
                pair_arr,
                covariance,
            )

    return momentum, momentum_var


def calculate_transverse_pz_nbpm(
    orig_data: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    twiss_elements: pd.DataFrame | None = None,
    acdipole_element_name: str | None = None,
    acd_kicks: pd.DataFrame | None = None,
    info: bool = True,
    max_bpm_distance: int = 11,
    min_abs_cos: float = 0.10,
    bad_phase_threshold_rad: float = 2.0 * np.pi * 1.0e-2,
    measurement_variance_floor: float | None = None,
    ac_dipole_config: ACDipoleConfig | None = None,
) -> pd.DataFrame:
    window_radius = max(1, (int(max_bpm_distance) - 1) // 2)
    LOGGER.info(
        "Calculating N-BPM transverse momentum - max_bpm_distance=%s, window_radius=%s",
        max_bpm_distance,
        window_radius,
    )

    features = validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")

    variance_floor = (
        float(measurement_variance_floor)
        if measurement_variance_floor is not None
        else float(POSITION_STD_DEV**2)
    )
    data = _sanitize_measurement_variances(data, variance_floor)

    tws_bpm = _normalise_twiss(tws)
    tws_bpm_names = set(tws_bpm.index).intersection(data["name"].astype(str).str.upper().unique())
    data = data.copy(deep=True)
    data["name"] = data["name"].astype(str).str.upper()
    data = data[data["name"].isin(tws_bpm_names)].copy(deep=True)
    tws_bpm = tws_bpm.loc[tws_bpm.index.isin(tws_bpm_names)].copy(deep=True)
    bpm_names = tws_bpm.index.astype(str).tolist()
    twiss_reference = None if twiss_elements is None else _normalise_twiss(twiss_elements)

    acd_result: pd.DataFrame | None = None
    data_for_acd = data.copy(deep=True)
    if ac_dipole_config is not None:
        if acd_kicks is not None:
            raise ValueError("Provide either acd_kicks or ac_dipole_config to N-BPM, not both")
        acd_result = run_ac_dipole_reconstruction(data_for_acd, tws, ac_dipole_config)
        acd_kicks = acd_result
        if acdipole_element_name is None:
            acdipole_element_name = str(
                acd_result.attrs.get("acd_marker", ac_dipole_config.ac_dipole_marker)
            )
        if twiss_reference is None:
            twiss_reference = _normalise_twiss(ac_dipole_config.model.twiss_elements)

    data = remove_closed_orbit(data, tws_bpm)
    observation_cube = _build_observation_cube(data, bpm_names)
    kick_by_turn_x: np.ndarray | None = None
    kick_by_turn_y: np.ndarray | None = None
    if acd_kicks is not None:
        kick_frame = acd_kicks.copy(deep=True)
        if "row_type" in kick_frame.columns:
            kick_frame = kick_frame.loc[kick_frame["row_type"].fillna("summary") == "summary"].copy(
                deep=True
            )
        kick_frame["turn"] = kick_frame["turn"].astype(int)
        kick_col_x = "dpx" if "dpx" in kick_frame.columns else "dpx_fit_rad"
        kick_col_y = "dpy" if "dpy" in kick_frame.columns else "dpy_fit_rad"
        kick_frame = kick_frame.set_index("turn")
        kick_by_turn_x = (
            kick_frame[kick_col_x]
            .reindex(
                observation_cube.turn_values,
                fill_value=0.0,
            )
            .to_numpy(dtype=float)
        )
        kick_by_turn_y = (
            kick_frame[kick_col_y]
            .reindex(
                observation_cube.turn_values,
                fill_value=0.0,
            )
            .to_numpy(dtype=float)
        )
    pair_templates_x = _build_pair_templates(
        tws_bpm,
        twiss_elements=twiss_reference,
        plane="x",
        max_bpm_distance=window_radius,
        min_abs_cos=min_abs_cos,
        bad_phase_threshold_rad=bad_phase_threshold_rad,
        excluded_element_name=acdipole_element_name,
        allow_excluded_crossing=acd_kicks is not None and acdipole_element_name is not None,
    )
    pair_templates_y = _build_pair_templates(
        tws_bpm,
        twiss_elements=twiss_reference,
        plane="y",
        max_bpm_distance=window_radius,
        min_abs_cos=min_abs_cos,
        bad_phase_threshold_rad=bad_phase_threshold_rad,
        excluded_element_name=acdipole_element_name,
        allow_excluded_crossing=acd_kicks is not None and acdipole_element_name is not None,
    )

    px_cube, var_px_cube = _reconstruct_plane_cube(
        bpm_names=bpm_names,
        observation_cube=observation_cube,
        pair_templates=pair_templates_x,
        plane="x",
        kick_by_turn=kick_by_turn_x,
    )
    py_cube, var_py_cube = _reconstruct_plane_cube(
        bpm_names=bpm_names,
        observation_cube=observation_cube,
        pair_templates=pair_templates_y,
        plane="y",
        kick_by_turn=kick_by_turn_y,
    )
    result = data[["name", "turn", "x", "y"]].copy(deep=True)
    result["px"] = px_cube[observation_cube.row_bpm_idx, observation_cube.row_turn_idx]
    result["py"] = py_cube[observation_cube.row_bpm_idx, observation_cube.row_turn_idx]
    result["var_px"] = var_px_cube[observation_cube.row_bpm_idx, observation_cube.row_turn_idx]
    result["var_py"] = var_py_cube[observation_cube.row_bpm_idx, observation_cube.row_turn_idx]
    result = restore_closed_orbit_and_reference_momenta(result, tws_bpm)
    if ac_dipole_config is not None:
        result = apply_ac_dipole_bpm_overrides(
            result=result,
            data=data_for_acd,
            tws=tws,
            config=ac_dipole_config,
            acd_result=acd_result,
        )
    elif (
        acd_kicks is not None
        and {
            "px_bpm_upstream_cleaned",
            "py_bpm_upstream_cleaned",
            "px_bpm_downstream_cleaned",
            "py_bpm_downstream_cleaned",
        }.issubset(acd_kicks.columns)
        and ("bpm_upstream" in acd_kicks.columns or "bpm_upstream" in acd_kicks.attrs)
        and ("bpm_downstream" in acd_kicks.columns or "bpm_downstream" in acd_kicks.attrs)
    ):
        result = apply_precomputed_ac_dipole_bpm_overrides(
            result=result,
            acd_result=acd_kicks,
        )

    orig_order = orig_data.copy(deep=True)
    orig_order["name"] = orig_order["name"].astype(str).str.upper()
    result = (
        result.set_index(["name", "turn"])
        .reindex(orig_order.set_index(["name", "turn"]).index)
        .reset_index()
    )

    for col in OUT_COLS:
        if col not in result.columns:
            if col in orig_order.columns:
                result[col] = orig_order[col]
            else:
                raise KeyError(
                    f"Required output column {col!r} is missing from N-BPM reconstruction"
                )

    diagnostics(orig_order, result, result, result, info, features)
    return result[OUT_COLS]
