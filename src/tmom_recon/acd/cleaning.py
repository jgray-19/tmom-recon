from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse import linalg as spla

from .models import ACDipoleHarmonicFit, ACDipoleStateEstimate, ACDipoleStateSeries


def _align_estimate_component(
    turns: np.ndarray,
    estimate_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    variances = []
    for table in estimate_tables:
        aligned = pd.DataFrame({"turn": turns}).merge(
            table[["turn", value_col, variance_col]],
            on="turn",
            how="left",
            copy=False,
        )
        values.append(aligned[value_col].to_numpy(dtype=float))
        variances.append(aligned[variance_col].to_numpy(dtype=float))
    return np.vstack(values), np.vstack(variances)


def _combine_many_estimates(
    values: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
    safe_variances = np.where(valid, variances, np.inf)
    inv_vars = np.where(valid, 1.0 / safe_variances, 0.0)
    weight_sum = np.sum(inv_vars, axis=0)
    combined = np.divide(
        np.sum(values * inv_vars, axis=0),
        weight_sum,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=weight_sum > 0.0,
    )
    combined_var = np.divide(
        1.0,
        weight_sum,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=weight_sum > 0.0,
    )
    return combined, combined_var


def _combine_state_tables(
    turns: np.ndarray, estimate_tables: list[pd.DataFrame]
) -> ACDipoleStateEstimate:
    x, var_x = _combine_many_estimates(
        *_align_estimate_component(turns, estimate_tables, value_col="x", variance_col="var_x")
    )
    px, var_px = _combine_many_estimates(
        *_align_estimate_component(turns, estimate_tables, value_col="px", variance_col="var_px")
    )
    y, var_y = _combine_many_estimates(
        *_align_estimate_component(turns, estimate_tables, value_col="y", variance_col="var_y")
    )
    py, var_py = _combine_many_estimates(
        *_align_estimate_component(turns, estimate_tables, value_col="py", variance_col="var_py")
    )
    return ACDipoleStateEstimate(
        state=ACDipoleStateSeries(x, px, y, py),
        var_x=var_x,
        var_px=var_px,
        var_y=var_y,
        var_py=var_py,
    )


def _build_harmonic_design(turns: np.ndarray, tune: float) -> np.ndarray:
    omega_turn = 2.0 * np.pi * float(tune)
    return np.column_stack(
        [
            np.sin(omega_turn * turns),
            np.cos(omega_turn * turns),
            np.ones_like(turns),
        ]
    )


def _align_table_columns(
    turns: np.ndarray,
    table: pd.DataFrame,
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = pd.DataFrame({"turn": turns}).merge(
        table[["turn", value_col, variance_col]],
        on="turn",
        how="left",
        copy=False,
    )
    return (
        aligned[value_col].to_numpy(dtype=float),
        aligned[variance_col].to_numpy(dtype=float),
    )


def _refine_known_kick_fit(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
    tune_hint: float,
) -> ACDipoleHarmonicFit:
    design = _build_harmonic_design(turns, tune_hint)
    n_turns = len(turns)
    n_params = n_turns + design.shape[1]

    rows: list[np.ndarray] = []
    rhs: list[float] = []
    weights: list[float] = []

    for table in upstream_tables:
        values, variances = _align_table_columns(
            turns,
            table,
            value_col=value_col,
            variance_col=variance_col,
        )
        valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
        for idx in np.flatnonzero(valid):
            row = np.zeros(n_params, dtype=float)
            row[idx] = 1.0
            rows.append(row)
            rhs.append(float(values[idx]))
            weights.append(float(1.0 / np.sqrt(variances[idx])))

    for table in downstream_tables:
        values, variances = _align_table_columns(
            turns,
            table,
            value_col=value_col,
            variance_col=variance_col,
        )
        valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
        for idx in np.flatnonzero(valid):
            row = np.zeros(n_params, dtype=float)
            row[idx] = 1.0
            row[n_turns:] = design[idx]
            rows.append(row)
            rhs.append(float(values[idx]))
            weights.append(float(1.0 / np.sqrt(variances[idx])))

    if not rows:
        raise ValueError(
            f"No valid {value_col} observations were available for AC-dipole kick fitting"
        )

    matrix = np.vstack(rows)
    rhs_array = np.asarray(rhs, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    weighted_matrix = matrix * weight_array[:, None]
    weighted_rhs = rhs_array * weight_array
    solution, *_ = np.linalg.lstsq(weighted_matrix, weighted_rhs, rcond=None)

    sin_coeff, cos_coeff, offset = solution[n_turns:]
    fitted = design @ solution[n_turns:]
    return ACDipoleHarmonicFit(
        tune=float(tune_hint),
        amplitude=float(np.hypot(sin_coeff, cos_coeff)),
        phase=float(np.arctan2(cos_coeff, sin_coeff)),
        offset=float(offset),
        fitted=fitted,
    )


def _build_second_difference_operator(n_turns: int) -> sp.csr_matrix:
    if n_turns <= 2:
        return sp.csr_matrix((0, n_turns), dtype=float)
    return sp.diags(
        diagonals=[
            np.ones(n_turns - 2, dtype=float),
            -2.0 * np.ones(n_turns - 2, dtype=float),
            np.ones(n_turns - 2, dtype=float),
        ],
        offsets=[0, 1, 2],
        shape=(n_turns - 2, n_turns),
        format="csr",
    )


def _solve_smoothed_pre_momentum_with_known_kick(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
    kick_values: np.ndarray,
    smooth_lambda: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    upstream_hat, upstream_var = _combine_many_estimates(
        *_align_estimate_component(
            turns,
            upstream_tables,
            value_col=value_col,
            variance_col=variance_col,
        )
    )
    downstream_hat, downstream_var = _combine_many_estimates(
        *_align_estimate_component(
            turns,
            downstream_tables,
            value_col=value_col,
            variance_col=variance_col,
        )
    )

    mapped_downstream_hat = downstream_hat - kick_values

    valid_upstream = np.isfinite(upstream_hat) & np.isfinite(upstream_var) & (upstream_var > 0.0)
    valid_downstream = (
        np.isfinite(mapped_downstream_hat) & np.isfinite(downstream_var) & (downstream_var > 0.0)
    )

    if not np.any(valid_upstream | valid_downstream):
        raise ValueError(f"No valid {value_col} observations were available for AC-dipole cleaning")

    upstream_weights = np.where(valid_upstream, 1.0 / upstream_var, 0.0)
    downstream_weights = np.where(valid_downstream, 1.0 / downstream_var, 0.0)

    rhs = np.where(valid_upstream, upstream_weights * upstream_hat, 0.0) + np.where(
        valid_downstream,
        downstream_weights * mapped_downstream_hat,
        0.0,
    )

    n_turns = len(turns)
    smoothing_operator = _build_second_difference_operator(n_turns)
    system_matrix = sp.diags(upstream_weights + downstream_weights, format="csr")
    if smooth_lambda > 0.0 and smoothing_operator.shape[0] > 0:
        system_matrix = system_matrix + float(smooth_lambda) * (
            smoothing_operator.T @ smoothing_operator
        )

    pre_values = spla.spsolve(system_matrix, rhs)
    post_values = pre_values + kick_values

    covariance = np.linalg.pinv(system_matrix.toarray(), hermitian=True)
    var_pre = np.clip(np.diag(covariance), 0.0, None)
    # Kick is treated as known from the harmonic fit, so post variance equals pre variance.
    var_post = var_pre.copy()
    return pre_values, post_values, var_pre, var_post


def _clean_ac_dipole_states(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    dpx_tune: float,
    dpy_tune: float,
    smooth_lambda: float = 1,
) -> tuple[ACDipoleStateEstimate, ACDipoleStateEstimate, ACDipoleHarmonicFit, ACDipoleHarmonicFit]:
    x_common, var_x_common = _combine_many_estimates(
        *_align_estimate_component(
            turns,
            upstream_tables + downstream_tables,
            value_col="x",
            variance_col="var_x",
        )
    )
    y_common, var_y_common = _combine_many_estimates(
        *_align_estimate_component(
            turns,
            upstream_tables + downstream_tables,
            value_col="y",
            variance_col="var_y",
        )
    )
    dpx_fit = _refine_known_kick_fit(
        turns,
        upstream_tables,
        downstream_tables,
        value_col="px",
        variance_col="var_px",
        tune_hint=dpx_tune,
    )
    dpy_fit = _refine_known_kick_fit(
        turns,
        upstream_tables,
        downstream_tables,
        value_col="py",
        variance_col="var_py",
        tune_hint=dpy_tune,
    )

    px_pre, px_post, var_px_pre, var_px_post = _solve_smoothed_pre_momentum_with_known_kick(
        turns,
        upstream_tables,
        downstream_tables,
        value_col="px",
        variance_col="var_px",
        kick_values=dpx_fit.fitted,
        smooth_lambda=smooth_lambda,
    )
    py_pre, py_post, var_py_pre, var_py_post = _solve_smoothed_pre_momentum_with_known_kick(
        turns,
        upstream_tables,
        downstream_tables,
        value_col="py",
        variance_col="var_py",
        kick_values=dpy_fit.fitted,
        smooth_lambda=smooth_lambda,
    )

    pre_kick = ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            x_common,
            px_pre,
            y_common,
            py_pre,
        ),
        var_x=var_x_common,
        var_px=var_px_pre,
        var_y=var_y_common,
        var_py=var_py_pre,
    )
    post_kick = ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            x_common,
            px_post,
            y_common,
            py_post,
        ),
        var_x=var_x_common,
        var_px=var_px_post,
        var_y=var_y_common,
        var_py=var_py_post,
    )
    return pre_kick, post_kick, dpx_fit, dpy_fit
