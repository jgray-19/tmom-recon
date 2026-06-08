"""AC-dipole state cleaning utilities.

This module combines noisy upstream/downstream BPM state estimates into a
physically consistent pre-/post-kick state at the AC dipole.

Two key numerical building blocks are implemented here:
1. A sparse weighted fit for the known kick waveform parameters.
2. A regularized linear solve for smoothed pre-kick momentum with the kick fixed.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.linalg import solveh_banded
from scipy.optimize import minimize_scalar
from scipy.sparse import linalg as spla

from .models import ACDipoleHarmonicFit, ACDipoleStateEstimate, ACDipoleStateSeries

LOGGER = logging.getLogger(__name__)
_MIN_TUNE = 1.0e-6
_MAX_TUNE = 0.5 - 1.0e-6
_TUNE_REFINE_GRID_POINTS = 17


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
    )
    return (
        aligned[value_col].to_numpy(dtype=float),
        aligned[variance_col].to_numpy(dtype=float),
    )


def _align_estimate_component(
    turns: np.ndarray,
    estimate_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    aligned_pairs = [
        _align_table_columns(
            turns,
            table,
            value_col=value_col,
            variance_col=variance_col,
        )
        for table in estimate_tables
    ]
    values, variances = zip(*aligned_pairs)
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


def _combine_marker_transverse_positions(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine the ACD marker transverse positions into one common same-turn state.

    The AC dipole is modeled as an instantaneous kick at the marker: it changes
    ``px``/``py`` but not ``x``/``y``.  That means upstream- and
    downstream-transported states for the same turn should refer to the same
    marker position, modulo reconstruction noise.  We therefore average all
    available ``x`` and ``y`` marker estimates into a single common position
    per turn and use that shared trajectory for both the pre-kick and
    post-kick ACD states.
    """
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
    return x_common, y_common, var_x_common, var_y_common


def _build_harmonic_design(turns: np.ndarray, tune: float) -> np.ndarray:
    omega_turn = 2.0 * np.pi * float(tune)
    return np.column_stack(
        [
            np.sin(omega_turn * turns),
            np.cos(omega_turn * turns),
            np.ones_like(turns),
        ]
    )


def _collect_kick_fit_observations(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return consolidated weighted observations for the harmonic kick fit."""
    up_turn_idx_parts: list[np.ndarray] = []
    up_rhs_parts: list[np.ndarray] = []
    up_weight_parts: list[np.ndarray] = []
    for table in upstream_tables:
        values, variances = _align_table_columns(
            turns,
            table,
            value_col=value_col,
            variance_col=variance_col,
        )
        valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
        idxs = np.flatnonzero(valid)
        up_turn_idx_parts.append(idxs)
        up_rhs_parts.append(values[idxs])
        up_weight_parts.append(1.0 / np.sqrt(variances[idxs]))

    dn_turn_idx_parts: list[np.ndarray] = []
    dn_rhs_parts: list[np.ndarray] = []
    dn_weight_parts: list[np.ndarray] = []
    for table in downstream_tables:
        values, variances = _align_table_columns(
            turns,
            table,
            value_col=value_col,
            variance_col=variance_col,
        )
        valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
        idxs = np.flatnonzero(valid)
        dn_turn_idx_parts.append(idxs)
        dn_rhs_parts.append(values[idxs])
        dn_weight_parts.append(1.0 / np.sqrt(variances[idxs]))

    up_turn_idx = np.concatenate(up_turn_idx_parts) if up_turn_idx_parts else np.empty(0, int)
    dn_turn_idx = np.concatenate(dn_turn_idx_parts) if dn_turn_idx_parts else np.empty(0, int)
    up_rhs = np.concatenate(up_rhs_parts) if up_rhs_parts else np.empty(0, float)
    dn_rhs = np.concatenate(dn_rhs_parts) if dn_rhs_parts else np.empty(0, float)
    up_weights = np.concatenate(up_weight_parts) if up_weight_parts else np.empty(0, float)
    dn_weights = np.concatenate(dn_weight_parts) if dn_weight_parts else np.empty(0, float)

    n_up = len(up_turn_idx)
    n_dn = len(dn_turn_idx)
    if n_up + n_dn == 0:
        raise ValueError(
            f"No valid {value_col} observations were available for AC-dipole kick fitting"
        )

    turn_array = np.concatenate([up_turn_idx, dn_turn_idx])
    rhs_array = np.concatenate([up_rhs, dn_rhs])
    weight_array = np.concatenate([up_weights, dn_weights])
    return turn_array, rhs_array, weight_array, dn_turn_idx, n_up


def _solve_known_kick_fit(
    turns: np.ndarray,
    *,
    tune: float,
    turn_array: np.ndarray,
    rhs_array: np.ndarray,
    weight_array: np.ndarray,
    downstream_turn_idx: np.ndarray,
    n_upstream_obs: int,
) -> tuple[ACDipoleHarmonicFit, float]:
    """Solve the weighted sparse harmonic fit for a specific tune value."""
    design = _build_harmonic_design(turns, tune)
    n_turns = len(turns)
    n_obs = len(turn_array)
    n_params = n_turns + design.shape[1]

    obs_rows = np.arange(n_obs)
    dn_obs_rows = np.arange(n_upstream_obs, n_obs)
    dn_design = design[downstream_turn_idx]
    design_obs_rows = np.repeat(dn_obs_rows, 3)
    design_cols = np.tile(np.arange(n_turns, n_turns + 3), len(downstream_turn_idx))
    design_vals = dn_design.ravel()

    coo_rows = np.concatenate([obs_rows, design_obs_rows])
    coo_cols = np.concatenate([turn_array, design_cols])
    coo_data = np.concatenate([np.ones(n_obs), design_vals])

    matrix_sparse = sp.coo_matrix(
        (coo_data, (coo_rows, coo_cols)), shape=(n_obs, n_params), dtype=float
    ).tocsr()

    weighted_rhs = rhs_array * weight_array

    weights_diag = sp.diags(weight_array, format="csr")
    weighted_matrix = weights_diag @ matrix_sparse

    try:
        ata = weighted_matrix.T @ weighted_matrix
        atb = weighted_matrix.T @ weighted_rhs
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            solution = spla.spsolve(ata.tocsr(), atb)
        if isinstance(solution, np.matrix):
            solution = np.asarray(solution).ravel()
        if not np.all(np.isfinite(solution)):
            raise ValueError("Sparse harmonic fit produced non-finite coefficients")
    except (RuntimeError, ValueError, spla.MatrixRankWarning):
        weighted_matrix_dense = weighted_matrix.toarray()
        solution, *_ = np.linalg.lstsq(weighted_matrix_dense, weighted_rhs, rcond=None)

    residual = weighted_matrix @ solution - weighted_rhs
    weighted_rss = float(np.dot(residual, residual))
    sin_coeff, cos_coeff, offset = solution[n_turns:]
    fitted = design @ solution[n_turns:]
    return (
        ACDipoleHarmonicFit(
            tune=float(tune),
            amplitude=float(np.hypot(sin_coeff, cos_coeff)),
            phase=float(np.arctan2(cos_coeff, sin_coeff)),
            offset=float(offset),
            fitted=fitted,
        ),
        weighted_rss,
    )


def _local_tune_refinement_window(turns: np.ndarray) -> float:
    """Choose a conservative local tune-search half-width around the supplied tune."""
    n_turns = max(int(len(turns)), 1)
    fft_resolution = 1.0 / float(n_turns)
    return min(0.02, max(4.0 * fft_resolution, 0.002))


def _refine_known_kick_fit(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
    tune_hint: float,
) -> ACDipoleHarmonicFit:
    """Fit the harmonic kick waveform, refining tune locally to improve phase/amplitude."""
    turn_array, rhs_array, weight_array, dn_turn_idx, n_up = _collect_kick_fit_observations(
        turns,
        upstream_tables,
        downstream_tables,
        value_col=value_col,
        variance_col=variance_col,
    )

    tune_centre = float(np.clip(tune_hint, _MIN_TUNE, _MAX_TUNE))
    half_width = _local_tune_refinement_window(turns)
    lower = max(_MIN_TUNE, tune_centre - half_width)
    upper = min(_MAX_TUNE, tune_centre + half_width)

    def solve_for_tune(candidate_tune: float) -> tuple[ACDipoleHarmonicFit, float]:
        return _solve_known_kick_fit(
            turns,
            tune=float(np.clip(candidate_tune, _MIN_TUNE, _MAX_TUNE)),
            turn_array=turn_array,
            rhs_array=rhs_array,
            weight_array=weight_array,
            downstream_turn_idx=dn_turn_idx,
            n_upstream_obs=n_up,
        )

    if upper <= lower:
        best_fit, _ = solve_for_tune(tune_centre)
        return best_fit

    coarse_grid = np.linspace(lower, upper, _TUNE_REFINE_GRID_POINTS)
    coarse_scores = np.empty_like(coarse_grid)
    coarse_fits: list[ACDipoleHarmonicFit] = []
    for idx, candidate in enumerate(coarse_grid):
        fit, score = solve_for_tune(float(candidate))
        coarse_fits.append(fit)
        coarse_scores[idx] = score

    best_idx = int(np.argmin(coarse_scores))
    best_fit = coarse_fits[best_idx]
    best_score = float(coarse_scores[best_idx])

    bracket_low_idx = max(best_idx - 1, 0)
    bracket_high_idx = min(best_idx + 1, len(coarse_grid) - 1)
    bracket_low = float(coarse_grid[bracket_low_idx])
    bracket_high = float(coarse_grid[bracket_high_idx])

    if bracket_high > bracket_low:
        optimum = minimize_scalar(
            lambda tune: solve_for_tune(float(tune))[1],
            bounds=(bracket_low, bracket_high),
            method="bounded",
            options={"xatol": max(1.0e-6, (bracket_high - bracket_low) / 200.0)},
        )
        refined_fit, refined_score = solve_for_tune(float(optimum.x))
        if refined_score <= best_score:
            best_fit = refined_fit

    LOGGER.info(
        "Refined %s harmonic tune from %.6f to %.6f (local window +/- %.6f)",
        value_col,
        tune_hint,
        best_fit.tune,
        half_width,
    )
    return best_fit


def _sparse_to_upper_banded(mat: sp.csr_matrix, bandwidth: int) -> np.ndarray:
    """Extract the upper banded form of a symmetric sparse matrix.

    Returns ab with shape (bandwidth+1, n) in the format expected by
    scipy.linalg.solveh_banded (lower=False): ab[bandwidth-k, k:] = diagonal k.
    """
    n = mat.shape[0]
    ab = np.zeros((bandwidth + 1, n), dtype=float)
    for k in range(bandwidth + 1):
        diag = mat.diagonal(k)
        ab[bandwidth - k, k:] = diag
    return ab


def _build_second_difference_operator(n_turns: int) -> sp.csr_matrix:
    """Return sparse second-difference operator D2 for turn-to-turn smoothness.

    For a vector ``p`` of per-turn momentum values:
    ``(D2 p)_t = p_t - 2 p_{t+1} + p_{t+2}``.

    Penalizing ``||D2 p||^2`` suppresses rapid curvature while preserving slow
    trends, which is a natural regularizer for turn-domain trajectories.
    """
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
    """Solve smoothed pre-/post-kick momentum with kick waveform treated as known.

    Let ``u_t`` be combined upstream estimate and ``d_t`` combined downstream estimate.
    Given fitted kick ``k_t``, downstream estimates are mapped to pre-kick space as
    ``d_t - k_t``. We solve for ``p_t`` by minimizing:

    ``sum_t w_u(t) * (p_t - u_t)^2 + sum_t w_d(t) * (p_t - (d_t - k_t))^2
       + lambda * ||D2 p||^2``

    where ``w`` are inverse variances and ``D2`` is the second-difference operator.

    This gives sparse normal equations:
    ``(W_u + W_d + lambda * D2^T D2) p = W_u u + W_d (d - k)``.

    Variance estimate:
    - We need only diagonal entries of the covariance.
    - Fast path uses dense ``inv`` on the assembled system matrix.
    - ``pinv`` fallback is kept for singular/near-singular cases.
    """
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

    # Extract diagonal of A^{-1} using the banded SPD structure.
    # system_matrix = W + λD2^T D2 is pentadiagonal (bandwidth 2), so Cholesky
    # factorization and back-substitution are O(bandwidth × T) per RHS, giving
    # O(bandwidth × T^2) total vs O(T^3) for dense inversion.
    try:
        ab = _sparse_to_upper_banded(system_matrix.tocsr(), bandwidth=2)
        inv_cols = solveh_banded(ab, np.eye(n_turns), lower=False, check_finite=False)
        var_pre = np.clip(np.diag(inv_cols), 0.0, None)
    except (np.linalg.LinAlgError, ValueError):
        # Robust fallback for singular/near-singular systems.
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
    """Produce physically consistent pre-/post-kick state estimates.

    Pipeline:
    1. Combine x/y marker estimates from all selected BPM reconstructions into
       one common same-turn ACD position.
    2. Fit harmonic kick waveforms for px and py.
    3. Solve smoothed pre-kick momentum with fitted kick constraint.
    4. Reconstruct post-kick momentum via ``post = pre + fitted_kick``.
    """
    x_common, y_common, var_x_common, var_y_common = _combine_marker_transverse_positions(
        turns,
        upstream_tables,
        downstream_tables,
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
