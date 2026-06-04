"""AC-dipole state cleaning utilities.

This module combines noisy upstream/downstream BPM state estimates into a
physically consistent pre-/post-kick state at the AC dipole.

Two key numerical building blocks are implemented here:
1. A sparse weighted fit for the known kick waveform parameters.
2. A regularized linear solve for smoothed pre-kick momentum with the kick fixed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.linalg import solveh_banded
from scipy.sparse import linalg as spla

from .models import ACDipoleHarmonicFit, ACDipoleStateEstimate, ACDipoleStateSeries

LOGGER = logging.getLogger(__name__)


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
            table[["turn", value_col, variance_col]], on="turn", how="left"
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
    """Fit the harmonic kick waveform using a sparse weighted least-squares model.

    Model construction (one equation per valid observation):
    - Upstream BPM estimate at turn ``t``: ``m_t = p_t + e_t``
    - Downstream BPM estimate at turn ``t``: ``m_t = p_t + h_t(theta) + e_t``

    where:
    - ``p_t`` is the unknown per-turn pre-kick momentum,
    - ``h_t(theta)`` is a 3-parameter harmonic model (sin, cos, offset),
    - ``e_t`` is measurement noise.

    Unknown vector layout is ``[p_0, ..., p_{T-1}, sin_coeff, cos_coeff, offset]``.

        Why sparse:
        - Each observation row touches exactly one per-turn variable and at most three
            harmonic columns, so the system matrix is extremely sparse.
        - Solving this in dense form is unnecessarily expensive for long turn series.

    Solve strategy:
    - Build weighted sparse system ``A x ~= b``.
    - Solve normal equations ``(A^T A) x = A^T b`` with ``scipy.sparse.linalg.spsolve``.
    - Fall back to dense ``np.linalg.lstsq`` only if sparse solve fails.
    """
    design = _build_harmonic_design(turns, tune_hint)
    n_turns = len(turns)
    n_params = n_turns + design.shape[1]

    # Collect upstream observations in bulk — no Python element loop.
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

    # Collect downstream observations in bulk.
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

    n_obs = n_up + n_dn
    turn_array = np.concatenate([up_turn_idx, dn_turn_idx])

    # Build sparse COO data with pure numpy — no Python element loop.
    # Identity block: one entry per observation (col = turn index).
    obs_rows = np.arange(n_obs)
    # Design block: 3 entries per downstream observation (cols = n_turns+0/1/2).
    dn_obs_rows = np.arange(n_up, n_obs)
    dn_design = design[dn_turn_idx]  # (n_dn, 3)
    design_obs_rows = np.repeat(dn_obs_rows, 3)  # (n_dn*3,)
    design_cols = np.tile(np.arange(n_turns, n_turns + 3), n_dn)  # (n_dn*3,)
    design_vals = dn_design.ravel()  # (n_dn*3,)

    coo_rows = np.concatenate([obs_rows, design_obs_rows])
    coo_cols = np.concatenate([turn_array, design_cols])
    coo_data = np.concatenate([np.ones(n_obs), design_vals])

    # Build sparse matrix in COO format, then convert to CSR for efficient operations
    matrix_sparse = sp.coo_matrix(
        (coo_data, (coo_rows, coo_cols)), shape=(n_obs, n_params), dtype=float
    ).tocsr()

    # Apply weights and build weighted system
    weight_array = np.concatenate([up_weights, dn_weights])
    rhs_array = np.concatenate([up_rhs, dn_rhs])
    weighted_rhs = rhs_array * weight_array

    # Weight the sparse matrix rows
    weights_diag = sp.diags(weight_array, format="csr")
    weighted_matrix = weights_diag @ matrix_sparse

    # Solve weighted normal equations using sparse direct solve.
    # This removes the dense lstsq hotspot for long turn counts.
    try:
        ata = weighted_matrix.T @ weighted_matrix
        atb = weighted_matrix.T @ weighted_rhs
        solution = spla.spsolve(ata.tocsr(), atb)
        if isinstance(solution, np.matrix):
            solution = np.asarray(solution).ravel()
    except (RuntimeError, ValueError):
        # Fallback to dense lstsq if sparse solve fails
        weighted_matrix_dense = weighted_matrix.toarray()
        solution, *_ = np.linalg.lstsq(weighted_matrix_dense, weighted_rhs, rcond=None)

    sin_coeff, cos_coeff, offset = solution[n_turns:]
    fitted = design @ solution[n_turns:]
    return ACDipoleHarmonicFit(
        tune=float(tune_hint),
        amplitude=float(np.hypot(sin_coeff, cos_coeff)),
        phase=float(np.arctan2(cos_coeff, sin_coeff)),
        offset=float(offset),
        fitted=fitted,
    )


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
    1. Combine x/y state estimates from all selected BPM reconstructions.
    2. Fit harmonic kick waveforms for px and py.
    3. Solve smoothed pre-kick momentum with fitted kick constraint.
    4. Reconstruct post-kick momentum via ``post = pre + fitted_kick``.
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
