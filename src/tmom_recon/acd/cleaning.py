"""AC-dipole state cleaning utilities.

Combines noisy upstream/downstream BPM state estimates into physically
consistent pre-/post-kick states at the AC-dipole marker.

Two numerical building blocks:

1. A weighted harmonic fit for the kick waveform parameters, with local
   tune refinement via bounded scalar optimisation.
2. A regularised linear solve for the smoothed pre-kick momentum with the
   kick waveform treated as known, using a banded-SPD Cholesky fast-path.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.linalg import solveh_banded
from scipy.optimize import least_squares
from scipy.sparse import linalg as spla

from .models import (
    ACDipoleCleaningResult,
    ACDipoleHarmonicFit,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)

LOGGER = logging.getLogger(__name__)

_MIN_TUNE = 1.0e-6
_MAX_TUNE = 0.5 - 1.0e-6
_FIT_TUNE_MIN_TURNS = 999
_DEFAULT_TUNE_SIGMA = 1.0e-2

# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------


def _align_table_columns(
    turns: np.ndarray,
    table: pd.DataFrame,
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-join *table* onto *turns* and extract a value/variance pair.

    Args:
        turns: Reference turn array to align against.
        table: DataFrame containing at least ``"turn"``, *value_col*, and
            *variance_col* columns.
        value_col: Name of the value column to extract.
        variance_col: Name of the variance column to extract.

    Returns:
        A ``(values, variances)`` tuple, each shape ``(len(turns),)``.
        Missing turns produce ``NaN``.
    """
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
    """Align *value_col* and *variance_col* from multiple tables onto *turns*.

    Args:
        turns: Reference turn array.
        estimate_tables: List of DataFrames, each containing ``"turn"``,
            *value_col*, and *variance_col* columns.
        value_col: Name of the value column.
        variance_col: Name of the variance column.

    Returns:
        ``(values, variances)`` where each array has shape
        ``(len(estimate_tables), len(turns))``.
    """
    aligned_pairs = [
        _align_table_columns(turns, table, value_col=value_col, variance_col=variance_col)
        for table in estimate_tables
    ]
    values, variances = zip(*aligned_pairs)
    return np.vstack(values), np.vstack(variances)


# ---------------------------------------------------------------------------
# Inverse-variance combination
# ---------------------------------------------------------------------------


def _combine_many_estimates(
    values: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine multiple noisy estimates via inverse-variance weighting.

    Args:
        values: Shape ``(n_sources, n_turns)``.
        variances: Shape ``(n_sources, n_turns)``.  Non-positive or non-finite
            entries are treated as missing (weight = 0).

    Returns:
        ``(combined, combined_var)`` each shape ``(n_turns,)``.  Turns with no
        valid source produce ``NaN``.
    """
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
    """Inverse-variance combine per-coordinate estimates from multiple BPM tables.

    Args:
        turns: Reference turn array used for alignment.
        estimate_tables: List of DataFrames, each with columns
            ``x, px, y, py, var_x, var_px, var_y, var_py, turn``.

    Returns:
        An :class:`ACDipoleStateEstimate` with combined central values and
        variances at each turn.
    """
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
    # All t and pt should be identical across tables, so just take the first non-NaN value per turn
    t = np.zeros_like(x)
    pt, _ = _align_table_columns(turns, estimate_tables[0], value_col="pt", variance_col="var_x")
    return ACDipoleStateEstimate(
        state=ACDipoleStateSeries(x, px, y, py, t, pt),
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
    """Combine marker transverse positions from both upstream and downstream estimates.

    The AC dipole is modelled as an instantaneous kick: it changes ``px``/``py``
    but not ``x``/``y``. Upstream and downstream transported states for the same
    turn therefore share the same marker position modulo noise, so all available
    ``x`` and ``y`` estimates are pooled into a single common trajectory.

    Args:
        turns: Reference turn array.
        upstream_tables: Tracked-state tables from upstream BPMs.
        downstream_tables: Tracked-state tables from downstream BPMs.

    Returns:
        ``(x_common, y_common, var_x_common, var_y_common)`` each shape
        ``(n_turns,)``.
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


# ---------------------------------------------------------------------------
# Harmonic fitting
# ---------------------------------------------------------------------------


def _build_harmonic_design(turns: np.ndarray, tune: float) -> np.ndarray:
    """Build the three-column harmonic design matrix ``[sin, cos, 1]``."""
    omega_turn = 2.0 * np.pi * float(tune)
    return np.column_stack(
        [
            np.sin(omega_turn * turns),
            np.cos(omega_turn * turns),
            np.ones_like(turns),
        ]
    )


def _build_raw_kick_series(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the raw same-turn kick series and its propagated variance."""
    upstream_hat, upstream_var = _combine_many_estimates(
        *_align_estimate_component(
            turns, upstream_tables, value_col=value_col, variance_col=variance_col
        )
    )
    downstream_hat, downstream_var = _combine_many_estimates(
        *_align_estimate_component(
            turns, downstream_tables, value_col=value_col, variance_col=variance_col
        )
    )

    raw_kick = downstream_hat - upstream_hat
    raw_var = upstream_var + downstream_var

    valid = np.isfinite(raw_kick) & np.isfinite(raw_var) & (raw_var > 0.0)
    if not np.any(valid):
        raise ValueError(
            f"No valid {value_col} observations were available for AC-dipole kick fitting"
        )

    return raw_kick, raw_var


def _solve_harmonic_series_fit(
    turns: np.ndarray,
    *,
    tune: float,
    values: np.ndarray,
    variances: np.ndarray,
) -> tuple[ACDipoleHarmonicFit, float]:
    """Solve a weighted harmonic least-squares fit at fixed tune."""
    tune = float(np.clip(tune, _MIN_TUNE, _MAX_TUNE))
    design = _build_harmonic_design(turns, tune)

    valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
    obs = values[valid]
    design_valid = design[valid]

    weights = 1.0 / np.sqrt(variances[valid])
    weighted_rhs = obs * weights
    weighted_design = design_valid * weights[:, None]

    try:
        ata = weighted_design.T @ weighted_design
        atb = weighted_design.T @ weighted_rhs
        solution = np.linalg.solve(ata, atb)
        if not np.all(np.isfinite(solution)):
            raise ValueError("Harmonic fit produced non-finite coefficients")
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        solution, *_ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)

    residual = weighted_design @ solution - weighted_rhs
    weighted_rss = float(np.dot(residual, residual))

    sin_coeff, cos_coeff, offset = solution
    fitted = design @ solution

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


def _fit_harmonic_series_with_tune(
    turns: np.ndarray,
    *,
    tune_hint: float,
    tune_sigma: float,
    values: np.ndarray,
    variances: np.ndarray,
    log_label: str,
) -> ACDipoleHarmonicFit:
    """Fit harmonic kick waveform with tune included in the same fit.

    The fitted model is::

        k(t) = a sin(2πQt) + b cos(2πQt) + c

    The fitted parameters are ``a``, ``b``, ``c`` and ``Q``.  The supplied tune
    enters as a Gaussian prior, not as a hard search window::

        χ² = Σ ((yᵢ - kᵢ) / σᵢ)² + ((Q - Q₀) / σ_Q)²
    """
    if tune_sigma <= 0.0 or not np.isfinite(tune_sigma):
        raise ValueError("tune_sigma must be positive and finite")

    valid = np.isfinite(values) & np.isfinite(variances) & (variances > 0.0)
    if np.count_nonzero(valid) < 4:
        raise ValueError(
            f"At least four valid observations are required to fit {log_label} harmonic tune"
        )

    t_valid = turns[valid].astype(float)
    y_valid = values[valid].astype(float)
    inv_sigma = 1.0 / np.sqrt(variances[valid])

    tune0 = float(np.clip(tune_hint, _MIN_TUNE, _MAX_TUNE))

    initial_fit, initial_score = _solve_harmonic_series_fit(
        turns,
        tune=tune0,
        values=values,
        variances=variances,
    )

    # Grid search over tunes near the hint to find the best linear-fit starting
    # point.  Starting from the linear fit exactly at tune_hint fails when the
    # hint and true tunes are nearly orthogonal over the record
    # (N × |Q_hint - Q_true| ≈ integer): the linear fit yields amplitude ≈ 0, the
    # nonlinear optimizer starts at a saddle where ∂RSS/∂tune = 0 everywhere on
    # the zero-amplitude manifold, and the tune never moves.  The grid search
    # guarantees the optimizer begins near the global linear-fit minimum.
    _grid_step = 1.0 / max(len(t_valid), 1)  # Rayleigh resolution
    _n_half = max(3, int(np.ceil(2.0 * tune_sigma / _grid_step)))
    _tune_grid = np.unique(
        np.clip(
            tune0 + _grid_step * np.arange(-_n_half, _n_half + 1, dtype=float),
            _MIN_TUNE,
            _MAX_TUNE,
        )
    )
    _best_rss = np.inf
    _best_tune = tune0
    _best_fit = initial_fit
    for _tg in _tune_grid:
        _fg, _rss_g = _solve_harmonic_series_fit(
            turns, tune=float(_tg), values=values, variances=variances
        )
        if _rss_g < _best_rss:
            _best_rss = _rss_g
            _best_tune = float(_tg)
            _best_fit = _fg

    sin0 = _best_fit.amplitude * np.cos(_best_fit.phase)
    cos0 = _best_fit.amplitude * np.sin(_best_fit.phase)
    p0 = np.array([sin0, cos0, _best_fit.offset, _best_tune], dtype=float)

    amp_scale = max(float(_best_fit.amplitude), 1.0)
    offset_scale = max(abs(float(_best_fit.offset)), amp_scale, 1.0)

    # The Gaussian prior on tune is intentionally omitted here: the grid search
    # above already starts the optimizer near the data-driven minimum, so no
    # pull toward the hint is needed.  Including the prior blocks convergence
    # whenever the true tune lies more than ~1σ from the hint, because for small
    # signal amplitudes the prior penalty dominates the data-fit residuals.
    tune_scale = 1.0 / max(len(t_valid), 1)  # Rayleigh resolution

    def residual(params: np.ndarray) -> np.ndarray:
        sin_coeff, cos_coeff, offset, tune = params
        omega_turn = 2.0 * np.pi * tune
        model = (
            sin_coeff * np.sin(omega_turn * t_valid)
            + cos_coeff * np.cos(omega_turn * t_valid)
            + offset
        )
        return (model - y_valid) * inv_sigma

    result = least_squares(
        residual,
        p0,
        bounds=(
            [-np.inf, -np.inf, -np.inf, _MIN_TUNE],
            [np.inf, np.inf, np.inf, _MAX_TUNE],
        ),
        x_scale=[amp_scale, amp_scale, offset_scale, tune_scale],
        loss="linear",
        ftol=1.0e-12,
        xtol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=300,
    )

    if not result.success:
        LOGGER.warning(
            "%s harmonic tune fit did not fully converge: %s",
            log_label,
            result.message,
        )

    sin_coeff, cos_coeff, offset, tune = result.x
    design = _build_harmonic_design(turns, float(tune))
    fitted = design @ np.array([sin_coeff, cos_coeff, offset], dtype=float)

    weighted_rss_final = float(np.dot(result.fun, result.fun))

    LOGGER.info(
        "Fitted %s harmonic tune from hint %.8f to %.8f (grid best %.8f); "
        "initial weighted RSS %.6e, final weighted RSS %.6e",
        log_label,
        tune0,
        tune,
        _best_tune,
        initial_score,
        weighted_rss_final,
    )

    return ACDipoleHarmonicFit(
        tune=float(tune),
        amplitude=float(np.hypot(sin_coeff, cos_coeff)),
        phase=float(np.arctan2(cos_coeff, sin_coeff)),
        offset=float(offset),
        fitted=fitted,
    )


def _refine_known_kick_fit(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    value_col: str,
    variance_col: str,
    tune_hint: float,
    tune_sigma: float = _DEFAULT_TUNE_SIGMA,
) -> ACDipoleHarmonicFit:
    """Fit the harmonic kick waveform.

    For short records, the tune is held fixed because sub-1/N tune shifts are
    poorly identifiable.  For long records, tune is included in the weighted
    nonlinear fit with a Gaussian prior centred on the supplied tune.
    """
    raw_kick, raw_var = _build_raw_kick_series(
        turns,
        upstream_tables,
        downstream_tables,
        value_col=value_col,
        variance_col=variance_col,
    )

    valid = np.isfinite(raw_kick) & np.isfinite(raw_var) & (raw_var > 0.0)
    n_valid_turns = int(np.count_nonzero(valid))

    if n_valid_turns < _FIT_TUNE_MIN_TURNS:
        fit, score = _solve_harmonic_series_fit(
            turns,
            tune=tune_hint,
            values=raw_kick,
            variances=raw_var,
        )
        LOGGER.info(
            "Fitted %s harmonic with fixed tune %.8f over %d valid turns; weighted RSS %.6e",
            value_col,
            fit.tune,
            n_valid_turns,
            score,
        )
        return fit

    return _fit_harmonic_series_with_tune(
        turns,
        tune_hint=tune_hint,
        tune_sigma=tune_sigma,
        values=raw_kick,
        variances=raw_var,
        log_label=value_col,
    )


# ---------------------------------------------------------------------------
# Smoothed momentum solve
# ---------------------------------------------------------------------------


def _sparse_to_upper_banded(mat: sp.csr_matrix, bandwidth: int) -> np.ndarray:
    """Extract the upper banded form of a symmetric sparse matrix.

    Returns ``ab`` with shape ``(bandwidth+1, n)`` in the format expected by
    :func:`scipy.linalg.solveh_banded` with ``lower=False``:
    ``ab[bandwidth - k, k:]`` holds diagonal ``k``.

    Args:
        mat: Symmetric sparse matrix in CSR format.
        bandwidth: Number of super-diagonals to extract.

    Returns:
        Upper banded representation, shape ``(bandwidth+1, n)``.
    """
    n = mat.shape[0]
    ab = np.zeros((bandwidth + 1, n), dtype=float)
    for k in range(bandwidth + 1):
        ab[bandwidth - k, k:] = mat.diagonal(k)
    return ab


def _build_second_difference_operator(n_turns: int) -> sp.csr_matrix:
    """Return the sparse second-difference operator D2.

    For a vector ``p``, ``(D2 p)_t = p_t - 2 p_{t+1} + p_{t+2}``.
    Penalising ``‖D2 p‖²`` suppresses rapid curvature while preserving slow
    trends — a natural regulariser for turn-domain trajectories.

    Args:
        n_turns: Length of the turn series.

    Returns:
        Sparse CSR matrix of shape ``(n_turns-2, n_turns)``, or an empty
        ``(0, n_turns)`` matrix for ``n_turns <= 2``.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve smoothed pre-/post-kick momentum with the kick waveform fixed.

    Let ``u_t`` be the combined upstream estimate and ``d_t`` the downstream
    estimate per turn. With fitted kick ``k_t``, the downstream estimates are
    mapped to pre-kick space as ``d_t - k_t``. We solve for ``p_t`` by
    minimising::

        Σ_t w_u(t) (p_t - u_t)² + Σ_t w_d(t) (p_t - (d_t - k_t))²
            + λ ‖D2 p‖²

    where ``w`` are inverse variances and D2 is the second-difference
    operator. Post-kick values are then ``p_t + k_t``.

    Uses a banded Cholesky solve with ``pinv`` as the singular-case fallback.

    Args:
        turns: Reference turn array.
        upstream_tables: Tracked-state tables from upstream BPMs.
        downstream_tables: Tracked-state tables from downstream BPMs.
        value_col: Momentum column to solve for (``"px"`` or ``"py"``).
        variance_col: Corresponding variance column.
        kick_values: Per-turn fitted kick waveform, shape ``(n_turns,)``.
        smooth_lambda: Regularisation strength for the second-difference penalty.

    Returns:
        ``(pre_values, post_values, var_pre, var_post)`` each shape ``(n_turns,)``.

    Raises:
        ValueError: If no valid observations exist for the specified coordinate.
    """
    upstream_hat, upstream_var = _combine_many_estimates(
        *_align_estimate_component(
            turns, upstream_tables, value_col=value_col, variance_col=variance_col
        )
    )
    downstream_hat, downstream_var = _combine_many_estimates(
        *_align_estimate_component(
            turns, downstream_tables, value_col=value_col, variance_col=variance_col
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
        valid_downstream, downstream_weights * mapped_downstream_hat, 0.0
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

    # Fast path: system_matrix is pentadiagonal (bandwidth 2), so Cholesky
    # back-substitution is O(bandwidth × T) per RHS.
    try:
        ab = _sparse_to_upper_banded(system_matrix.tocsr(), bandwidth=2)
        inv_cols = solveh_banded(ab, np.eye(n_turns), lower=False, check_finite=False)
        var_pre = np.clip(np.diag(inv_cols), 0.0, None)
    except (np.linalg.LinAlgError, ValueError):
        covariance = np.linalg.pinv(system_matrix.toarray(), hermitian=True)
        var_pre = np.clip(np.diag(covariance), 0.0, None)

    t = np.zeros_like(pre_values)
    pt_col = upstream_tables[0].get("pt")
    pt = pt_col.to_numpy(dtype=float) if pt_col is not None else np.zeros_like(pre_values)

    # Kick is treated as known, so post variance equals pre variance.
    return pre_values, post_values, var_pre, var_pre.copy(), t, pt


# ---------------------------------------------------------------------------
# Main cleaning entry point
# ---------------------------------------------------------------------------


def _clean_ac_dipole_states(
    turns: np.ndarray,
    upstream_tables: list[pd.DataFrame],
    downstream_tables: list[pd.DataFrame],
    *,
    dpx_tune: float,
    dpy_tune: float,
    smooth_lambda: float = 1,
) -> ACDipoleCleaningResult:
    """Produce physically consistent pre-/post-kick state estimates in one pass.

    Pipeline:

    1. Pool ``x``/``y`` marker estimates from all BPM reconstructions into a
       single common same-turn position.
    2. Fit harmonic kick waveforms for ``px`` and ``py`` with local tune
       refinement.
    3. Solve smoothed pre-kick momentum with the fitted kick as a constraint.
    4. Return post-kick momentum as ``pre + fitted_kick``.

    Args:
        turns: Reference turn array.
        upstream_tables: Tracked-state DataFrames from upstream BPM reconstructions.
        downstream_tables: Tracked-state DataFrames from downstream BPM reconstructions.
        dpx_tune: Horizontal driven tune (fractional, 0 < tune < 0.5).
        dpy_tune: Vertical driven tune (fractional, 0 < tune < 0.5).
        smooth_lambda: Second-difference regularisation strength.

    Returns:
        An :class:`ACDipoleCleaningResult` with pre-kick (upstream) and
        post-kick (downstream) state estimates plus fitted kick waveforms.
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
    px_pre, px_post, var_px_pre, var_px_post, t_pre, pt_pre = (
        _solve_smoothed_pre_momentum_with_known_kick(
            turns,
            upstream_tables,
            downstream_tables,
            value_col="px",
            variance_col="var_px",
            kick_values=dpx_fit.fitted,
            smooth_lambda=smooth_lambda,
        )
    )
    py_pre, py_post, var_py_pre, var_py_post, t_post, pt_post = (
        _solve_smoothed_pre_momentum_with_known_kick(
            turns,
            upstream_tables,
            downstream_tables,
            value_col="py",
            variance_col="var_py",
            kick_values=dpy_fit.fitted,
            smooth_lambda=smooth_lambda,
        )
    )
    return ACDipoleCleaningResult(
        upstream=ACDipoleStateEstimate(
            state=ACDipoleStateSeries(x_common, px_pre, y_common, py_pre, t_pre, pt_pre),
            var_x=var_x_common,
            var_px=var_px_pre,
            var_y=var_y_common,
            var_py=var_py_pre,
        ),
        downstream=ACDipoleStateEstimate(
            state=ACDipoleStateSeries(x_common, px_post, y_common, py_post, t_post, pt_post),
            var_x=var_x_common,
            var_px=var_px_post,
            var_y=var_y_common,
            var_py=var_py_post,
        ),
        dpx_fit=dpx_fit,
        dpy_fit=dpy_fit,
    )
