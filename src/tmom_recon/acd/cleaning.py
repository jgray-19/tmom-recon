"""AC-dipole state cleaning utilities.

Combines noisy upstream/downstream BPM state estimates into physically
consistent pre-/post-kick states at the AC-dipole marker.

Two building blocks:

1. A weighted harmonic fit for the kick waveform parameters, with local
   tune refinement via bounded scalar optimisation.
2. A per-turn inverse-variance mean of the pre-kick momentum with the
   kick waveform treated as known.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

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


# ---------------------------------------------------------------------------
# Inverse-variance combination
# ---------------------------------------------------------------------------


def _state_estimate(turns: np.ndarray, table: pd.DataFrame) -> ACDipoleStateEstimate:
    """Align a tracked-state table onto *turns* as an :class:`ACDipoleStateEstimate`.

    Args:
        turns: Reference turn array used for alignment.
        table: DataFrame with columns ``x, px, y, py, var_x, var_px, var_y, var_py, turn``.

    Returns:
        The state and variances at each turn.
    """
    x, var_x = _align_table_columns(turns, table, value_col="x", variance_col="var_x")
    px, var_px = _align_table_columns(turns, table, value_col="px", variance_col="var_px")
    y, var_y = _align_table_columns(turns, table, value_col="y", variance_col="var_y")
    py, var_py = _align_table_columns(turns, table, value_col="py", variance_col="var_py")
    t = np.zeros_like(x)
    pt, _ = _align_table_columns(turns, table, value_col="pt", variance_col="var_x")
    return ACDipoleStateEstimate(
        state=ACDipoleStateSeries(x, px, y, py, t, pt),
        var_x=var_x,
        var_px=var_px,
        var_y=var_y,
        var_py=var_py,
    )


def _combine_marker_transverse_positions(
    turns: np.ndarray,
    upstream_table: pd.DataFrame,
    downstream_table: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Combine marker transverse positions from both upstream and downstream estimates.

    The AC dipole is modelled as an instantaneous kick: it changes ``px``/``py``
    but not ``x``/``y``. Upstream and downstream transported states for the same
    turn therefore share the same marker position modulo noise, so all available
    ``x`` and ``y`` estimates are pooled into a single common trajectory.

    Args:
        turns: Reference turn array.
        upstream_table: Tracked-state table from the upstream BPM.
        downstream_table: Tracked-state table from the downstream BPM.

    Returns:
        ``(x_common, y_common, var_x_common, var_y_common)`` each shape
        ``(n_turns,)``.
    """
    pooled = []
    for col in ("x", "y"):
        value_up, var_up = _align_table_columns(
            turns, upstream_table, value_col=col, variance_col=f"var_{col}"
        )
        value_down, var_down = _align_table_columns(
            turns, downstream_table, value_col=col, variance_col=f"var_{col}"
        )
        weight_up, weight_down = 1.0 / var_up, 1.0 / var_down
        total = weight_up + weight_down
        pooled.append(((weight_up * value_up + weight_down * value_down) / total, 1.0 / total))
    (x_common, var_x_common), (y_common, var_y_common) = pooled
    return x_common, y_common, var_x_common, var_y_common


# ---------------------------------------------------------------------------
# Harmonic fitting
# ---------------------------------------------------------------------------


def _build_harmonic_design(turns: np.ndarray, tune: float) -> np.ndarray:
    """Build the two-column harmonic design matrix ``[sin, cos]``."""
    omega_turn = 2.0 * np.pi * float(tune)
    return np.column_stack([np.sin(omega_turn * turns), np.cos(omega_turn * turns)])


def _build_raw_kick_series(
    turns: np.ndarray,
    upstream_table: pd.DataFrame,
    downstream_table: pd.DataFrame,
    *,
    value_col: str,
    variance_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the raw same-turn kick series and its propagated variance."""
    upstream_hat, upstream_var = _align_table_columns(
        turns, upstream_table, value_col=value_col, variance_col=variance_col
    )
    downstream_hat, downstream_var = _align_table_columns(
        turns, downstream_table, value_col=value_col, variance_col=variance_col
    )
    return downstream_hat - upstream_hat, upstream_var + downstream_var


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

    weights = 1.0 / np.sqrt(variances)
    weighted_rhs = values * weights
    weighted_design = design * weights[:, None]

    try:
        solution = np.linalg.solve(
            weighted_design.T @ weighted_design, weighted_design.T @ weighted_rhs
        )
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(weighted_design, weighted_rhs, rcond=None)

    residual = weighted_design @ solution - weighted_rhs
    weighted_rss = float(np.dot(residual, residual))

    sin_coeff, cos_coeff = solution
    fitted = design @ solution

    return (
        ACDipoleHarmonicFit(
            tune=float(tune),
            amplitude=float(np.hypot(sin_coeff, cos_coeff)),
            phase=float(np.arctan2(cos_coeff, sin_coeff)),
            fitted=fitted,
        ),
        weighted_rss,
    )


def _harmonic_grid_rss(
    t: np.ndarray, y: np.ndarray, w: np.ndarray, tune0: float, n_half: int
) -> np.ndarray:
    """Weighted RSS of the fixed-tune fit at ``tune0 + k/len(t)``, ``|k| <= n_half``.

    The normal equations need only ``sum w y exp(2 pi i Q t)`` and
    ``sum w exp(4 pi i Q t)``; for integer turns ``t`` both are one FFT over
    ``t mod len(t)`` on this grid.
    """
    n = len(t)
    residue = np.mod(t.astype(np.int64), n)
    rotation = np.exp(2j * np.pi * tune0 * t)
    z1 = n * np.fft.ifft(
        np.bincount(residue, weights=(w * y * rotation).real, minlength=n)
        + 1j * np.bincount(residue, weights=(w * y * rotation).imag, minlength=n)
    )
    rotation2 = rotation**2
    z2 = n * np.fft.ifft(
        np.bincount(residue, weights=(w * rotation2).real, minlength=n)
        + 1j * np.bincount(residue, weights=(w * rotation2).imag, minlength=n)
    )
    k = np.arange(-n_half, n_half + 1)
    s1, c1 = z1[k % n].imag, z1[k % n].real
    s2, c2 = z2[(2 * k) % n].imag, z2[(2 * k) % n].real
    total = w.sum()
    ss, cc, sc = 0.5 * (total - c2), 0.5 * (total + c2), 0.5 * s2
    det = ss * cc - sc**2
    a = (cc * s1 - sc * c1) / det
    b = (ss * c1 - sc * s1) / det
    return np.dot(w * y, y) - 2.0 * (a * s1 + b * c1) + a * a * ss + 2.0 * a * b * sc + b * b * cc


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

        k(t) = a sin(2πQt) + b cos(2πQt)

    The fitted parameters are ``a``, ``b`` and ``Q``; there is no constant offset.
    The supplied tune enters as a Gaussian prior, not as a hard search window::

        χ² = Σ ((yᵢ - kᵢ) / σᵢ)² + ((Q - Q₀) / σ_Q)²
    """
    t_valid = turns.astype(float)
    y_valid = values.astype(float)
    inv_sigma = 1.0 / np.sqrt(variances)

    tune0 = float(np.clip(tune_hint, _MIN_TUNE, _MAX_TUNE))

    _, initial_score = _solve_harmonic_series_fit(
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
    _tune_grid = tune0 + _grid_step * np.arange(-_n_half, _n_half + 1, dtype=float)
    grid_rss = _harmonic_grid_rss(t_valid, y_valid, inv_sigma**2, tune0, _n_half)
    grid_rss[(_tune_grid < _MIN_TUNE) | (_tune_grid > _MAX_TUNE)] = np.inf
    _best_tune = float(_tune_grid[int(np.argmin(grid_rss))])
    _best_fit, _ = _solve_harmonic_series_fit(
        turns, tune=_best_tune, values=values, variances=variances
    )

    sin0 = _best_fit.amplitude * np.cos(_best_fit.phase)
    cos0 = _best_fit.amplitude * np.sin(_best_fit.phase)
    p0 = np.array([sin0, cos0, _best_tune], dtype=float)

    amp_scale = max(float(_best_fit.amplitude), 1.0)

    # The Gaussian prior on tune is intentionally omitted here: the grid search
    # above already starts the optimizer near the data-driven minimum, so no
    # pull toward the hint is needed.  Including the prior blocks convergence
    # whenever the true tune lies more than ~1σ from the hint, because for small
    # signal amplitudes the prior penalty dominates the data-fit residuals.
    tune_scale = 1.0 / max(len(t_valid), 1)  # Rayleigh resolution

    def residual(params: np.ndarray) -> np.ndarray:
        sin_coeff, cos_coeff, tune = params
        omega_turn = 2.0 * np.pi * tune
        model = sin_coeff * np.sin(omega_turn * t_valid) + cos_coeff * np.cos(omega_turn * t_valid)
        return (model - y_valid) * inv_sigma

    result = least_squares(
        residual,
        p0,
        bounds=(
            [-np.inf, -np.inf, _MIN_TUNE],
            [np.inf, np.inf, _MAX_TUNE],
        ),
        x_scale=[amp_scale, amp_scale, tune_scale],
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

    sin_coeff, cos_coeff, tune = result.x
    design = _build_harmonic_design(turns, float(tune))
    fitted = design @ np.array([sin_coeff, cos_coeff], dtype=float)

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
        fitted=fitted,
    )


def _refine_known_kick_fit(
    turns: np.ndarray,
    upstream_table: pd.DataFrame,
    downstream_table: pd.DataFrame,
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
        upstream_table,
        downstream_table,
        value_col=value_col,
        variance_col=variance_col,
    )

    n_turns = len(turns)

    if n_turns < _FIT_TUNE_MIN_TURNS:
        fit, score = _solve_harmonic_series_fit(
            turns,
            tune=tune_hint,
            values=raw_kick,
            variances=raw_var,
        )
        LOGGER.info(
            "Fitted %s harmonic with fixed tune %.8f over %d turns; weighted RSS %.6e",
            value_col,
            fit.tune,
            n_turns,
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
# Pre-kick momentum combination
# ---------------------------------------------------------------------------


def _combine_pre_momentum_with_known_kick(
    turns: np.ndarray,
    upstream_table: pd.DataFrame,
    downstream_table: pd.DataFrame,
    *,
    value_col: str,
    variance_col: str,
    kick_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inverse-variance mean of the pre-kick momentum with the kick waveform fixed.

    With upstream estimate ``u_t``, downstream estimate ``d_t`` and fitted kick
    ``k_t``, the pre-kick momentum is ``p_t = (w_u u_t + w_d (d_t - k_t)) / (w_u + w_d)``
    with ``w`` the inverse variances, and ``var(p_t) = 1 / (w_u + w_d)``. Post-kick
    values are ``p_t + k_t``. The kick is taken as known, so its uncertainty is not
    propagated.

    Args:
        turns: Reference turn array.
        upstream_table: Tracked-state table from the upstream BPM.
        downstream_table: Tracked-state table from the downstream BPM.
        value_col: Momentum column to solve for (``"px"`` or ``"py"``).
        variance_col: Corresponding variance column.
        kick_values: Per-turn fitted kick waveform, shape ``(n_turns,)``.

    Returns:
        ``(pre_values, post_values, var_pre, var_post, t, pt)``, each shape ``(n_turns,)``.
    """
    upstream_hat, upstream_var = _align_table_columns(
        turns, upstream_table, value_col=value_col, variance_col=variance_col
    )
    downstream_hat, downstream_var = _align_table_columns(
        turns, downstream_table, value_col=value_col, variance_col=variance_col
    )

    upstream_weights = 1.0 / upstream_var
    downstream_weights = 1.0 / downstream_var
    total_weights = upstream_weights + downstream_weights
    pre_values = (
        upstream_weights * upstream_hat + downstream_weights * (downstream_hat - kick_values)
    ) / total_weights
    post_values = pre_values + kick_values
    var_pre = 1.0 / total_weights

    t = np.zeros_like(pre_values)
    pt_col = upstream_table.get("pt")
    pt = pt_col.to_numpy(dtype=float) if pt_col is not None else np.zeros_like(pre_values)

    return pre_values, post_values, var_pre, var_pre.copy(), t, pt


def _remove_closed_orbit(
    turns: np.ndarray,
    table: pd.DataFrame,
    *,
    value_col: str,
    variance_col: str,
) -> tuple[pd.DataFrame, float]:
    """Subtract one side's closed orbit from *value_col* in *table*.

    The closed orbit is the inverse-variance mean of the side's series.
    Returns the deviation table and the removed closed-orbit value.
    """
    values, variances = _align_table_columns(
        turns, table, value_col=value_col, variance_col=variance_col
    )
    orbit = float(np.average(values, weights=1.0 / variances))
    return table.assign(**{value_col: table[value_col] - orbit}), orbit


# ---------------------------------------------------------------------------
# Main cleaning entry point
# ---------------------------------------------------------------------------


def _clean_ac_dipole_states(
    turns: np.ndarray,
    upstream_table: pd.DataFrame,
    downstream_table: pd.DataFrame,
    *,
    dpx_tune: float,
    dpy_tune: float,
) -> ACDipoleCleaningResult:
    """Produce physically consistent pre-/post-kick state estimates in one pass.

    Pipeline:

    1. Pool ``x``/``y`` marker estimates from both BPM reconstructions into a
       single common same-turn position.
    2. Remove each side's closed orbit from ``px``/``py``; the AC dipole only
       oscillates about it.
    3. Fit zero-mean harmonic kick waveforms to the deviations with local tune
       refinement.
    4. Combine pre-kick momentum estimates with the fitted kick as a constraint.
    5. Return pre-/post-kick momentum as ``pre + orbit_up`` and
       ``pre + fitted_kick + orbit_down``.

    Args:
        turns: Reference turn array.
        upstream_table: Tracked-state DataFrame from the upstream BPM reconstruction.
        downstream_table: Tracked-state DataFrame from the downstream BPM reconstruction.
        dpx_tune: Horizontal driven tune (fractional, 0 < tune < 0.5).
        dpy_tune: Vertical driven tune (fractional, 0 < tune < 0.5).

    Returns:
        An :class:`ACDipoleCleaningResult` with pre-kick (upstream) and
        post-kick (downstream) state estimates plus fitted kick waveforms.
    """
    x_common, y_common, var_x_common, var_y_common = _combine_marker_transverse_positions(
        turns,
        upstream_table,
        downstream_table,
    )
    fits = {}
    solved = {}
    steps = {}
    for plane, tune in (("px", dpx_tune), ("py", dpy_tune)):
        variance_col = f"var_{plane}"
        # The AC dipole oscillates about the closed orbit, so its kick is
        # zero-mean.  Any constant step between the transported upstream and
        # downstream states is an orbit-angle difference, not a kick: fit and
        # combine the deviations, then restore each side's closed orbit.
        up_dev, orbit_up = _remove_closed_orbit(
            turns, upstream_table, value_col=plane, variance_col=variance_col
        )
        down_dev, orbit_down = _remove_closed_orbit(
            turns, downstream_table, value_col=plane, variance_col=variance_col
        )
        fits[plane] = _refine_known_kick_fit(
            turns,
            up_dev,
            down_dev,
            value_col=plane,
            variance_col=variance_col,
            tune_hint=tune,
        )
        pre, post, var_pre, var_post, t, pt = _combine_pre_momentum_with_known_kick(
            turns,
            up_dev,
            down_dev,
            value_col=plane,
            variance_col=variance_col,
            kick_values=fits[plane].fitted,
        )
        solved[plane] = (pre + orbit_up, post + orbit_down, var_pre, var_post, t, pt)
        steps[plane] = orbit_down - orbit_up
    dpx_fit, dpy_fit = fits["px"], fits["py"]
    px_pre, px_post, var_px_pre, var_px_post, t_pre, pt_pre = solved["px"]
    py_pre, py_post, var_py_pre, var_py_post, t_post, pt_post = solved["py"]
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
        dpx_orbit_step=steps["px"],
        dpy_orbit_step=steps["py"],
    )
