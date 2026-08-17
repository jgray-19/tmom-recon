from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tmom_recon.data.schema import (
    NEXT,
    POSITION_COLS,
    PREV,
    SUFFIX_NEXT,
    SUFFIX_PREV,
    VARIANCE_COLS,
)
from tmom_recon.lattice.core import neighbour_plane_factors
from tmom_recon.physics.errors import (
    compute_measurement_errors,
    compute_optics_errors,
)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

LOGGER = logging.getLogger(__name__)


def _column_or_zeros(frame, column: str, template: np.ndarray) -> np.ndarray:
    if column in frame.columns:
        return frame[column].to_numpy()
    return np.zeros_like(template, dtype=float)


def _require_columns(frame, cols: set[str], context: str) -> None:
    missing = cols.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for {context}: {sorted(missing)}")


def _require_momentum_columns(frame: pd.DataFrame, names, suffix: str, context: str) -> None:
    required = (
        set(POSITION_COLS)
        | set(VARIANCE_COLS)
        | {
            names.x,
            names.y,
            names.var_x,
            names.var_y,
            f"sqrt_betax_{suffix}",
            f"sqrt_betay_{suffix}",
        }
    )
    _require_columns(frame, required, context)


def _has_uncertainty_columns(data: pd.DataFrame, neighbor_suffix: str, names) -> bool:
    """Check if DataFrame has optical uncertainty columns.

    Args:
        data: DataFrame to check.
        neighbor_suffix: Suffix for neighbor beta columns ('p' for prev, 'n' for next).

    Returns:
        True if uncertainty columns exist.
    """
    required_err_cols = {
        "sqrt_betax_err",
        "sqrt_betay_err",
        f"sqrt_betax_{neighbor_suffix}_err",
        f"sqrt_betay_{neighbor_suffix}_err",
        "alfax_err",
        "alfay_err",
        names.delta_x_err,
        names.delta_y_err,
    }
    # Dispersion error columns are optional - check if dx column exists first
    has_dispersion = "dx" in data.columns
    if has_dispersion:
        # If dispersion exists, require all dispersion error columns
        dispersion_err_cols = {
            "dx_err",
            f"dx_{neighbor_suffix}_err",
            "dpx_err",
            "dy_err",
            f"dy_{neighbor_suffix}_err",
            "dpy_err",
        }
        required_err_cols |= dispersion_err_cols

    return required_err_cols.issubset(data.columns)


def _compute_nominal_momenta(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    *,
    is_prev: bool,
    pt_est: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute nominal (error-free) momentum values.

    Uses

    .. math::

       \phi_x = 2 \pi \Delta_x,
       \qquad
       \phi_y = 2 \pi \Delta_y

    where :math:`\Delta` is the ``delta`` column of
    :mod:`tmom_recon.physics.bpm_phases`: the selected neighbour's advance
    *minus a quarter turn*, in turns. So :math:`\phi` here is
    :math:`\phi_{\mathrm{code}} = \phi_x - \pi/2`, **not** the phase advance and
    **not** the twiss ``mu1``/``mu2``. That origin is what makes ``sec``/``tan``
    the right functions below instead of ``csc``/``cot``.

    with normalized coordinates

    .. math::

       \tilde x = \frac{x - p_t D_x - p_t^2 D_x^{(2)}}{\sqrt{\beta_x}},
       \qquad
       \tilde x_n = \frac{x_n - p_t D_{x,n} - p_t^2 D_{x,n}^{(2)}}{\sqrt{\beta_{x,n}}},

    .. math::

       \tilde y = \frac{y - p_t D_y - p_t^2 D_y^{(2)}}{\sqrt{\beta_y}},
       \qquad
       \tilde y_n = \frac{y_n - p_t D_{y,n} - p_t^2 D_{y,n}^{(2)}}{\sqrt{\beta_{y,n}}},

    where :math:`p_t` is the MAD-NG longitudinal energy coordinate. MAD-NG
    dispersion columns are derivatives with respect to ``pt``, not ``dp/p``.

    The branch signs are :math:`(s, a)=(-1,+1)` for the previous neighbor and
    :math:`(+1,-1)` for the next neighbor. The reconstructed momenta are

    .. math::

       p_x =
       s \frac{\tilde x_n \sec \phi_x + \tilde x (\tan \phi_x + a \alpha_x)}
       {\sqrt{\beta_x}}
       + D_x' p_t + D_x^{(2)\prime} p_t^2,

    .. math::

       p_y =
       s \frac{\tilde y_n \sec \phi_y + \tilde y (\tan \phi_y + a \alpha_y)}
       {\sqrt{\beta_y}}
       + D_y' p_t + D_y^{(2)\prime} p_t^2.

    Second-order terms come from ``chrom=true`` Twiss columns and are zero when
    those columns are absent.

    Args:
        data: DataFrame with position and optics columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.
        pt_est: Momentum offset from the reference orbit (see
            :mod:`tmom_recon.reference`) -- never an absolute pt.

    Returns:
        Tuple of (px, py) arrays.
    """
    x_current = data["x"].to_numpy()
    y_current = data["y"].to_numpy()
    x_neighbor = data[names.x].to_numpy()
    y_neighbor = data[names.y].to_numpy()

    sqrt_beta_x = data["sqrt_betax"].to_numpy()
    sqrt_beta_y = data["sqrt_betay"].to_numpy()
    sqrt_beta_x_neigh = data[f"sqrt_betax_{neighbor_suffix}"].to_numpy()
    sqrt_beta_y_neigh = data[f"sqrt_betay_{neighbor_suffix}"].to_numpy()

    alpha_x = data["alfax"].to_numpy()
    alpha_y = data["alfay"].to_numpy()

    dx_current = _column_or_zeros(data, "dx", x_current)
    dx_neighbor = _column_or_zeros(data, names.dx, x_neighbor)
    dpx_current = _column_or_zeros(data, "dpx", x_current)
    dy_current = _column_or_zeros(data, "dy", y_current)
    dy_neighbor = _column_or_zeros(data, names.dy, y_neighbor)
    dpy_current = _column_or_zeros(data, "dpy", y_current)

    # Second-order dispersion, zero when the twiss carries no `chrom` columns.
    ddx_current = _column_or_zeros(data, "ddx", x_current)
    ddx_neighbor = _column_or_zeros(data, names.ddx, x_neighbor)
    ddpx_current = _column_or_zeros(data, "ddpx", x_current)
    ddy_current = _column_or_zeros(data, "ddy", y_current)
    ddy_neighbor = _column_or_zeros(data, names.ddy, y_neighbor)
    ddpy_current = _column_or_zeros(data, "ddpy", y_current)

    phi_x = data[names.delta_x].to_numpy() * 2 * np.pi
    phi_y = data[names.delta_y].to_numpy() * 2 * np.pi

    sign_x, alpha_sign_x, cos_phi_x, tan_phi_x, sec_phi_x = neighbour_plane_factors(
        phi_x, is_prev=is_prev
    )
    sign_y, alpha_sign_y, cos_phi_y, tan_phi_y, sec_phi_y = neighbour_plane_factors(
        phi_y, is_prev=is_prev
    )

    # Vertical dispersion should be very small, or typically 0, but included for completeness
    pt2 = pt_est * pt_est
    x_current_norm = (x_current - pt_est * dx_current - pt2 * ddx_current) / sqrt_beta_x
    x_neighbor_norm = (x_neighbor - pt_est * dx_neighbor - pt2 * ddx_neighbor) / sqrt_beta_x_neigh
    y_current_norm = (y_current - pt_est * dy_current - pt2 * ddy_current) / sqrt_beta_y
    y_neighbor_norm = (y_neighbor - pt_est * dy_neighbor - pt2 * ddy_neighbor) / sqrt_beta_y_neigh

    # Nominal momenta
    px = (
        sign_x
        * (x_neighbor_norm * sec_phi_x + x_current_norm * (tan_phi_x + alpha_sign_x * alpha_x))
        / sqrt_beta_x
        + dpx_current * pt_est
        + ddpx_current * pt2
    )
    py = (
        sign_y
        * (y_neighbor_norm * sec_phi_y + y_current_norm * (tan_phi_y + alpha_sign_y * alpha_y))
        / sqrt_beta_y
        + dpy_current * pt_est
        + ddpy_current * pt2
    )

    return px, py


def _compute_momenta(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    *,
    is_prev: bool,
    pt_est: float = 0.0,
    include_optics_errors: bool = False,
) -> pd.DataFrame:
    r"""Compute momenta with error propagation.

    The total variances are built as

    .. math::

       \operatorname{var}(p_x) =
       \operatorname{var}_{\mathrm{meas}}(p_x) +
       \operatorname{var}_{\mathrm{opt}}(p_x),

    .. math::

       \operatorname{var}(p_y) =
       \operatorname{var}_{\mathrm{meas}}(p_y) +
       \operatorname{var}_{\mathrm{opt}}(p_y),

    where the optics term is only added when the required uncertainty columns
    are present and ``include_optics_errors=True``.

    Args:
        data: DataFrame with position and optics columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.
        pt_est: Estimated MAD-NG pt.
        include_optics_errors: Whether to include optics uncertainties.

    Returns:
        DataFrame with px, py, var_px, var_py columns added.
    """
    _require_momentum_columns(data, names, neighbor_suffix, "momenta")

    has_optics_uncertainties = _has_uncertainty_columns(data, neighbor_suffix, names)
    use_optics_errors = include_optics_errors and has_optics_uncertainties
    if use_optics_errors:
        LOGGER.debug("Including optical function uncertainties for %s momenta", neighbor_suffix)

    # Compute nominal momenta
    px, py = _compute_nominal_momenta(data, names, neighbor_suffix, is_prev=is_prev, pt_est=pt_est)

    # Compute measurement errors (always included)
    var_px, var_py = compute_measurement_errors(data, names, neighbor_suffix, is_prev)

    # Add optics errors if requested and available
    if use_optics_errors:
        var_px_opt_errors, var_py_opt_errors = compute_optics_errors(
            data, names, neighbor_suffix, is_prev, pt_est
        )
        var_px = var_px + np.sum(var_px_opt_errors, axis=0)
        var_py = var_py + np.sum(var_py_opt_errors, axis=0)

    data["px"] = px
    data["py"] = py
    data["var_px"] = var_px
    data["var_py"] = var_py

    return data


def momenta_from_prev(
    data_p: pd.DataFrame, pt_est: float = 0.0, *, include_optics_errors: bool = False
) -> pd.DataFrame:
    return _compute_momenta(
        data_p,
        PREV,
        SUFFIX_PREV,
        is_prev=True,
        pt_est=pt_est,
        include_optics_errors=include_optics_errors,
    )


def momenta_from_next(
    data_n: pd.DataFrame, pt_est: float = 0.0, *, include_optics_errors: bool = False
) -> pd.DataFrame:
    return _compute_momenta(
        data_n,
        NEXT,
        SUFFIX_NEXT,
        is_prev=False,
        pt_est=pt_est,
        include_optics_errors=include_optics_errors,
    )
