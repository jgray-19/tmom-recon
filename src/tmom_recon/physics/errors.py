"""
Error propagation utilities for momentum reconstruction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tmom_recon.lattice.core import neighbour_plane_factors

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

PLANES = ("x", "y")


def column_or_zeros(frame: pd.DataFrame, column: str, template: np.ndarray) -> np.ndarray:
    """``frame[column]`` as an array, or zeros shaped like *template* when absent."""
    if column in frame.columns:
        return frame[column].to_numpy(dtype=float)
    return np.zeros_like(template, dtype=float)


def _plane_factors(data: pd.DataFrame, names, plane: str, is_prev: bool):
    """``(sign, alpha_sign, tan_phi, sec_phi)`` for *plane*; ``phi`` is stored in turns."""
    phi = data[getattr(names, f"delta_{plane}")].to_numpy() * 2 * np.pi
    sign, alpha_sign, _cos_phi, tan_phi, sec_phi = neighbour_plane_factors(phi, is_prev=is_prev)
    return sign, alpha_sign, tan_phi, sec_phi


def _plane_measurement_variance(
    data: pd.DataFrame, names, neighbor_suffix: str, plane: str, is_prev: bool
) -> np.ndarray:
    var_current = data[f"var_{plane}"].to_numpy()
    var_neighbor = data[getattr(names, f"var_{plane}")].to_numpy()
    sqrt_beta = data[f"sqrt_beta{plane}"].to_numpy()
    sqrt_beta_neigh = data[f"sqrt_beta{plane}_{neighbor_suffix}"].to_numpy()
    alpha = data[f"alfa{plane}"].to_numpy()
    sign, alpha_sign, tan_phi, sec_phi = _plane_factors(data, names, plane, is_prev)

    return (
        var_neighbor * (sign * sec_phi / (sqrt_beta * sqrt_beta_neigh)) ** 2
        + var_current * (sign * (tan_phi + alpha_sign * alpha) / sqrt_beta**2) ** 2
    )


def compute_measurement_errors(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    is_prev: bool,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute measurement-only error contributions to momentum variances.

    For each plane :math:`u \in \{x, y\}`,

    .. math::

       \operatorname{var}_{\mathrm{meas}}(p_u)
       =
       \sigma^2_{u_n}
       \left(
       \frac{s \sec \phi_u}{\sqrt{\beta_u}\sqrt{\beta_{u,n}}}
       \right)^2
       +
       \sigma^2_u
       \left(
       \frac{s (\tan \phi_u + a \alpha_u)}{\beta_u}
       \right)^2,

    with :math:`s = -1, a = +1` for the previous neighbor and
    :math:`s = +1, a = -1` for the next neighbor.

    Args:
        data: DataFrame with position and variance columns.
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.

    Returns:
        Tuple of (var_px_measurement, var_py_measurement).
    """
    var_px, var_py = (
        _plane_measurement_variance(data, names, neighbor_suffix, plane, is_prev)
        for plane in PLANES
    )
    return var_px, var_py


def _plane_optics_variance(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    plane: str,
    is_prev: bool,
    pt_est: float,
) -> np.ndarray:
    current = data[plane].to_numpy()
    neighbor = data[getattr(names, plane)].to_numpy()

    sqrt_beta = data[f"sqrt_beta{plane}"].to_numpy()
    sqrt_beta_neigh = data[f"sqrt_beta{plane}_{neighbor_suffix}"].to_numpy()
    alpha = data[f"alfa{plane}"].to_numpy()

    # Dispersion about pt = 0, zero when absent -- as in momenta._compute_nominal_momenta.
    dispersion_column = f"d{plane}"
    neighbor_dispersion_column = getattr(names, f"d{plane}")
    if pt_est != 0.0 and dispersion_column not in data.columns:
        raise ValueError(f"Column {dispersion_column!r} missing but pt_est is non-zero.")
    d_current = column_or_zeros(data, dispersion_column, current)
    d_neighbor = column_or_zeros(data, neighbor_dispersion_column, neighbor)
    dd_current = column_or_zeros(data, f"dd{plane}", current)
    dd_neighbor = column_or_zeros(data, getattr(names, f"dd{plane}"), neighbor)

    # One-sigma optics errors; the phase error is stored in TURNS.
    sigma_sqrt_beta = data[f"sqrt_beta{plane}_err"].to_numpy()
    sigma_sqrt_beta_neigh = data[f"sqrt_beta{plane}_{neighbor_suffix}_err"].to_numpy()
    sigma_alpha = data[f"alfa{plane}_err"].to_numpy()
    sigma_d_current = column_or_zeros(data, f"{dispersion_column}_err", current)
    sigma_dp_current = column_or_zeros(data, f"dp{plane}_err", current)
    sigma_d_neighbor = column_or_zeros(data, f"{neighbor_dispersion_column}_err", neighbor)
    sigma_delta = data[getattr(names, f"delta_{plane}_err")].to_numpy()

    sign, alpha_sign, tan_phi, sec_phi = _plane_factors(data, names, plane, is_prev)

    pt2 = pt_est * pt_est
    current_norm = (current - pt_est * d_current - pt2 * dd_current) / sqrt_beta
    neighbor_norm = (neighbor - pt_est * d_neighbor - pt2 * dd_neighbor) / sqrt_beta_neigh

    a = tan_phi + alpha_sign * alpha
    dp_dd_neigh = sign * (-pt_est) * sec_phi / (sqrt_beta * sqrt_beta_neigh)
    dp_dd_curr = sign * (-pt_est) * a / sqrt_beta**2
    dp_ddp = pt_est
    dp_dalpha = sign * alpha_sign * current_norm / sqrt_beta
    dp_ds = -(sign / sqrt_beta**2) * (neighbor_norm * sec_phi + 2.0 * current_norm * a)
    dp_ds_neigh = -sign * neighbor_norm * sec_phi / (sqrt_beta * sqrt_beta_neigh)
    dp_dphi = (sign / sqrt_beta) * (neighbor_norm * sec_phi * tan_phi + current_norm * sec_phi**2)
    # Chain rule to the phase in turns: phi = 2*pi*delta.
    dp_ddelta = dp_dphi * 2.0 * np.pi

    return np.vstack(
        (
            sigma_d_neighbor**2 * dp_dd_neigh**2,
            sigma_d_current**2 * dp_dd_curr**2,
            sigma_dp_current**2 * dp_ddp**2,
            sigma_alpha**2 * dp_dalpha**2,
            sigma_sqrt_beta**2 * dp_ds**2,
            sigma_sqrt_beta_neigh**2 * dp_ds_neigh**2,
            sigma_delta**2 * dp_ddelta**2,
        )
    )


def compute_optics_errors(
    data: pd.DataFrame,
    names,
    neighbor_suffix: str,
    is_prev: bool,
    pt_est: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute optics error contributions to momentum variances.

    The code applies first-order uncertainty propagation, for each plane
    :math:`u \in \{x, y\}`,

    .. math::

       \operatorname{var}_{\mathrm{opt}}(p_u) =
       \sum_i \sigma_i^2 \left(\frac{\partial p_u}{\partial q_i}\right)^2,
       \qquad
       q_i \in
       \{D_{u,n}, D_u, D_u', \alpha_u, \sqrt{\beta_u}, \sqrt{\beta_{u,n}}, \Delta_u\}.

    The derivatives are taken of the momenta of
    :func:`tmom_recon.physics.momenta._compute_nominal_momenta`, including its
    second-order dispersion (``ddx``/``ddy``, zero when absent), which is
    treated as exact. The phase-advance uncertainties are stored in turns and
    converted to radians through :math:`\phi = 2 \pi \Delta`, so the
    implementation uses :math:`\partial p / \partial \Delta = 2 \pi \, \partial p / \partial \phi`.

    Args:
        data: DataFrame with optics and error columns. Dispersion and its
            errors are optional (zero when absent).
        names: Neighbor column names.
        neighbor_suffix: Suffix for neighbor columns ('p' or 'n').
        is_prev: Whether this is previous neighbor calculation.
        pt_est: Estimated MAD-NG pt.

    Returns:
        Tuple of (var_px_errors, var_py_errors), each of shape (7, N): one row
        per contribution, in the order of ``q_i`` above. Sum along axis 0 and
        add to the measurement variance for the total variance.

    Raises:
        ValueError: If *pt_est* is non-zero and a plane's dispersion column
            (``dx``/``dy``) is missing.
    """
    var_px, var_py = (
        _plane_optics_variance(data, names, neighbor_suffix, plane, is_prev, pt_est)
        for plane in PLANES
    )
    return var_px, var_py
