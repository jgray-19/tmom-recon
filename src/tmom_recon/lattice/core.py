from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tmom_recon.data.config import FILE_COLUMNS, POSITION_STD_DEV
from tmom_recon.data.schema import (
    CORE_ID_COLS,
    CORE_MOM_COLS,
)

LOGGER = logging.getLogger(__name__)

#: Second-order dispersion columns, optional throughout: everything degrades to
#: the first-order treatment when they are missing.
SECOND_ORDER_DISPERSION_COLUMNS = ("ddx", "ddpx", "ddy", "ddpy")
OUT_COLS = list(FILE_COLUMNS)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Mapping

    import pandas as pd


@dataclass(frozen=True)
class LatticeMaps:
    """Optics parameters mapped by BPM name."""

    sqrt_betax: Mapping[str, float]
    sqrt_betay: Mapping[str, float]
    betax: Mapping[str, float]
    betay: Mapping[str, float]
    alfax: Mapping[str, float]
    alfay: Mapping[str, float]
    dx: Mapping[str, float] | None = None
    dpx: Mapping[str, float] | None = None
    dy: Mapping[str, float] | None = None
    dpy: Mapping[str, float] | None = None
    # Second-order dispersion, per unit pt**2 (MAD-NG `chrom=true` folds the
    # Taylor 1/2 in, so the orbit is pt*dx + pt**2*ddx). Absent when the twiss
    # was not run with chrom, or came from a measurement.
    ddx: Mapping[str, float] | None = None
    ddpx: Mapping[str, float] | None = None
    ddy: Mapping[str, float] | None = None
    ddpy: Mapping[str, float] | None = None


def get_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng or np.random.default_rng()


def neighbour_plane_factors(
    phi: np.ndarray, *, is_prev: bool
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Compute trigonometric factors for neighbor plane calculations.

    Args:
        phi: Phase differences array.
        is_prev: Whether this is previous neighbor calculation.

    Returns:
        Tuple of (sign, alpha_sign, cos_phi, tan_phi, sec_phi).
    """
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)
    sec_phi = 1.0 / cos_phi
    sign = -1 if is_prev else 1
    alpha_sign = 1 if is_prev else -1
    return sign, alpha_sign, cos_phi, tan_phi, sec_phi


def combine_two_estimates(
    value_a: np.ndarray,
    var_a: np.ndarray,
    value_b: np.ndarray,
    var_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine two estimates using inverse-variance weighting.

    Handles NaN values, non-positive variances, and infinite variances.

    Args:
        value_a: Values from estimate A.
        var_a: Variances from estimate A.
        value_b: Values from estimate B.
        var_b: Variances from estimate B.

    Returns:
        Tuple of (combined_value, combined_var).
    """
    # Mask for valid estimates
    valid_a = np.isfinite(var_a) & (var_a > 0.0) & np.isfinite(value_a)
    valid_b = np.isfinite(var_b) & (var_b > 0.0) & np.isfinite(value_b)

    # Inverse variances (0 for invalid)
    inv_var_a = np.where(valid_a, 1.0 / var_a, 0.0)
    inv_var_b = np.where(valid_b, 1.0 / var_b, 0.0)

    # Combined inverse variance
    inv_var_combined = inv_var_a + inv_var_b

    # Combined value
    combined_value = np.where(
        inv_var_combined > 0.0,
        (inv_var_a * value_a + inv_var_b * value_b) / inv_var_combined,
        np.nan,  # or some fallback
    )

    # Fallbacks when only one is valid
    combined_value = np.where(valid_a & ~valid_b, value_a, combined_value)
    combined_value = np.where(valid_b & ~valid_a, value_b, combined_value)

    # Combined variance
    combined_var = np.where(inv_var_combined > 0.0, 1.0 / inv_var_combined, np.inf)
    combined_var = np.where(valid_a & ~valid_b, var_a, combined_var)
    combined_var = np.where(valid_b & ~valid_a, var_b, combined_var)

    return combined_value, combined_var


def inject_noise_xy(
    df: pd.DataFrame,
    rng: np.random.Generator,
    noise_std: float = POSITION_STD_DEV,
) -> pd.DataFrame:
    """Add Gaussian position noise and declare it in the variance columns.

    The variances are overwritten rather than accumulated: whatever a tracking
    pipeline put there was a placeholder for noiseless data, and leaving it in
    place makes the reconstruction weight and report an uncertainty unrelated to
    the noise actually present.
    """
    out = df.copy(deep=True)
    n_rows = len(df)
    LOGGER.debug("Adding Gaussian noise: std=%g", noise_std)
    noise_x = rng.normal(0.0, noise_std, size=n_rows)
    noise_y = rng.normal(0.0, noise_std, size=n_rows)

    out["x"] = df["x"] + noise_x
    out["y"] = df["y"] + noise_y
    out["var_x"] = noise_std**2
    out["var_y"] = noise_std**2
    return out


def build_lattice_maps(tws: pd.DataFrame) -> LatticeMaps:
    sqrt_betax = np.sqrt(tws["betx"])
    sqrt_betay = np.sqrt(tws["bety"])
    params: dict[str, Mapping[str, float]] = {
        "sqrt_betax": sqrt_betax.to_dict(),
        "sqrt_betay": sqrt_betay.to_dict(),
        "betax": tws["betx"].to_dict(),
        "betay": tws["bety"].to_dict(),
        "alfax": tws["alfx"].to_dict(),
        "alfay": tws["alfy"].to_dict(),
    }
    params["dx"] = tws["dx"].to_dict()
    params["dpx"] = tws["dpx"].to_dict()
    params["dy"] = tws["dy"].to_dict()
    params["dpy"] = tws["dpy"].to_dict()
    for col in SECOND_ORDER_DISPERSION_COLUMNS:
        if col in tws.columns:
            params[col] = tws[col].to_dict()
    return LatticeMaps(**params)


def attach_lattice_columns(df: pd.DataFrame, maps: LatticeMaps) -> pd.DataFrame:
    out = df.copy(deep=True)
    out["sqrt_betax"] = out["name"].map(maps.sqrt_betax)
    out["sqrt_betay"] = out["name"].map(maps.sqrt_betay)
    out["betax"] = out["name"].map(maps.betax)
    out["betay"] = out["name"].map(maps.betay)
    out["alfax"] = out["name"].map(maps.alfax)
    out["alfay"] = out["name"].map(maps.alfay)

    if maps.dx is not None:
        out["dx"] = out["name"].map(maps.dx)
    if maps.dpx is not None:
        out["dpx"] = out["name"].map(maps.dpx)
    if maps.dy is not None:
        out["dy"] = out["name"].map(maps.dy)
    if maps.dpy is not None:
        out["dpy"] = out["name"].map(maps.dpy)
    for col in SECOND_ORDER_DISPERSION_COLUMNS:
        mapping = getattr(maps, col)
        if mapping is not None:
            out[col] = out["name"].map(mapping)
    return out


def weights(psi: np.ndarray, inv_beta1: np.ndarray, inv_beta2: np.ndarray) -> np.ndarray:
    pref = 1.0 / (np.sqrt(2.0) * np.abs(np.sin(psi)))
    inside = (
        inv_beta1
        + inv_beta2
        + np.sqrt(inv_beta1**2 + inv_beta2**2 + 2.0 * inv_beta1 * inv_beta2 * np.cos(2.0 * psi))
    )
    f = pref * np.sqrt(inside)
    return 1.0 / f


def align_by_name_turn(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames by sorting on name and turn."""
    a_aligned = a.sort_values(list(CORE_ID_COLS)).reset_index(drop=True)
    b_aligned = b.sort_values(list(CORE_ID_COLS)).reset_index(drop=True)
    return a_aligned, b_aligned


def weighted_average_from_weights(data_p: pd.DataFrame, data_n: pd.DataFrame) -> pd.DataFrame:
    # Align dataframes by sorting on name and turn
    data_p_aligned, data_n_aligned = align_by_name_turn(data_p, data_n)

    data_avg = data_p_aligned.copy(deep=True)

    # Combine px estimates
    px_combined, var_px_combined = combine_two_estimates(
        data_p_aligned["px"].to_numpy(),
        data_p_aligned["var_px"].to_numpy(),
        data_n_aligned["px"].to_numpy(),
        data_n_aligned["var_px"].to_numpy(),
    )
    data_avg["px"] = px_combined
    data_avg["var_px"] = var_px_combined

    # Combine py estimates
    py_combined, var_py_combined = combine_two_estimates(
        data_p_aligned["py"].to_numpy(),
        data_p_aligned["var_py"].to_numpy(),
        data_n_aligned["py"].to_numpy(),
        data_n_aligned["var_py"].to_numpy(),
    )
    data_avg["py"] = py_combined
    data_avg["var_py"] = var_py_combined

    return data_avg


def weighted_average_from_angles(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    beta_x_map: Mapping[str, float],
    beta_y_map: Mapping[str, float],
) -> pd.DataFrame:
    # Align dataframes by sorting on name and turn
    data_p_aligned, data_n_aligned = align_by_name_turn(data_p, data_n)

    data_avg = data_p_aligned.copy(deep=True)

    psi_x_prev = (data_p_aligned["delta_x_p"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_prev = (data_p_aligned["delta_y_p"].to_numpy() + 0.25) * 2 * np.pi
    psi_x_next = (data_n_aligned["delta_x_n"].to_numpy() + 0.25) * 2 * np.pi
    psi_y_next = (data_n_aligned["delta_y_n"].to_numpy() + 0.25) * 2 * np.pi

    inv_beta_x = 1.0 / data_p_aligned["betax"].to_numpy()
    inv_beta_y = 1.0 / data_p_aligned["betay"].to_numpy()
    inv_beta_p_x = 1.0 / data_p_aligned["bpm_x_p"].map(beta_x_map).to_numpy()
    inv_beta_p_y = 1.0 / data_p_aligned["bpm_y_p"].map(beta_y_map).to_numpy()
    inv_beta_n_x = 1.0 / data_n_aligned["bpm_x_n"].map(beta_x_map).to_numpy()
    inv_beta_n_y = 1.0 / data_n_aligned["bpm_y_n"].map(beta_y_map).to_numpy()

    wpx_prev = weights(psi_x_prev, inv_beta_p_x, inv_beta_x)
    wpy_prev = weights(psi_y_prev, inv_beta_p_y, inv_beta_y)
    wpx_next = weights(psi_x_next, inv_beta_n_x, inv_beta_x)
    wpy_next = weights(psi_y_next, inv_beta_n_y, inv_beta_y)

    eps = 0.0
    data_avg["px"] = (
        wpx_prev * data_p_aligned["px"].to_numpy() + wpx_next * data_n_aligned["px"].to_numpy()
    ) / (wpx_prev + wpx_next + eps)
    data_avg["py"] = (
        wpy_prev * data_p_aligned["py"].to_numpy() + wpy_next * data_n_aligned["py"].to_numpy()
    ) / (wpy_prev + wpy_next + eps)

    # Handle NaNs: if one df has NaN, use the other df's value
    mask_px_p_nan = np.isnan(data_p_aligned["px"])
    mask_px_n_nan = np.isnan(data_n_aligned["px"])
    mask_py_p_nan = np.isnan(data_p_aligned["py"])
    mask_py_n_nan = np.isnan(data_n_aligned["py"])

    # fmt: off
    data_avg["px"] = np.where(mask_px_p_nan & ~mask_px_n_nan, data_n_aligned["px"], data_avg["px"])
    data_avg["px"] = np.where(mask_px_n_nan & ~mask_px_p_nan, data_p_aligned["px"], data_avg["px"])

    data_avg["py"] = np.where(mask_py_p_nan & ~mask_py_n_nan, data_n_aligned["py"], data_avg["py"])
    data_avg["py"] = np.where(mask_py_n_nan & ~mask_py_p_nan, data_p_aligned["py"], data_avg["py"])
    # fmt: on

    # Restore original order
    return data_avg


def sync_endpoints(data_p: pd.DataFrame, data_n: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_p_out = data_p.copy(deep=True)
    data_n_out = data_n.copy(deep=True)
    for col in CORE_MOM_COLS:
        data_n_out.iloc[-1, data_n_out.columns.get_loc(col)] = data_p.iloc[
            -1, data_p.columns.get_loc(col)
        ]
        data_p_out.iloc[0, data_p_out.columns.get_loc(col)] = data_n.iloc[
            0, data_n.columns.get_loc(col)
        ]
    return data_p_out, data_n_out


def _require_full_coverage(data: pd.DataFrame, co: pd.DataFrame, what: str) -> None:
    """Fail loudly when *co* does not cover every BPM in *data*.

    Mapping a missing BPM yields NaN, which silently propagates through the
    whole reconstruction as a plausible-looking result. An explicit error is the
    only safe behaviour.
    """
    # `pandas` is a typing-only import here, so use the Series method.
    missing = set(data["name"].unique()) - set(co.index)
    if missing:
        raise ValueError(
            f"{what} is missing {len(missing)} BPM(s) present in the data: "
            f"{sorted(map(str, missing))[:10]}"
        )
