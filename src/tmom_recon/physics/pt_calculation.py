from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

# NEW: we need pandas for the aligned return
import pandas as pd

from tmom_recon.data.config import FILE_COLUMNS
from tmom_recon.physics.bpm_phases import (
    next_bpm_to_pi,
    next_bpm_to_pi_2,
    phase_advance_matrix_from_tws,
    prev_bpm_to_pi,
    prev_bpm_to_pi_2,
)
from tmom_recon.physics.closed_orbit import estimate_closed_orbit

LOGGER = logging.getLogger(__name__)
OUT_COLS = list(FILE_COLUMNS)

Direction = Literal["next", "prev"]

DX_TOL = 1e-2
DEN_TOL = 1e-10


@dataclass(frozen=True)
class TwissMaps:
    sqrt_betax: dict[str, float]
    dx: dict[str, float]
    x_co: dict[str, float]


def _twiss_maps(tws: pd.DataFrame) -> TwissMaps:
    sqrt_betax = np.sqrt(tws["beta11"]).to_dict()
    dx = tws["dx"].to_dict()
    x_co = tws["x"].to_dict()
    return TwissMaps(sqrt_betax=sqrt_betax, dx=dx, x_co=x_co)


def _partner_tables(direction: Direction, mu1, q1):
    if direction == "next":
        fwd = phase_advance_matrix_from_tws(mu1, q1, forward=True)
        tbl_pi = next_bpm_to_pi(fwd).rename(columns={"next_bpm": "bpm_bar", "delta": "d_pi"})
        tbl_pi2 = next_bpm_to_pi_2(fwd).rename(columns={"next_bpm": "bpm_tilde", "delta": "d_pi2"})
    else:
        bwd = phase_advance_matrix_from_tws(mu1, q1, forward=False)
        tbl_pi = prev_bpm_to_pi(bwd).rename(columns={"prev_bpm": "bpm_bar", "delta": "d_pi"})
        tbl_pi2 = prev_bpm_to_pi_2(bwd).rename(columns={"prev_bpm": "bpm_tilde", "delta": "d_pi2"})
    return tbl_pi, tbl_pi2


def _wrap_turns(
    direction: Direction, df, bpm_index: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    cur_i = df["name"].map(bpm_index)
    i_bar = df["bpm_bar"].map(bpm_index)
    i_tilde = df["bpm_tilde"].map(bpm_index)
    if direction == "next":
        turn_bar = df["turn"] + (cur_i > i_bar).astype(np.int16)
        turn_tilde = df["turn"] + (cur_i > i_tilde).astype(np.int16)
    else:
        turn_bar = df["turn"] - (cur_i < i_bar).astype(np.int16)
        turn_tilde = df["turn"] - (cur_i < i_tilde).astype(np.int16)
    return turn_bar, turn_tilde


def _merge_partner_coords(df):
    coords = df[["turn", "name", "x"]]
    coords_bar = coords.rename(columns={"turn": "turn_bar", "name": "bpm_bar", "x": "x_bar"})[
        ["turn_bar", "bpm_bar", "x_bar"]
    ]
    coords_tilde = coords.rename(
        columns={"turn": "turn_tilde", "name": "bpm_tilde", "x": "x_tilde"}
    )[["turn_tilde", "bpm_tilde", "x_tilde"]]
    df = df.merge(coords_bar, on=["turn_bar", "bpm_bar"], how="left", copy=False)
    df = df.merge(coords_tilde, on=["turn_tilde", "bpm_tilde"], how="left", copy=False)

    if df[["x_bar", "x_tilde"]].isnull().any().any():
        # Check that the turn of the null values is either 0 (prev) or max turn (next), otherwise log
        max_turn = df["turn"].max()
        null_rows = df[df["x_bar"].isnull() | df["x_tilde"].isnull()]
        for _, row in null_rows.iterrows():
            if row["turn"] != 0 and row["turn"] != max_turn:
                LOGGER.warning(
                    "Missing partner coordinates for BPM %s at turn %d",
                    row["name"],
                    row["turn"],
                )
    # Clean up
    return df.drop(columns=["turn_bar", "turn_tilde"])


def _phases_radians(d_pi: np.ndarray, d_pi2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    psi_bar = (d_pi + 0.5) * 2.0 * np.pi
    psi_tilde = (d_pi2 + 0.25) * 2.0 * np.pi
    return psi_bar, psi_tilde


def _compute_delta(
    x1: np.ndarray,
    x_bar: np.ndarray,
    x_tilde: np.ndarray,
    sqrt_beta_1: np.ndarray,
    sqrt_beta_bar: np.ndarray,
    sqrt_beta_tilde: np.ndarray,
    eta_1: np.ndarray,
    eta_bar: np.ndarray,
    eta_tilde: np.ndarray,
    psi_bar: np.ndarray,
    psi_tilde: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sin_bar = np.sin(psi_bar)
    sin_tilde = np.sin(psi_tilde)
    sin_bar_minus_tilde = np.sin(psi_bar - psi_tilde)
    num = (
        (x_tilde / sqrt_beta_tilde) * sin_bar
        - (x_bar / sqrt_beta_bar) * sin_tilde
        - (x1 / sqrt_beta_1) * sin_bar_minus_tilde
    )
    den = (
        (eta_tilde / sqrt_beta_tilde) * sin_bar
        - (eta_bar / sqrt_beta_bar) * sin_tilde
        - (eta_1 / sqrt_beta_1) * sin_bar_minus_tilde
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pt_all = num / den
    return num, den, pt_all


def _valid_mask(eta_1, eta_bar, eta_tilde, den) -> np.ndarray:
    # valid_dx = (
    #     ~np.isclose(eta_1, 0.0, atol=DX_TOL)
    #     & ~np.isclose(eta_bar, 0.0, atol=DX_TOL)
    #     & ~np.isclose(eta_tilde, 0.0, atol=DX_TOL)
    #     & (eta_1 > 0.0)
    #     & (eta_bar > 0.0)
    #     & (eta_tilde > 0.0)
    # )
    # valid_den = np.abs(den) > DEN_TOL
    # return valid_dx & valid_den
    return np.abs(den) > DEN_TOL


def _maybe_log_stats(pt: np.ndarray, info: bool, label: str = "") -> None:
    if not info:
        return
    prefix = f"[{label}] " if label else ""
    if pt.size:
        LOGGER.info(prefix + "Count: %d", pt.size)
        LOGGER.info(prefix + "pt mean: %s", np.nanmean(pt))
        LOGGER.info(prefix + "pt std:  %s", np.nanstd(pt))
        LOGGER.info(prefix + "pt min:  %s", np.nanmin(pt))
        LOGGER.info(prefix + "pt max:  %s", np.nanmax(pt))
    else:
        LOGGER.warning(prefix + "All samples were filtered out (check dx/den tolerances).")


def _calculate_pt_direction_aligned(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    maps: TwissMaps,
    direction: Direction,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Internal helper: compute per-row pt for a given direction.
    Returns:
        pt_all : np.ndarray (same length as data), may contain inf/NaN where invalid
        valid     : boolean mask of rows considered valid (dispersion & denominator)
    """
    df = data.copy(deep=True)
    mu1 = tws["mu1"]
    tbl_pi, tbl_pi2 = _partner_tables(direction, mu1, tws.q1)
    df = df.join(tbl_pi, on="name")
    df = df.join(tbl_pi2, on="name", rsuffix="_pi2")

    bpm_index = {b: i for i, b in enumerate(tws.index.to_list())}
    turn_bar, turn_tilde = _wrap_turns(direction, df, bpm_index)
    df["turn_bar"] = turn_bar
    df["turn_tilde"] = turn_tilde
    df = _merge_partner_coords(df)

    sqrt_beta_1 = df["name"].map(maps.sqrt_betax).to_numpy()
    sqrt_beta_bar = df["bpm_bar"].map(maps.sqrt_betax).to_numpy()
    sqrt_beta_tilde = df["bpm_tilde"].map(maps.sqrt_betax).to_numpy()

    eta_1 = df["name"].map(maps.dx).to_numpy()
    eta_bar = df["bpm_bar"].map(maps.dx).to_numpy()
    eta_tilde = df["bpm_tilde"].map(maps.dx).to_numpy()

    psi_bar, psi_tilde = _phases_radians(df["d_pi"].to_numpy(), df["d_pi2"].to_numpy())

    x1 = df["x"].to_numpy()
    x_bar = df["x_bar"].to_numpy()
    x_tilde = df["x_tilde"].to_numpy()

    _, den, pt_all = _compute_delta(
        x1,
        x_bar,
        x_tilde,
        sqrt_beta_1,
        sqrt_beta_bar,
        sqrt_beta_tilde,
        eta_1,
        eta_bar,
        eta_tilde,
        psi_bar,
        psi_tilde,
    )

    valid = _valid_mask(eta_1, eta_bar, eta_tilde, den)
    return pt_all, valid


def calculate_pt_both(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    info: bool = True,
) -> pd.DataFrame:
    """
    Compute MAD-NG pt using BOTH 'prev' and 'next' partner BPM triplets.
    Returns a DataFrame aligned to the input rows with columns:
        - 'pt_prev'
        - 'pt_next'
        - 'pt_avg'  (simple unweighted average where both are valid; NaN otherwise)
    """
    LOGGER.info("Calculating pt using BOTH directions (prev + next)")
    # Make a copy to avoid modifying input
    data = data.copy(deep=True)

    # Ensure tws is indexed by BPM name
    if tws.index.name != "name":
        tws = tws.set_index("name")

    # Reduce data and tws rows based on BPMs present in data
    bpms_in_data = data["name"].unique().tolist()
    tws = tws.loc[tws.index.isin(bpms_in_data)]

    maps = _twiss_maps(tws)

    # remove closed orbit
    data["x"] = data["x"] - data["name"].map(maps.x_co)

    # Per-direction, aligned arrays and masks
    next_all, next_valid = _calculate_pt_direction_aligned(data, tws, maps, "next")
    prev_all, prev_valid = _calculate_pt_direction_aligned(data, tws, maps, "prev")

    # Build aligned Series with NaN where invalid
    pt_next = np.where(next_valid, next_all, np.nan)
    pt_prev = np.where(prev_valid, prev_all, np.nan)

    # Unweighted average only where both are valid
    both_valid = next_valid & prev_valid
    pt_avg = np.full_like(pt_next, np.nan, dtype=float)
    pt_avg[both_valid] = 0.5 * (pt_prev[both_valid] + pt_next[both_valid])

    # Diagnostics
    _maybe_log_stats(pt_prev[prev_valid], info, label="prev")
    _maybe_log_stats(pt_next[next_valid], info, label="next")
    _maybe_log_stats(pt_avg[np.isfinite(pt_avg)], info, label="avg")

    # Return as DataFrame aligned to input index
    return pd.DataFrame(
        {
            "pt_prev": pt_prev,
            "pt_next": pt_next,
            "pt_avg": pt_avg,
        },
        index=data.index,
    )


def get_mean_pt(data: pd.DataFrame, tws: pd.DataFrame, info: bool = True) -> float:
    """
    Compute mean pt using BOTH 'prev' and 'next' partner BPM triplets.
    Returns a single float value representing the mean pt across all valid measurements.
    """
    pt_df = calculate_pt_both(data, tws=tws, info=info)
    mean_pt = np.nanmean(pt_df["pt_avg"])
    if info:
        LOGGER.info("Mean pt across all valid measurements: %s", mean_pt)
    return mean_pt if np.isfinite(mean_pt) else 0.0


LHC_ARC_PATTERN = r"BPM.*\.0*(1[5-9]|[2-9]\d|[1-9]\d{2,})[RL]"


def _solve_pt_quadratic(numerator: float, s_dx2: float, s_ddx_dx: float) -> float:
    """Solve ``numerator = pt*s_dx2 + pt**2*s_ddx_dx`` for pt.

    ``pt`` and ``pt**2`` are one unknown, not two, so a single orbit determines
    the second-order solution -- no momentum scan is needed. Of the two roots,
    the physical one is adjacent to the first-order solution ``numerator/s_dx2``
    (the quadratic term is a ~0.2% correction at dp/p = 8e-3, so the roots are
    nowhere near each other).
    """
    linear = numerator / s_dx2
    if s_ddx_dx == 0.0:
        return linear
    discriminant = s_dx2 * s_dx2 + 4.0 * s_ddx_dx * numerator
    if discriminant < 0.0:
        LOGGER.warning(
            "Second-order pt solve has no real root (discriminant %.3e); "
            "falling back to the first-order estimate.",
            discriminant,
        )
        return linear
    root = np.sqrt(discriminant)
    candidates = ((-s_dx2 + root) / (2.0 * s_ddx_dx), (-s_dx2 - root) / (2.0 * s_ddx_dx))
    return min(candidates, key=lambda value: abs(value - linear))


def estimate_pt_from_model(
    data: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    reference_co: pd.DataFrame,
    info: bool = True,
) -> float:
    """
    Estimate MAD-NG pt from the closed orbit, using first- and second-order dispersion.

    The orbit is projected onto the model dispersion,
    ``pt = sum(x_co*dx) / sum(dx**2)``, extended to second order by solving
    ``sum(x_co*dx) = pt*sum(dx**2) + pt**2*sum(ddx*dx)`` whenever the twiss
    carries ``ddx``. On PSB ring 3 that removes a relative bias of 2.3e-4 to
    2.3e-3 (growing with dp/p), leaving 3e-7 to 2e-5 -- a 97-670x reduction.
    Note this is a *bias*: unlike BPM noise it does not average down with turns.

    ``reference_co`` is mandatory and must be a **measured** closed orbit taken at
    nominal RF. It cannot be replaced by a model closed orbit: the bend response
    matrix spans the entire horizontal BPM space (rank 16 of 16 on PSB ring 3),
    so an unknown dipole-error orbit is exactly degenerate with the dispersive
    orbit and a model that does not carry the machine's real errors biases pt by
    ~43% at dp/p = 1e-3. Subtracting a measured orbit cancels the error orbit
    identically, whatever it is, without needing to know the bend errors at all.

    The estimate is relative to whatever momentum the reference orbit sat at. A
    reference at ``pt_r != 0`` leaves a flat gain error of ~``2*pt_r*ddx/dx``
    (4.6e-5 at dp/p_ref = 1e-4), which is why the reference must be at nominal RF.

    Args:
        data: Tracking data with BPM readings. Must contain columns: ["name", "x"].
        tws: Twiss parameters DataFrame. Must have column "dx" and be indexed by BPM
            name. When "ddx" is present the second-order solution is used.
        reference_co: Measured closed orbit at nominal RF, indexed by BPM name with
            an "x" column. Must cover every BPM used for the estimate.
        info: If True, log diagnostic information.
    Returns:
        Estimated pt value as a float, relative to *reference_co*.
    Raises:
        ValueError: If *reference_co* is missing, malformed, or does not cover the
            BPMs selected for the estimate.
    """
    if reference_co is None:
        raise ValueError(
            "estimate_pt_from_model requires `reference_co`, a measured closed orbit "
            "at nominal RF. A model closed orbit is not a valid substitute: dipole "
            "errors are exactly degenerate with the dispersive orbit at a single "
            "momentum, so a mismatched model biases pt by tens of percent."
        )
    if "x" not in getattr(reference_co, "columns", ()):
        raise ValueError('`reference_co` must be a DataFrame with an "x" column.')
    data_bpms = set(data["name"].unique())
    tws_bpms = set(tws.index)

    missing_bpms = data_bpms - tws_bpms
    if missing_bpms:
        raise ValueError(f"Data contains BPMs not present in tws: {missing_bpms}")

    extra_bpms = tws_bpms - data_bpms
    if extra_bpms:
        LOGGER.warning(f"tws contains BPMs not present in data: {extra_bpms}")
        tws = tws.loc[tws.index.intersection(data_bpms)]

    is_lhc = tws.index.str.match(LHC_ARC_PATTERN).any()
    closed_orbit = estimate_closed_orbit(data, tws)

    if is_lhc:
        filtered_co = closed_orbit[closed_orbit.index.str.match(LHC_ARC_PATTERN)]
        filtered_tws = tws.loc[filtered_co.index.unique()]
        if info:
            LOGGER.info(
                "LHC arc BPM pattern detected. Using %d BPMs for δ estimation.",
                filtered_tws.shape[0],
            )
    else:
        bpms_with_small_dx = tws[np.abs(tws["dx"]) > DX_TOL].index
        filtered_co = closed_orbit[closed_orbit.index.isin(bpms_with_small_dx)]
        filtered_tws = tws.loc[filtered_co.index.unique()]
        if info:
            LOGGER.info(
                "Using BPMs with |dx| > %.2e. Selected %d BPMs for δ estimation.",
                DX_TOL,
                filtered_tws.shape[0],
            )
    if filtered_tws.empty:
        raise ValueError("No BPMs available for δ estimation after filtering.")

    missing_reference = filtered_co.index.difference(reference_co.index)
    if len(missing_reference):
        raise ValueError(
            "`reference_co` is missing BPMs used for the pt estimate: "
            f"{sorted(map(str, missing_reference))}"
        )
    # Referencing to a *measured* nominal-RF orbit cancels the machine's error
    # closed orbit identically; see the docstring for why a model CO cannot.
    orbit = filtered_co["x"] - reference_co.loc[filtered_co.index, "x"].astype(float)

    numerator = float(np.sum(orbit * filtered_tws["dx"]))
    denominator = float(np.sum(filtered_tws["dx"] ** 2))

    if "ddx" in filtered_tws.columns:
        s_ddx_dx = float(np.sum(filtered_tws["ddx"] * filtered_tws["dx"]))
        pt = _solve_pt_quadratic(numerator, denominator, s_ddx_dx)
        order = "second"
    else:
        LOGGER.warning(
            "Twiss has no 'ddx' column; falling back to first-order dispersion. "
            "This leaves a relative pt bias growing with dp/p (2.3e-3 at 1e-2 on "
            "PSB ring 3). Run the model twiss with chrom=True to enable it."
        )
        pt = numerator / denominator
        order = "first"

    if info:
        LOGGER.info(
            "Estimated pt from %s-order dispersion: %s (from %.2e/%.2e), "
            "referenced to a measured nominal-RF closed orbit over %d BPMs",
            order,
            pt,
            numerator,
            denominator,
            len(filtered_tws),
        )
    return pt
