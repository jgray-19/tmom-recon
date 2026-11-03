"""Estimate the momentum offset of a measurement from its closed orbit.

The estimate is an offset from the measured orbit-zero frame.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tmom_recon.model import ModelDetails, resolve_model_details
from tmom_recon.physics.closed_orbit import estimate_closed_orbit

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd


LOGGER = logging.getLogger(__name__)

DX_TOL = 1e-2


LHC_ARC_PATTERN = r"BPM.*\.0*(1[5-9]|[2-9]\d|[1-9]\d{2,})[RL]"


def _solve_pt_quadratic(numerator: float, s_dx2: float, s_ddx_dx: float) -> float:
    """Solve ``numerator = pt*s_dx2 + pt**2*s_ddx_dx`` for pt.

    Select the root nearest the first-order solution.
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


def estimate_closed_orbit_pt(
    data: pd.DataFrame,
    model_details: ModelDetails,
    *,
    closed_orbit_at_zero: pd.DataFrame,
    info: bool = True,
) -> float:
    """Estimate momentum from turn-by-turn data using a chromatic Twiss around ``pt=0``.

    ``model_details`` carries the accelerator and all lattice settings needed to
    build the dispersion model. Its ``pt`` is deliberately required to be zero:
    ``dx`` and ``ddx`` are expansion coefficients around the nominal momentum.
    The turn-by-turn readings are averaged per BPM and passed to
    :func:`estimate_pt_from_orbit`.
    """
    if model_details.pt != 0.0:
        raise ValueError(
            f"estimate_closed_orbit_pt requires ModelDetails.pt == 0.0; got {model_details.pt!r}"
        )
    tws = resolve_model_details(model_details).tws.copy(deep=True)
    tws.index = tws.index.astype(str).str.upper()
    data = data.copy(deep=True)
    data["name"] = data["name"].astype(str).str.upper()
    # BPMs the model lacks come back as NaN rows, which estimate_pt_from_orbit rejects.
    orbit = estimate_closed_orbit(data, tws).reindex(data["name"].unique())
    return estimate_pt_from_orbit(orbit, tws, closed_orbit_at_zero=closed_orbit_at_zero, info=info)


def estimate_pt_from_orbit(
    orbit: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    closed_orbit_at_zero: pd.DataFrame,
    info: bool = True,
) -> float:
    """Estimate MAD-NG ``pt`` from a measured closed orbit.

    The horizontal orbit relative to ``closed_orbit_at_zero`` is projected onto the
    model dispersion: second order when *tws* carries ``ddx``, first order otherwise.
    LHC data uses only the arc BPMs; other machines use BPMs with ``|dx| > DX_TOL``.

    Args:
        orbit: Closed orbit indexed by BPM name, with an ``x`` column.
        tws: Model Twiss about ``pt=0`` indexed by element name, with ``dx`` and
            optionally ``ddx``.
        closed_orbit_at_zero: Measured orbit that defines ``pt=0``.
        info: Log the BPM selection and the estimate.
    """
    from tmom_recon.orbit_reference import normalize_closed_orbit

    orbit = orbit.copy(deep=True)
    orbit.index = orbit.index.astype(str).str.upper()
    tws = tws.copy(deep=True)
    tws.index = tws.index.astype(str).str.upper()
    orbit_bpms = set(orbit.index)
    tws_bpms = set(tws.index)

    missing_bpms = orbit_bpms - tws_bpms
    if missing_bpms:
        raise ValueError(f"Orbit contains BPMs not present in tws: {missing_bpms}")

    extra_bpms = tws_bpms - orbit_bpms
    if extra_bpms:
        LOGGER.warning(f"tws contains BPMs not present in the orbit: {extra_bpms}")
        tws = tws.loc[tws.index.intersection(orbit_bpms)]

    is_lhc = tws.index.str.match(LHC_ARC_PATTERN).any()
    origin = normalize_closed_orbit(closed_orbit_at_zero, list(orbit_bpms))
    # Twiss order, so the projection sums run in lattice order.
    closed_orbit = orbit.loc[tws.index, ["x"]]
    closed_orbit["x"] -= origin.loc[closed_orbit.index, "x"].to_numpy()
    if is_lhc:
        filtered_co = closed_orbit[closed_orbit.index.str.match(LHC_ARC_PATTERN)]
        filtered_tws = tws.loc[filtered_co.index.unique()]
        if info:
            LOGGER.info(
                "LHC arc BPM pattern detected. Using %d BPMs for δ estimation.",
                filtered_tws.shape[0],
            )
        if filtered_tws.empty:
            raise ValueError("No BPMs available for δ estimation after filtering.")
    else:
        dispersive_bpms = tws.index[np.abs(tws["dx"]) > DX_TOL]
        filtered_co = closed_orbit[closed_orbit.index.isin(dispersive_bpms)]
        filtered_tws = tws.loc[filtered_co.index.unique()]
        if info:
            LOGGER.info(
                "Using BPMs with |dx| > %.2e. Selected %d BPMs for δ estimation.",
                DX_TOL,
                filtered_tws.shape[0],
            )
        if filtered_tws.empty:
            raise ValueError("No BPMs available for δ estimation after filtering.")

    numerator = float(np.sum(closed_orbit.loc[filtered_tws.index, "x"] * filtered_tws["dx"]))
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
            "as an offset from the frame reference over %d BPMs",
            order,
            pt,
            numerator,
            denominator,
            len(filtered_tws),
        )
    return pt
