"""Estimate the momentum offset of a measurement from its closed orbit.

The estimate is an *offset from the reference orbit*, never an absolute
momentum; see :mod:`tmom_recon.reference`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tmom_recon.physics.closed_orbit import estimate_closed_orbit

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

    from tmom_recon.reference import MomentumReference

LOGGER = logging.getLogger(__name__)

# BPMs with |dx| below this carry no usable dispersive signal.
DX_TOL = 1e-2


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
    reference: MomentumReference,
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

    ``reference`` is mandatory and must carry a **measured** closed orbit. It
    cannot be replaced by a model closed orbit: the bend response
    matrix spans the entire horizontal BPM space (rank 16 of 16 on PSB ring 3),
    so an unknown dipole-error orbit is exactly degenerate with the dispersive
    orbit and a model that does not carry the machine's real errors biases pt by
    ~43% at dp/p = 1e-3. Subtracting a measured orbit cancels the error orbit
    identically, whatever it is, without needing to know the bend errors at all.

    The returned value is the momentum **offset from the reference orbit**, which
    is exactly what the reconstruction expands the dispersion in. A reference at
    ``MomentumReference.pt != 0`` additionally leaves a flat gain error of
    ~``2*pt_r*ddx/dx`` (4.6e-5 at dp/p_ref = 1e-4) on the offset itself, because
    the model dispersion is evaluated on momentum; prefer a nominal-RF blank.

    Args:
        data: Tracking data with BPM readings. Must contain columns: ["name", "x"].
        tws: Twiss parameters DataFrame. Must have column "dx" and be indexed by BPM
            name. When "ddx" is present the second-order solution is used.
        reference: The momentum origin (:class:`~tmom_recon.reference.MomentumReference`).
            Its closed orbit must cover every BPM used for the estimate.
        info: If True, log diagnostic information.
    Returns:
        The momentum offset of *data* from the reference orbit.
    Raises:
        ValueError: If *reference* is missing or does not cover the BPMs selected
            for the estimate.
    """
    if reference is None or not reference.measured:
        raise ValueError(
            "estimate_pt_from_model requires a `reference` MomentumReference built "
            "from a measured closed orbit. Neither a model closed orbit nor the "
            "pinned zero of a dynamic-part run is a valid substitute: dipole errors "
            "are exactly degenerate with the dispersive orbit at a single momentum, "
            "so a mismatched origin biases pt by tens of percent. This is the one "
            "place the measured orbit is load-bearing, which is why the requirement "
            "lives here rather than at an entry point that may never reach it."
        )
    reference_co = reference.closed_orbit
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
            "The reference closed orbit is missing BPMs used for the pt estimate: "
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
            "as an offset from the reference orbit (reference pt %s) over %d BPMs",
            order,
            pt,
            numerator,
            denominator,
            reference.pt,
            len(filtered_tws),
        )
    return pt
