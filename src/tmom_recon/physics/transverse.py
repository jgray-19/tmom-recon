"""Core neighbour-pair momentum reconstruction pipeline.

Internal module: the public entry point is
:func:`tmom_recon.reconstruction.calculate_pz`, which resolves the optics
sources and delegates here.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

from tmom_recon.data.columns import NEIGHBOR_BPM_ERROR_SPEC
from tmom_recon.data.schema import SUFFIX_NEXT, SUFFIX_PREV
from tmom_recon.lattice.core import (
    OUT_COLS,
    diagnostics,
    remove_closed_orbit,
    restore_closed_orbit_and_reference_momenta,
    sync_endpoints,
    validate_input,
)
from tmom_recon.lattice.core import (
    weighted_average_from_weights as weighted_average,
)
from tmom_recon.lattice.neighbors import (
    compute_turn_wraps,
    merge_neighbor_coords,
    prepare_neighbor_views,
)
from tmom_recon.physics.momenta import (
    momenta_from_next,
    momenta_from_prev,
)
from tmom_recon.physics.pt_calculation import estimate_pt_from_model

if TYPE_CHECKING:
    from collections.abc import Collection

    import pandas as pd

    from tmom_recon.optics import ResolvedOptics

LOGGER = logging.getLogger(__name__)

# Uncertainty columns attached to every row from the resolved twiss
CURRENT_ERROR_COLS = ("sqrt_betax_err", "sqrt_betay_err", "alfax_err", "alfay_err")
DISPERSION_ERROR_COLS = ("dx_err", "dy_err", "dpx_err", "dpy_err")


def attach_error_columns(
    data_p: pd.DataFrame,
    data_n: pd.DataFrame,
    tws: pd.DataFrame,
    *,
    use_dispersion: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return neighbour views with current-BPM and neighbour-BPM uncertainty columns.

    Args:
        data_p: Previous-neighbour view (carries ``bpm_x_p``/``bpm_y_p``).
        data_n: Next-neighbour view (carries ``bpm_x_n``/``bpm_y_n``).
        tws: Resolved twiss with ``*_err`` columns.
        use_dispersion: Whether dispersion error columns are required.

    Raises:
        KeyError: If a required uncertainty column is missing from *tws*.
    """
    required = list(CURRENT_ERROR_COLS)
    if use_dispersion:
        required += list(DISPERSION_ERROR_COLS)
    missing = [col for col in required if col not in tws.columns]
    if missing:
        raise KeyError(f"Resolved twiss is missing uncertainty columns: {missing}")

    data_p_out = data_p.copy(deep=True)
    data_n_out = data_n.copy(deep=True)
    tws_dict = {col: tws[col].to_dict() for col in required}
    for frame in (data_p_out, data_n_out):
        for err_col in required:
            frame[err_col] = frame["name"].map(tws_dict[err_col])

    for frame, suffix in ((data_p_out, SUFFIX_PREV), (data_n_out, SUFFIX_NEXT)):
        for target_tpl, neighbor_tpl, source_col in NEIGHBOR_BPM_ERROR_SPEC:
            if source_col not in tws_dict:
                continue
            target = target_tpl.format(suffix)
            neighbor = neighbor_tpl.format(suffix)
            frame[target] = frame[neighbor].map(tws_dict[source_col])
    return data_p_out, data_n_out


def reconstruct_momenta(
    orig_data: pd.DataFrame,
    optics: ResolvedOptics,
    *,
    pt_override: float | None = None,
    info: bool = True,
    barrier_s: float | None = None,
    bpm_names: Collection[str] | None = None,
) -> pd.DataFrame:
    """Reconstruct per-turn transverse momenta at every BPM.

    Args:
        orig_data: Turn-by-turn BPM data with columns ``name, turn, x, y``
            and position variances ``var_x, var_y``.
        optics: Resolved optics bundle from :func:`tmom_recon.optics.resolve_optics`.
        pt_override: Use this MAD-NG pt instead of estimating it.
        info: Whether to log diagnostics.
        barrier_s: Optional longitudinal position of a localised element (e.g.
            an AC dipole) that the neighbour-pair reconstruction must not
            transport across.
        bpm_names: Optional subset of BPM names to reconstruct. All BPMs in
            *orig_data* are still available as neighbour readings.

    Returns:
        DataFrame with the standard output columns and ``attrs["PT_EST"]``.
    """
    features = validate_input(orig_data)
    data = orig_data.copy(deep=True)
    with contextlib.suppress(AttributeError, TypeError, ValueError):
        data["name"] = data["name"].astype("category")

    tws = optics.tws
    shared_bpm_names = set(tws.index).intersection(data["name"].unique())
    data = data[data["name"].isin(shared_bpm_names)]
    tws = tws.loc[tws.index.isin(shared_bpm_names)]

    data = remove_closed_orbit(data, optics.co)
    complete_data = data
    if bpm_names is not None:
        requested = set(bpm_names)
        data = data[data["name"].isin(requested)]

    if pt_override is not None:
        pt_est = float(pt_override)
        LOGGER.info("Using provided pt override: %s", pt_est)
    elif optics.use_dispersion:
        pt_tws = optics.co if "dx" in optics.co.columns else tws
        pt_est = estimate_pt_from_model(complete_data, pt_tws, info)
    else:
        pt_est = 0.0

    data_p, data_n, bpm_index, _maps = prepare_neighbor_views(
        data,
        tws,
        complete_data=complete_data,
        include_dispersion=optics.use_dispersion,
        include_errors=True,
        barrier_s=barrier_s,
    )
    data_p, data_n = attach_error_columns(data_p, data_n, tws, use_dispersion=optics.use_dispersion)

    turn_x_p, turn_y_p, turn_x_n, turn_y_n = compute_turn_wraps(data_p, data_n, bpm_index)
    data_p, data_n = merge_neighbor_coords(
        data_p,
        data_n,
        turn_x_p,
        turn_y_p,
        turn_x_n,
        turn_y_n,
        complete_data=complete_data,
    )

    data_p = momenta_from_prev(data_p, pt_est, include_optics_errors=True)
    data_n = momenta_from_next(data_n, pt_est, include_optics_errors=True)

    if bpm_names is None:
        data_p, data_n = sync_endpoints(data_p, data_n)

    data_avg = weighted_average(data_p, data_n)

    data_avg = restore_closed_orbit_and_reference_momenta(data_avg, optics.co)

    data_avg.attrs["PT_EST"] = pt_est

    # Restore original order of orig_data
    orig_output = orig_data if bpm_names is None else orig_data[orig_data["name"].isin(requested)]
    orig_order = orig_output.set_index(["name", "turn"]).index
    data_avg = data_avg.set_index(["name", "turn"]).reindex(orig_order).reset_index()

    for col in OUT_COLS:
        if col not in data_avg.columns:
            if col in orig_data.columns:
                data_avg[col] = orig_output[col].to_numpy()
            else:
                raise KeyError(
                    f"Required output column {col!r} is missing from both "
                    "the reconstructed data and the original input data."
                )

    diagnostics(orig_data, data_p, data_n, data_avg, info, features)
    return data_avg[OUT_COLS]
