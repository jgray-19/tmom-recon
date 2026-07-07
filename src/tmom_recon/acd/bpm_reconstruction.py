"""Local BPM momentum reconstruction from optics neighbors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tmom_recon.data.schema import NEXT, PREV
from tmom_recon.lattice.core import attach_lattice_columns, build_lattice_maps
from tmom_recon.physics.bpm_phases import (
    next_bpm_to_pi_2,
    phase_advance_difference,
    phase_advance_matrix_from_tws,
    prev_bpm_to_pi_2,
)
from tmom_recon.physics.momenta import momenta_from_next, momenta_from_prev

from .models import ACDipoleBPMWindow

if TYPE_CHECKING:
    from tmom_recon.lattice.core import LatticeMaps

# ---------------------------------------------------------------------------
# Twiss utilities
# ---------------------------------------------------------------------------


def _get_tune(tws: pd.DataFrame, header_key: str) -> float:
    """Extract a tune value from a Twiss table header or attribute.

    Args:
        tws: Twiss DataFrame, optionally carrying a ``headers`` dict.
        header_key: Key to look up (e.g. ``"q1"`` or ``"q2"``).

    Returns:
        The tune as a float.

    Raises:
        KeyError: If the key is not found in headers or as an attribute.
    """
    headers = dict(getattr(tws, "headers", {}) or {})
    if header_key in headers:
        return float(headers[header_key])
    value = getattr(tws, header_key, None)
    if value is not None and not callable(value):
        return float(value)
    raise KeyError(
        f"Twiss table is missing tune {header_key!r}; "
        "expected it in twiss headers or as an attribute."
    )


def _normalise_supplied_tune(label: str, tune: float) -> float:
    """Validate and fold a driven tune into the range ``(0, 0.5)``.

    Args:
        label: Human-readable label used in error messages (e.g. ``"dpx"``).
        tune: Raw supplied tune (may be integer + fractional part).

    Returns:
        Fractional part folded into ``(0, 0.5)``.

    Raises:
        ValueError: If the tune is non-finite, has a zero fractional part, or
            cannot be mapped into ``(0, 0.5)``.
    """
    tune_value = float(tune)
    if not np.isfinite(tune_value):
        raise ValueError(f"Provided {label} AC-dipole tune must be finite")
    fractional = tune_value % 1.0
    if np.isclose(fractional, 0.0):
        raise ValueError(f"Provided {label} AC-dipole tune must have a non-zero fractional part")
    if fractional > 0.5:
        fractional = 1.0 - fractional
    if not (0.0 < fractional < 0.5):
        raise ValueError(f"Provided {label} AC-dipole tune must map to 0 < tune < 0.5")
    return float(fractional)


# ---------------------------------------------------------------------------
# Neighbor table construction
# ---------------------------------------------------------------------------


def _require_neighbor_name(value: object, bpm_name: str, plane: str, direction: str) -> str:
    """Assert that a neighbor BPM name was found and return it as a string.

    Args:
        value: The candidate neighbor name, or ``None``/``NaN`` if not found.
        bpm_name: Name of the primary BPM (used in the error message).
        plane: Plane label ``"x"`` or ``"y"``.
        direction: ``"previous"`` or ``"next"``.

    Returns:
        The neighbor name as a string.

    Raises:
        ValueError: If *value* is ``None`` or ``NaN``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(
            f"No {direction} pi/2 BPM was found for {plane}-plane reconstruction at {bpm_name!r}"
        )
    return str(value)


def _prepare_neighbor_tables(
    tws_bpm: pd.DataFrame,
    use_immediate_neighbors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build upstream and downstream pi/2 neighbor lookup tables for all BPMs.

    Args:
        tws_bpm: Twiss DataFrame indexed by BPM names, containing ``mu1``,
            ``mu2``, and tune headers ``q1`` / ``q2``.
        use_immediate_neighbors: If ``True``, use the immediate lattice neighbors
            instead of the BPMs at the closest pi/2 phase advance.

    Returns:
        ``(prev_x, prev_y, next_x, next_y)`` DataFrames indexed by BPM name,
        each carrying the neighbor name and delta (phase_advance − 0.25) columns.
    """
    q1 = _get_tune(tws_bpm, "q1")
    q2 = _get_tune(tws_bpm, "q2")

    bwd1 = phase_advance_matrix_from_tws(tws_bpm["mu1"], q1, forward=False)
    bwd2 = phase_advance_matrix_from_tws(tws_bpm["mu2"], q2, forward=False)
    fwd1 = phase_advance_matrix_from_tws(tws_bpm["mu1"], q1, forward=True)
    fwd2 = phase_advance_matrix_from_tws(tws_bpm["mu2"], q2, forward=True)

    prev_x = prev_bpm_to_pi_2(bwd1).rename(columns={"prev_bpm": PREV.bpm_x, "delta": PREV.delta_x})
    prev_y = prev_bpm_to_pi_2(bwd2).rename(columns={"prev_bpm": PREV.bpm_y, "delta": PREV.delta_y})
    next_x = next_bpm_to_pi_2(fwd1).rename(columns={"next_bpm": NEXT.bpm_x, "delta": NEXT.delta_x})
    next_y = next_bpm_to_pi_2(fwd2).rename(columns={"next_bpm": NEXT.bpm_y, "delta": NEXT.delta_y})

    if use_immediate_neighbors:
        bpm_order = pd.Series(tws_bpm.index.astype(str), index=tws_bpm.index)
        prev_names = bpm_order.shift(1).fillna(bpm_order.iloc[-1])
        next_names = bpm_order.shift(-1).fillna(bpm_order.iloc[0])
        prev_x[PREV.bpm_x] = prev_names
        prev_y[PREV.bpm_y] = prev_names
        next_x[NEXT.bpm_x] = next_names
        next_y[NEXT.bpm_y] = next_names

        mu1 = pd.Series(tws_bpm["mu1"].to_numpy(dtype=float), index=tws_bpm.index)
        mu2 = pd.Series(tws_bpm["mu2"].to_numpy(dtype=float), index=tws_bpm.index)
        prev_x[PREV.delta_x] = (
            phase_advance_difference(mu1.shift(1).fillna(mu1.iloc[-1]), mu1, q1) - 0.25
        )
        prev_y[PREV.delta_y] = (
            phase_advance_difference(mu2.shift(1).fillna(mu2.iloc[-1]), mu2, q2) - 0.25
        )
        next_x[NEXT.delta_x] = (
            phase_advance_difference(mu1, mu1.shift(-1).fillna(mu1.iloc[0]), q1) - 0.25
        )
        next_y[NEXT.delta_y] = (
            phase_advance_difference(mu2, mu2.shift(-1).fillna(mu2.iloc[0]), q2) - 0.25
        )

    return prev_x, prev_y, next_x, next_y


# ---------------------------------------------------------------------------
# Local BPM reconstruction
# ---------------------------------------------------------------------------


def _build_local_reconstruction_rows(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    bpm_name: str,
    *,
    pt_est: float,
) -> tuple[pd.DataFrame, LatticeMaps]:
    """Extract rows for a single BPM and attach lattice map columns.

    Args:
        data: Full measurement DataFrame.
        tws_bpm: Twiss DataFrame restricted to BPMs.
        bpm_name: Name of the BPM to extract.
        pt_est: Estimated MAD-NG pt used to decide whether to include dispersion.

    Returns:
        ``(rows, maps)`` where *rows* is the filtered DataFrame with lattice
        columns attached and *maps* is the lattice-map object.
    """
    maps = build_lattice_maps(tws_bpm, include_dispersion=not np.isclose(pt_est, 0.0))
    rows = data.loc[data["name"] == bpm_name, ["name", "turn", "x", "y", "var_x", "var_y"]].copy(
        deep=True
    )
    rows = attach_lattice_columns(rows, maps)
    return rows, maps


def _merge_prev_neighbor_data(
    rows: pd.DataFrame,
    data: pd.DataFrame,
    bpm_index: dict[str, int],
) -> pd.DataFrame:
    """Merge previous-neighbor BPM observations onto the primary-BPM rows.

    Args:
        rows: Primary-BPM rows already carrying neighbor-name and delta columns.
        data: Full measurement DataFrame.
        bpm_index: Mapping from BPM name to its lattice position index.

    Returns:
        *rows* with previous-neighbor position and variance columns merged in.
    """
    current_idx = bpm_index[str(rows["name"].iat[0])]
    x_neighbor_idx = bpm_index[str(rows[PREV.bpm_x].iat[0])]
    y_neighbor_idx = bpm_index[str(rows[PREV.bpm_y].iat[0])]

    rows["turn_x_p"] = rows["turn"] - int(current_idx < x_neighbor_idx)
    rows["turn_y_p"] = rows["turn"] - int(current_idx < y_neighbor_idx)

    x_coords = data[["turn", "name", "x", "var_x"]].rename(
        columns={"turn": "turn_x_p", "name": PREV.bpm_x, "x": PREV.x, "var_x": PREV.var_x}
    )
    y_coords = data[["turn", "name", "y", "var_y"]].rename(
        columns={"turn": "turn_y_p", "name": PREV.bpm_y, "y": PREV.y, "var_y": PREV.var_y}
    )
    rows = rows.merge(x_coords, on=["turn_x_p", PREV.bpm_x], how="left")
    rows = rows.merge(y_coords, on=["turn_y_p", PREV.bpm_y], how="left")
    return rows.drop(columns=["turn_x_p", "turn_y_p"])


def _merge_next_neighbor_data(
    rows: pd.DataFrame,
    data: pd.DataFrame,
    bpm_index: dict[str, int],
) -> pd.DataFrame:
    """Merge next-neighbor BPM observations onto the primary-BPM rows.

    Args:
        rows: Primary-BPM rows already carrying neighbor-name and delta columns.
        data: Full measurement DataFrame.
        bpm_index: Mapping from BPM name to lattice position index.

    Returns:
        *rows* with next-neighbor position and variance columns merged in.
    """
    current_idx = bpm_index[str(rows["name"].iat[0])]
    x_neighbor_idx = bpm_index[str(rows[NEXT.bpm_x].iat[0])]
    y_neighbor_idx = bpm_index[str(rows[NEXT.bpm_y].iat[0])]

    rows["turn_x_n"] = rows["turn"] + int(current_idx > x_neighbor_idx)
    rows["turn_y_n"] = rows["turn"] + int(current_idx > y_neighbor_idx)

    x_coords = data[["turn", "name", "x", "var_x"]].rename(
        columns={"turn": "turn_x_n", "name": NEXT.bpm_x, "x": NEXT.x, "var_x": NEXT.var_x}
    )
    y_coords = data[["turn", "name", "y", "var_y"]].rename(
        columns={"turn": "turn_y_n", "name": NEXT.bpm_y, "y": NEXT.y, "var_y": NEXT.var_y}
    )
    rows = rows.merge(x_coords, on=["turn_x_n", NEXT.bpm_x], how="left")
    rows = rows.merge(y_coords, on=["turn_y_n", NEXT.bpm_y], how="left")
    return rows.drop(columns=["turn_x_n", "turn_y_n"])


def _prepare_prev_reconstruction(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    bpm_name: str,
    prev_x: pd.DataFrame,
    prev_y: pd.DataFrame,
    bpm_index: dict[str, int],
    *,
    pt_est: float = 0.0,
) -> pd.DataFrame:
    """Build a momentum-reconstruction DataFrame for a single upstream BPM.

    Args:
        data: Full measurement DataFrame.
        tws_bpm: Twiss DataFrame restricted to BPMs.
        bpm_name: Name of the upstream BPM.
        prev_x: Previous-neighbor lookup table for the x-plane.
        prev_y: Previous-neighbor lookup table for the y-plane.
        bpm_index: Mapping from BPM name to lattice position index.
        pt_est: Estimated MAD-NG pt.

    Returns:
        DataFrame with reconstructed ``px``/``py`` and associated variances.
    """
    rows, maps = _build_local_reconstruction_rows(data, tws_bpm, bpm_name, pt_est=pt_est)
    rows[PREV.bpm_x] = _require_neighbor_name(
        prev_x.at[bpm_name, PREV.bpm_x], bpm_name, "x", "previous"
    )
    rows[PREV.delta_x] = float(prev_x.at[bpm_name, PREV.delta_x])
    rows[PREV.bpm_y] = _require_neighbor_name(
        prev_y.at[bpm_name, PREV.bpm_y], bpm_name, "y", "previous"
    )
    rows[PREV.delta_y] = float(prev_y.at[bpm_name, PREV.delta_y])
    rows["sqrt_betax_p"] = rows[PREV.bpm_x].map(maps.sqrt_betax)
    rows["sqrt_betay_p"] = rows[PREV.bpm_y].map(maps.sqrt_betay)
    if maps.dx is not None and maps.dy is not None:
        rows[PREV.dx] = rows[PREV.bpm_x].map(maps.dx)
        rows[PREV.dy] = rows[PREV.bpm_y].map(maps.dy)
    rows = _merge_prev_neighbor_data(rows, data, bpm_index)
    return momenta_from_prev(rows, pt_est)


def _prepare_next_reconstruction(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    bpm_name: str,
    next_x: pd.DataFrame,
    next_y: pd.DataFrame,
    bpm_index: dict[str, int],
    *,
    pt_est: float = 0.0,
) -> pd.DataFrame:
    """Build a momentum-reconstruction DataFrame for a single downstream BPM.

    Args:
        data: Full measurement DataFrame.
        tws_bpm: Twiss DataFrame restricted to BPMs.
        bpm_name: Name of the downstream BPM.
        next_x: Next-neighbor lookup table for the x-plane.
        next_y: Next-neighbor lookup table for the y-plane.
        bpm_index: Mapping from BPM name to lattice position index.
        pt_est: Estimated MAD-NG pt.

    Returns:
        DataFrame with reconstructed ``px``/``py`` and associated variances.
    """
    rows, maps = _build_local_reconstruction_rows(data, tws_bpm, bpm_name, pt_est=pt_est)
    rows[NEXT.bpm_x] = _require_neighbor_name(
        next_x.at[bpm_name, NEXT.bpm_x], bpm_name, "x", "next"
    )
    rows[NEXT.delta_x] = float(next_x.at[bpm_name, NEXT.delta_x])
    rows[NEXT.bpm_y] = _require_neighbor_name(
        next_y.at[bpm_name, NEXT.bpm_y], bpm_name, "y", "next"
    )
    rows[NEXT.delta_y] = float(next_y.at[bpm_name, NEXT.delta_y])
    rows["sqrt_betax_n"] = rows[NEXT.bpm_x].map(maps.sqrt_betax)
    rows["sqrt_betay_n"] = rows[NEXT.bpm_y].map(maps.sqrt_betay)
    if maps.dx is not None and maps.dy is not None:
        rows[NEXT.dx] = rows[NEXT.bpm_x].map(maps.dx)
        rows[NEXT.dy] = rows[NEXT.bpm_y].map(maps.dy)
    rows = _merge_next_neighbor_data(rows, data, bpm_index)
    return momenta_from_next(rows, pt_est)


def prepare_direct_bpm_reconstruction(
    data: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    *,
    window: ACDipoleBPMWindow,
    bpm_index: dict[str, int],
    pt_est: float = 0.0,
    use_immediate_neighbors: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Build per-BPM reconstruction DataFrames for all BPMs in the window.

    Args:
        data: Full measurement DataFrame.
        tws_bpm: Twiss DataFrame restricted to BPMs.
        window: The selected upstream/downstream BPM window.
        bpm_index: Mapping from BPM name to lattice position index.
        pt_est: Estimated MAD-NG pt.
        use_immediate_neighbors: Passed through to
            :func:`_prepare_neighbor_tables`.

    Returns:
        ``(upstream_frames, downstream_frames)`` where each is a dict mapping
        BPM name to its reconstructed-momentum DataFrame.
    """
    prev_x, prev_y, next_x, next_y = _prepare_neighbor_tables(
        tws_bpm, use_immediate_neighbors=use_immediate_neighbors
    )
    upstream_frames = {
        bpm_name: _prepare_prev_reconstruction(
            data, tws_bpm, bpm_name, prev_x, prev_y, bpm_index, pt_est=pt_est
        )
        for bpm_name in window.upstream
    }
    downstream_frames = {
        bpm_name: _prepare_next_reconstruction(
            data, tws_bpm, bpm_name, next_x, next_y, bpm_index, pt_est=pt_est
        )
        for bpm_name in window.downstream
    }
    return upstream_frames, downstream_frames
