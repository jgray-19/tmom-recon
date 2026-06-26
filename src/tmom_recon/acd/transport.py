"""MAD-NG state transport between BPMs and the AC-dipole marker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .bpm_reconstruction import _restore_reference_momenta
from .models import ACDipoleStateEstimate, ACDipoleStateSeries

if TYPE_CHECKING:
    import pandas as pd

    from .madng_driver import ACDipoleMadDriver


def _transport_to_marker(
    frame: pd.DataFrame,
    model: ACDipoleMadDriver,
    *,
    source_name: str,
    direction: int,
) -> ACDipoleStateSeries:
    """Transport a BPM-reconstructed state batch to the AC-dipole marker.

    Args:
        frame: Reconstructed-momentum DataFrame for a single BPM.
        model: MAD-NG driver used for particle tracking.
        source_name: Name of the source BPM element.
        direction: ``+1`` for forward (upstream → marker), ``-1`` for backward.

    Returns:
        An :class:`ACDipoleStateSeries` at the marker location.
    """
    source_state = frame[["x", "px", "y", "py"]].to_numpy(dtype=float)
    marker_name = model.acd_before if direction == +1 else model.acd_after
    s = model.track_particles(source_name, marker_name, source_state, direction=direction)
    return ACDipoleStateSeries(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5])


def build_tracked_state_table(
    frame: pd.DataFrame,
    model: ACDipoleMadDriver,
    *,
    source_name: str,
    direction: int,
) -> pd.DataFrame:
    """Build a turn-sorted DataFrame of marker states transported from a single BPM.

    Args:
        frame: Reconstructed-momentum DataFrame for a single BPM.
        model: MAD-NG driver used for particle tracking.
        source_name: Name of the source BPM element.
        direction: ``+1`` for forward, ``-1`` for backward.

    Returns:
        DataFrame with columns ``turn, source_bpm, x, px, y, py, t, pt,
        var_x, var_px, var_y, var_py``, sorted by turn.
    """
    tracked = _transport_to_marker(frame, model, source_name=source_name, direction=direction)
    table = frame[["turn", "var_x", "var_px", "var_y", "var_py"]].copy(deep=True)
    table["source_bpm"] = source_name
    table["x"] = tracked.x
    table["px"] = tracked.px
    table["y"] = tracked.y
    table["py"] = tracked.py
    table["t"] = tracked.t
    table["pt"] = tracked.pt
    return table.sort_values("turn").reset_index(drop=True)


def _transport_marker_state_to_bpm(
    state: ACDipoleStateSeries,
    model: ACDipoleMadDriver,
    *,
    bpm_name: str,
    marker_name: str,
    direction: int,
) -> ACDipoleStateSeries:
    """Transport a marker-side state series to a BPM location.

    Args:
        state: Phase-space state series at the marker. Must have ``t`` and
            ``pt`` set (always the case for tracked states).
        model: MAD-NG driver used for particle tracking.
        bpm_name: Name of the target BPM element.
        marker_name: Name of the AC-dipole marker element.
        direction: ``+1`` for forward (marker → downstream BPM), ``-1`` for
            backward.

    Returns:
        An :class:`ACDipoleStateSeries` at the BPM location.
    """
    marker_state = np.column_stack([state.x, state.px, state.y, state.py, state.t, state.pt])
    s = model.track_particles(marker_name, bpm_name, marker_state, direction=direction)
    return ACDipoleStateSeries(s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5])


def reconstruct_bpm_momentum_from_cleaned_acd(
    tws: pd.DataFrame,
    model: ACDipoleMadDriver,
    *,
    bpm_name: str,
    cleaned_acd: ACDipoleStateEstimate,
    marker_name: str,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-transport cleaned ACD marker states to a BPM and restore reference momenta.

    Args:
        tws: Twiss DataFrame (used to look up closed-orbit reference momenta).
        model: MAD-NG driver used for particle tracking.
        bpm_name: Name of the target BPM element.
        cleaned_acd: Cleaned state estimate at the AC-dipole marker.
        marker_name: Name of the AC-dipole marker element.
        direction: ``+1`` for forward, ``-1`` for backward.

    Returns:
        ``(px_cleaned, py_cleaned)`` at the BPM location with closed-orbit
        reference momenta restored.
    """
    bpm_state = _transport_marker_state_to_bpm(
        cleaned_acd.state, model, bpm_name=bpm_name, marker_name=marker_name, direction=direction
    )
    return (
        _restore_reference_momenta(bpm_state.px, tws, bpm_name, "px"),
        _restore_reference_momenta(bpm_state.py, tws, bpm_name, "py"),
    )
