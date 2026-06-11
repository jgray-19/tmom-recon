"""BPM selection utilities for AC-dipole reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tmom_recon.lattice.names import resolve_name

from .models import ACDipoleBPMSelection, ACDipoleBPMWindow

__all__ = ["resolve_name"]


def _collect_bpms_from_marker(
    names: list[str],
    marker_name: str,
    bpm_set: set[str],
    *,
    step: int,
    count: int,
) -> list[str]:
    """Collect BPMs by walking away from the marker in one direction.

    Starting immediately adjacent to *marker_name* and stepping by *step*
    (``-1`` for upstream, ``+1`` for downstream), collects up to *count*
    distinct BPMs from *bpm_set* in lattice order.

    Args:
        names: Ordered list of all lattice element names.
        marker_name: Name of the AC-dipole marker element.
        bpm_set: Set of available BPM names to select from.
        step: ``+1`` to walk forward (downstream), ``-1`` to walk backward (upstream).
        count: Maximum number of BPMs to collect.

    Returns:
        Ordered list of selected BPM names (closest first).
    """
    marker_idx = names.index(marker_name)
    selected: list[str] = []
    for shift in range(1, len(names) + 1):
        candidate = names[(marker_idx + step * shift) % len(names)]
        if candidate in bpm_set and candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def _collect_bpms_from_anchor(
    names: list[str],
    anchor_name: str,
    bpm_set: set[str],
    *,
    step: int,
    count: int,
) -> list[str]:
    """Collect BPMs by walking away from an anchor BPM in one direction.

    Unlike :func:`_collect_bpms_from_marker`, the anchor itself is included
    as the first candidate. If *count* BPMs cannot be found starting from
    *anchor_name*, the walk continues around the ring.

    Args:
        names: Ordered list of all lattice element names.
        anchor_name: Name of the anchor BPM (included as a candidate).
        bpm_set: Set of available BPM names to select from.
        step: ``+1`` to walk forward, ``-1`` to walk backward.
        count: Number of BPMs to collect.

    Returns:
        Ordered list of selected BPM names (anchor first).
    """
    anchor_idx = names.index(anchor_name)
    selected: list[str] = []
    for shift in range(count):
        candidate = names[(anchor_idx + step * shift) % len(names)]
        if candidate in bpm_set and candidate not in selected:
            selected.append(candidate)
    if len(selected) != count:
        for shift in range(count, len(names) + count):
            candidate = names[(anchor_idx + step * shift) % len(names)]
            if candidate in bpm_set and candidate not in selected:
                selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def select_ac_dipole_bpm_window(
    tws: pd.DataFrame,
    ac_dipole_marker: str,
    bpm_names: list[str] | np.ndarray | None = None,
    *,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
) -> ACDipoleBPMWindow:
    """Select the upstream and downstream BPMs bracketing the AC-dipole marker.

    When *bpm_upstream* / *bpm_downstream* are omitted, the immediately
    adjacent BPM on each side of the marker is selected automatically.
    When an anchor is supplied the anchor BPM itself (and possibly additional
    BPMs walking away from it) forms the window on that side.

    Args:
        tws: Twiss DataFrame whose index gives the ordered lattice element names.
        ac_dipole_marker: Name of the AC-dipole marker element (case-insensitive
            match against the lattice).
        bpm_names: Optional subset of BPM names to consider. If ``None``,
            all elements whose name starts with ``"BPM"`` (case-insensitive)
            are used.
        bpm_upstream: Optional explicit upstream anchor BPM name.
        bpm_downstream: Optional explicit downstream anchor BPM name.

    Returns:
        An :class:`ACDipoleBPMWindow` with one BPM on each side.

    Raises:
        ValueError: If no BPMs are available, or if BPMs cannot be found on
            both sides of the marker.
    """
    names = [str(name) for name in tws.index]
    marker_name = resolve_name(ac_dipole_marker, names)

    if bpm_names is None:
        bpm_set = {name for name in names if name.upper().startswith("BPM")}
    else:
        bpm_set = {resolve_name(str(name), names) for name in bpm_names}

    if not bpm_set:
        raise ValueError("No BPM names are available for AC-dipole reconstruction")

    resolved_upstream = None if bpm_upstream is None else resolve_name(bpm_upstream, bpm_set)
    resolved_downstream = None if bpm_downstream is None else resolve_name(bpm_downstream, bpm_set)

    if resolved_upstream is None:
        upstream = _collect_bpms_from_marker(names, marker_name, bpm_set, step=-1, count=1)
    else:
        upstream = _collect_bpms_from_anchor(names, resolved_upstream, bpm_set, step=-1, count=1)

    if resolved_downstream is None:
        downstream = _collect_bpms_from_marker(names, marker_name, bpm_set, step=1, count=1)
    else:
        downstream = _collect_bpms_from_anchor(names, resolved_downstream, bpm_set, step=1, count=1)

    if len(upstream) != 1 or len(downstream) != 1:
        raise ValueError(f"Unable to find BPMs on both sides of marker {marker_name!r}")

    return ACDipoleBPMWindow(upstream=tuple(upstream), downstream=tuple(downstream))


def select_ac_dipole_bpms(
    tws: pd.DataFrame,
    ac_dipole_marker: str,
    bpm_names: list[str] | np.ndarray | None = None,
    *,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
) -> ACDipoleBPMSelection:
    """Select the single closest upstream and downstream BPMs for the AC dipole.

    Convenience wrapper around :func:`select_ac_dipole_bpm_window` that
    returns only the primary BPM pair.

    Args:
        tws: Twiss DataFrame whose index gives the ordered lattice element names.
        ac_dipole_marker: Name of the AC-dipole marker element.
        bpm_names: Optional subset of BPM names to consider.
        bpm_upstream: Optional explicit upstream anchor BPM name.
        bpm_downstream: Optional explicit downstream anchor BPM name.

    Returns:
        An :class:`ACDipoleBPMSelection` with one BPM on each side.
    """
    return select_ac_dipole_bpm_window(
        tws,
        ac_dipole_marker,
        bpm_names,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
    ).primary
