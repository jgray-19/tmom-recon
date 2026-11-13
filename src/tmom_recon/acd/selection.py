"""BPM selection utilities for AC-dipole reconstruction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tmom_recon.lattice.names import resolve_name

from .models import ACDipoleBPMSelection

__all__ = ["resolve_name"]


def _bpm_from_marker(names: list[str], marker_name: str, bpm_set: set[str], *, step: int) -> str:
    """Return the BPM nearest *marker_name*, walking by *step* (-1 upstream, +1 downstream)."""
    marker_idx = names.index(marker_name)
    for shift in range(1, len(names) + 1):
        candidate = names[(marker_idx + step * shift) % len(names)]
        if candidate in bpm_set:
            return candidate
    raise ValueError(f"Unable to find a BPM next to marker {marker_name!r}")


def select_ac_dipole_bpms(
    tws: pd.DataFrame,
    ac_dipole_marker: str,
    bpm_names: list[str] | np.ndarray | None = None,
    *,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
) -> ACDipoleBPMSelection:
    """Select the upstream and downstream BPMs bracketing the AC-dipole marker.

    When *bpm_upstream* / *bpm_downstream* are omitted, the immediately
    adjacent BPM on each side of the marker is selected automatically; an
    explicit BPM is used as given.

    Args:
        tws: Twiss DataFrame whose index gives the ordered lattice element names.
        ac_dipole_marker: Name of the AC-dipole marker element (case-insensitive
            match against the lattice).
        bpm_names: Optional subset of BPM names to consider. If ``None``,
            all elements whose name starts with ``"BPM"`` (case-insensitive)
            are used.
        bpm_upstream: Optional explicit upstream BPM name.
        bpm_downstream: Optional explicit downstream BPM name.

    Returns:
        An :class:`ACDipoleBPMSelection` with one BPM on each side.

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

    upstream = (
        _bpm_from_marker(names, marker_name, bpm_set, step=-1)
        if bpm_upstream is None
        else resolve_name(bpm_upstream, bpm_set)
    )
    downstream = (
        _bpm_from_marker(names, marker_name, bpm_set, step=1)
        if bpm_downstream is None
        else resolve_name(bpm_downstream, bpm_set)
    )
    return ACDipoleBPMSelection(upstream=upstream, downstream=downstream)
