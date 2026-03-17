from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ACDipoleBPMSelection, ACDipoleBPMWindow


def resolve_name(name: str, candidates: list[str] | set[str]) -> str:
    if name in candidates:
        return name

    lowered = {str(candidate).lower(): str(candidate) for candidate in candidates}
    resolved = lowered.get(name.lower())
    if resolved is None:
        raise KeyError(f"Element {name!r} was not found")
    return resolved


def _collect_bpms_from_marker(
    names: list[str],
    marker_name: str,
    bpm_set: set[str],
    *,
    step: int,
    count: int,
) -> list[str]:
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
    n_bpms_each_side: int = 1,
) -> ACDipoleBPMWindow:
    if n_bpms_each_side < 1:
        raise ValueError(f"n_bpms_each_side must be >= 1, got {n_bpms_each_side}")

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
        upstream = _collect_bpms_from_marker(
            names,
            marker_name,
            bpm_set,
            step=-1,
            count=n_bpms_each_side,
        )
    else:
        upstream = _collect_bpms_from_anchor(
            names,
            resolved_upstream,
            bpm_set,
            step=-1,
            count=n_bpms_each_side,
        )

    if resolved_downstream is None:
        downstream = _collect_bpms_from_marker(
            names,
            marker_name,
            bpm_set,
            step=1,
            count=n_bpms_each_side,
        )
    else:
        downstream = _collect_bpms_from_anchor(
            names,
            resolved_downstream,
            bpm_set,
            step=1,
            count=n_bpms_each_side,
        )

    if len(upstream) != n_bpms_each_side or len(downstream) != n_bpms_each_side:
        raise ValueError(
            f"Unable to find {n_bpms_each_side} BPMs on each side of marker {marker_name!r}"
        )

    return ACDipoleBPMWindow(
        upstream=tuple(upstream),
        downstream=tuple(downstream),
    )


def select_ac_dipole_bpms(
    tws: pd.DataFrame,
    ac_dipole_marker: str,
    bpm_names: list[str] | np.ndarray | None = None,
    *,
    bpm_upstream: str | None = None,
    bpm_downstream: str | None = None,
) -> ACDipoleBPMSelection:
    return select_ac_dipole_bpm_window(
        tws,
        ac_dipole_marker,
        bpm_names,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        n_bpms_each_side=1,
    ).primary
