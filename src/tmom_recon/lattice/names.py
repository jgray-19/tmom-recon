"""Lattice element name resolution utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def resolve_name(name: str, candidates: list[str] | set[str]) -> str:
    """Resolve *name* to a canonical element name from *candidates*.

    Performs an exact match first, then falls back to a case-insensitive match.

    Args:
        name: The element name to resolve (may use non-canonical casing).
        candidates: Sequence or set of canonical element names.

    Returns:
        The matching canonical name.

    Raises:
        KeyError: If *name* cannot be matched (even case-insensitively).
    """
    if name in candidates:
        return name
    lowered = {str(candidate).lower(): str(candidate) for candidate in candidates}
    resolved = lowered.get(name.lower())
    if resolved is None:
        raise KeyError(f"Element {name!r} was not found")
    return resolved


def normalise_measurement_names(data: pd.DataFrame, tws_names: list[str]) -> pd.DataFrame:
    """Resolve measurement BPM names against the lattice name list.

    Args:
        data: Measurement DataFrame with a ``"name"`` column.
        tws_names: Canonical lattice element names.

    Returns:
        A copy of *data* with ``"name"`` mapped to canonical lattice names.
    """
    data = data.copy(deep=True)
    raw_name_map = {str(name): resolve_name(str(name), tws_names) for name in data["name"].unique()}
    data["name"] = data["name"].astype(str).map(raw_name_map)
    return data


def normalise_twiss_index(tws: pd.DataFrame, lattice_names: list[str]) -> pd.DataFrame:
    """Re-index a Twiss DataFrame using canonical lattice names.

    Args:
        tws: Twiss DataFrame whose index may use non-canonical casing.
        lattice_names: Canonical lattice element names.

    Returns:
        A copy of *tws* with its index replaced by canonical names.
    """
    normalised = tws.copy(deep=True)
    normalised.index = [resolve_name(str(n), lattice_names) for n in normalised.index]
    return normalised
