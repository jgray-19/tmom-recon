"""BPM list utilities for lattice-indexed tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def find_common_bpms(*dataframes: pd.DataFrame) -> list[str]:
    """Return BPM names common to all tables, preserving the first table order."""
    if not dataframes:
        return []

    common = set(dataframes[0].index)
    for dataframe in dataframes[1:]:
        common &= set(dataframe.index)

    return [str(bpm) for bpm in dataframes[0].index if bpm in common]
