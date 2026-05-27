"""Shared helpers for kicker-oriented integration tests."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

KICKER_PATTERNS = (
    r"^mk",
    r"kick",
    r"^mcb",
    r"^mke",
    r"^mki",
    r"\.ksw",
    r"\.dhz",
    r"\.dvt",
)


def build_twiss_for_recon(tws, tkicker_name: str) -> pd.DataFrame:
    """Convert an xtrack Twiss table to the kicker reconstruction format."""
    df = tws.to_pandas()
    df["name"] = df["name"].str.upper()
    df = df.rename(
        columns={
            "betx": "beta11",
            "bety": "beta22",
            "alfx": "alfa11",
            "alfy": "alfa22",
            "mux": "mu1",
            "muy": "mu2",
        }
    )
    df = df.set_index("name")

    kicker_row = df.loc[tkicker_name.upper()].copy()
    kicker_row.name = "kicker"

    bpm_df = df[df.index.str.match(r"(?i)^BPM")]
    result = pd.concat([bpm_df, kicker_row.to_frame().T])
    result.index.name = "name"
    return result


def select_kicker_element(line) -> str | None:
    """Return the first element name that looks like a kicker/corrector."""
    for pattern in KICKER_PATTERNS:
        for name in line.element_names:
            if re.search(pattern, name, flags=re.IGNORECASE):
                return name
    return None


def strip_inline_flags(pattern: str) -> str:
    """Remove a leading inline case-insensitive flag for safe embedding."""
    if pattern.startswith("(?i)"):
        return pattern[4:]
    return pattern


def realign_kicker_turns(
    tracking_df: pd.DataFrame,
    *,
    kicker_name: str,
    logical_turns: int,
) -> pd.DataFrame:
    """Realign turns so each logical turn starts immediately after the kicker."""
    parts: list[pd.DataFrame] = []
    marker_name = kicker_name.upper()

    for _turn, turn_df in tracking_df.groupby("turn", sort=False):
        turn_names = turn_df["name"].str.upper().to_numpy()
        marker_rows = np.flatnonzero(turn_names == marker_name)
        if marker_rows.size == 0:
            parts.append(turn_df)
            continue

        marker_idx = int(marker_rows[0])
        if marker_idx > 0:
            before = turn_df.iloc[:marker_idx].copy()
            before["turn"] = before["turn"] - 1
            parts.append(before)
        parts.append(turn_df.iloc[marker_idx:].copy())

    realigned = pd.concat(parts, ignore_index=True)
    return realigned.loc[(realigned["turn"] >= 1) & (realigned["turn"] <= logical_turns)].copy()
