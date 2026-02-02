"""Configuration constants for momentum reconstruction."""

from __future__ import annotations

# Standard deviation of position noise (meters).
POSITION_STD_DEV = 1e-5

# Global schema constant for data files.
FILE_COLUMNS: tuple[str, ...] = (
    "name",
    "turn",
    "x",
    "px",
    "y",
    "py",
    "var_x",
    "var_y",
    "var_px",
    "var_py",
    "kick_plane",
)
