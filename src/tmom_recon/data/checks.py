from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tmom_recon.data.schema import (
    CORE_ID_COLS,
    CORE_POS_COLS,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFeatures:
    has_px: bool
    has_py: bool


def validate_input(df: pd.DataFrame) -> InputFeatures:
    required = set(CORE_ID_COLS + CORE_POS_COLS)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    return InputFeatures(
        has_px=("px" in df.columns),
        has_py=("py" in df.columns),
    )
