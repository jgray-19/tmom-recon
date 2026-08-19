from __future__ import annotations

import pandas as pd
import pytest

from tmom_recon.lattice.bpms import find_common_bpms

pytestmark = pytest.mark.unit


def test_find_common_bpms_preserves_first_table_order() -> None:
    first = pd.DataFrame(index=["BPM.3", "BPM.1", "BPM.2", "BPM.4"])
    second = pd.DataFrame(index=["BPM.2", "BPM.3", "BPM.1"])
    third = pd.DataFrame(index=["BPM.1", "BPM.3"])

    assert find_common_bpms(first, second, third) == ["BPM.3", "BPM.1"]


def test_find_common_bpms_without_tables_returns_empty_list() -> None:
    assert find_common_bpms() == []
