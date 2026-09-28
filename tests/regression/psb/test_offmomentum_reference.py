"""Small PSB off-momentum reconstruction regressions."""

from __future__ import annotations

import pytest

from tests.support.scenarios import psb_offmomentum

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.regression]


def test_offmomentum_reference_matches_tracked_bpms(psb_model_dir) -> None:
    """Distinguish physical reconstruction BPMs from the BPMT pseudo-monitor."""
    scenario = psb_offmomentum(psb_model_dir, delta_p=1e-3)
    tracked_bpms = {
        str(name).upper()
        for name in scenario.measurement.data["name"].unique()
        if "BPM" in str(name).upper()
    }
    expected = {name.upper() for name in scenario.measurement.bpm_names}
    assert expected <= tracked_bpms
    assert tracked_bpms.difference(expected) == {"BR3.BPMT3L1"}
    assert scenario.measurement.pt > 0.0
