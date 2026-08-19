"""Small PSB off-momentum reconstruction regressions."""

from __future__ import annotations

import pytest

from tests.support.scenarios import psb_offmomentum

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.regression]


def test_offmomentum_reference_is_not_empty(psb_model_dir) -> None:
    """The off-momentum scenario carries a real tracking truth table for the BT/beamline data."""
    scenario = psb_offmomentum(psb_model_dir, delta_p=1e-3)
    assert len(scenario.measurement.truth) > 0
    assert scenario.measurement.pt > 0.0
