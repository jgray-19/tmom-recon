"""Generated PSB model infrastructure contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tests.psb_tracking import ACD_ELEMENT
from tests.support.scenarios import psb_clean, psb_offmomentum
from tests.support.truth import xsuite_to_ngtws

pytestmark = [pytest.mark.psb, pytest.mark.integration]


def _scenario(psb_model_dir, delta_p: float = 0.0):
    return psb_clean(psb_model_dir) if delta_p == 0.0 else psb_offmomentum(psb_model_dir, delta_p)


def test_generated_model_has_expected_bpm_set_and_acd_anchor(psb_model_dir) -> None:
    scenario = _scenario(psb_model_dir)
    model_twiss = scenario.machine.madng_twiss
    tracked_names = {str(name).upper() for name in scenario.measurement.data["name"].unique()}
    bpm_names = [
        str(name)
        for name in model_twiss.index
        if "BPM" in str(name).upper() and str(name).upper() in tracked_names
    ]

    assert len(bpm_names) >= 8
    assert ACD_ELEMENT.upper() in {
        str(name).upper() for name in scenario.machine.madng_model.twiss_elements.index
    }


def test_generated_model_nominal_orbit_is_closed(psb_model_dir) -> None:
    scenario = _scenario(psb_model_dir)
    twiss = scenario.machine.madng_twiss
    assert float(twiss["x"].abs().max()) < 1e-5
    assert float(twiss["y"].abs().max()) < 1e-5


def test_xsuite_and_madng_on_momentum_optics_agree(psb_model_dir) -> None:
    scenario = _scenario(psb_model_dir)
    xsuite = xsuite_to_ngtws(
        scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names
    )
    madng = scenario.machine.madng_twiss
    common = xsuite.index.intersection(madng.index)
    assert len(common) >= 8

    for coordinate in ("x", "px", "y", "py"):
        difference = np.abs(
            xsuite.loc[common, coordinate].to_numpy(float)
            - madng.loc[common, coordinate].to_numpy(float)
        )
        assert difference.max() < 1e-5, (
            f"{coordinate} Xsuite/MAD-NG mismatch: {difference.max():.3e}"
        )


def test_xsuite_and_madng_off_momentum_optics_agree(psb_model_dir) -> None:
    scenario = _scenario(psb_model_dir, delta_p=1e-3)
    xsuite = xsuite_to_ngtws(
        scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names
    )
    madng = scenario.machine.madng_twiss
    common = xsuite.index.intersection(madng.index)
    assert len(common) >= 8

    for coordinate in ("beta11", "beta22", "alfa11", "alfa22", "dx", "dpx"):
        difference = np.abs(
            xsuite.loc[common, coordinate].to_numpy(float)
            - madng.loc[common, coordinate].to_numpy(float)
        )
        assert difference.max() < 1e-3, (
            f"{coordinate} off-momentum Xsuite/MAD-NG mismatch: {difference.max():.3e}"
        )
