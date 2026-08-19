"""Explicit Xsuite-to-MAD-NG compatibility contracts for PSB."""

from __future__ import annotations

import numpy as np
import pytest

from tests.support.truth import xsuite_to_ngtws

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.crosscode]


@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_psb_xsuite_madng_optics_agreement(delta_p, psb_tracking_setup):
    """The generated PSB model agrees across the tracking and reconstruction codes."""
    scenario = psb_tracking_setup(delta_p)
    xsuite_tws = xsuite_to_ngtws(scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names)
    madng_tws = scenario.machine.madng_twiss
    common = xsuite_tws.index.intersection(madng_tws.index)
    assert len(common) > 4

    for coordinate in ("x", "px", "y", "py"):
        difference = np.abs(
            xsuite_tws.loc[common, coordinate].to_numpy(float)
            - madng_tws.loc[common, coordinate].to_numpy(float)
        )
        assert difference.max() < 1e-5, (
            f"{coordinate} Xsuite/MAD-NG mismatch: {difference.max():.3e}"
        )

    for coordinate in ("beta11", "beta22", "alfa11", "alfa22", "dx", "dpx"):
        difference = np.abs(
            xsuite_tws.loc[common, coordinate].to_numpy(float)
            - madng_tws.loc[common, coordinate].to_numpy(float)
        )
        assert difference.max() < 1e-3, (
            f"{coordinate} Xsuite/MAD-NG mismatch: {difference.max():.3e}"
        )

    madng_qx = madng_tws.headers.get("q1", madng_tws.headers.get("Q1"))
    madng_qy = madng_tws.headers.get("q2", madng_tws.headers.get("Q2"))
    assert madng_qx is not None and madng_qy is not None
    assert scenario.machine.xsuite_twiss.qx == pytest.approx(madng_qx, abs=1e-3)
    assert scenario.machine.xsuite_twiss.qy == pytest.approx(madng_qy, abs=1e-3)
