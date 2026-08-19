"""Explicit Xsuite-to-MAD-NG compatibility contracts for PSB."""

from __future__ import annotations

import numpy as np
import pytest

from tests.support.truth import xsuite_to_ngtws

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.crosscode, pytest.mark.slow]


def _assert_optics_agree(xsuite_tws, madng_tws, coordinates: tuple[str, ...], tolerance: float) -> None:
    common = xsuite_tws.index.intersection(madng_tws.index)
    assert len(common) > 4
    for coordinate in coordinates:
        difference = np.abs(
            xsuite_tws.loc[common, coordinate].to_numpy(float)
            - madng_tws.loc[common, coordinate].to_numpy(float)
        )
        worst = int(np.argmax(difference))
        bpm = common[worst]
        xsuite_value = float(xsuite_tws.loc[bpm, coordinate])
        madng_value = float(madng_tws.loc[bpm, coordinate])
        assert difference[worst] < tolerance, (
            f"{coordinate} mismatch at {bpm}: Xsuite={xsuite_value:.6e}, "
            f"MAD-NG={madng_value:.6e}, abs={difference[worst]:.3e}, "
            f"relative={difference[worst] / max(abs(madng_value), 1e-30):.3e}"
        )


@pytest.mark.parametrize("delta_p", [0.0, 1e-3], ids=["on_momentum", "off_momentum"])
def test_psb_xsuite_madng_optics_agreement(delta_p, psb_tracking_setup):
    """The generated PSB model agrees across the tracking and reconstruction codes."""
    scenario = psb_tracking_setup(delta_p)
    xsuite_tws = xsuite_to_ngtws(scenario.machine.xsuite_twiss, bpm_names=scenario.measurement.bpm_names)
    madng_tws = scenario.machine.madng_twiss
    _assert_optics_agree(xsuite_tws, madng_tws, ("x", "px", "y", "py"), 1e-5)
    _assert_optics_agree(
        xsuite_tws,
        madng_tws,
        ("beta11", "beta22", "alfa11", "alfa22", "dx", "dpx"),
        1e-3,
    )

    madng_qx = madng_tws.headers.get("q1", madng_tws.headers.get("Q1"))
    madng_qy = madng_tws.headers.get("q2", madng_tws.headers.get("Q2"))
    assert madng_qx is not None and madng_qy is not None
    assert scenario.machine.xsuite_twiss.qx == pytest.approx(madng_qx, abs=1e-3)
    assert scenario.machine.xsuite_twiss.qy == pytest.approx(madng_qy, abs=1e-3)
