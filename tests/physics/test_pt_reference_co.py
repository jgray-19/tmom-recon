"""Momentum estimation happens after subtraction of measured orbit zero."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon import ReconstructionFrame
from tmom_recon.physics.pt_calculation import _solve_pt_quadratic, estimate_pt_from_model

pytestmark = pytest.mark.unit
BPMS = [f"bpm{i}" for i in range(12)]
PT = 4.17e-3


def _twiss(with_ddx=True):
    phase = np.linspace(0, 2 * np.pi, len(BPMS), endpoint=False)
    tws = pd.DataFrame(
        {"dx": 2 + 0.9 * np.cos(phase), "dy": 0.0}, index=pd.Index(BPMS, name="name")
    )
    if with_ddx:
        tws["ddx"] = 1.3 + 0.5 * np.sin(phase)
    return tws


def _data(orbit):
    return pd.DataFrame(
        {
            "name": np.tile(BPMS, 4),
            "turn": np.repeat(np.arange(4), len(BPMS)),
            "x": np.tile(orbit, 4),
            "y": 0.0,
        }
    )


def _frame(orbit):
    origin = pd.DataFrame({"x": orbit, "y": 0.0}, index=pd.Index(BPMS, name="name"))
    return ReconstructionFrame(origin, dynamic_planes=("x", "y"))


def test_dynamic_estimate_uses_orbit_zero_removed_coordinates():
    tws = _twiss()
    error = 2e-3 * tws.dx.to_numpy()
    measured = error + PT * tws.dx.to_numpy() + PT**2 * tws.ddx.to_numpy()
    assert estimate_pt_from_model(
        _data(measured), tws, frame=_frame(error), info=False
    ) == pytest.approx(PT)


def test_per_measurement_zero_erases_momentum_signal():
    tws = _twiss()
    measured = PT * tws.dx.to_numpy() + PT**2 * tws.ddx.to_numpy()
    assert estimate_pt_from_model(
        _data(measured), tws, frame=_frame(measured), info=False
    ) == pytest.approx(0, abs=1e-15)


def test_second_order_beats_first_order():
    tws = _twiss()
    measured = PT * tws.dx.to_numpy() + PT**2 * tws.ddx.to_numpy()
    frame = _frame(np.zeros(len(BPMS)))
    second = estimate_pt_from_model(_data(measured), tws, frame=frame, info=False)
    first = estimate_pt_from_model(
        _data(measured), tws.drop(columns="ddx"), frame=frame, info=False
    )
    assert second == pytest.approx(PT)
    assert abs(second - PT) < abs(first - PT) / 100


def test_quadratic_solver_selects_near_root():
    numerator = 4e-3 * 50 + (4e-3) ** 2 * 30
    assert _solve_pt_quadratic(numerator, 50, 30) == pytest.approx(4e-3)
