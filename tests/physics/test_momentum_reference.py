"""The reconstruction's momentum input is an offset, and the API enforces it.

The trap this pins is invisible at first order: an absolute ``pt`` and an offset
from the reference differ only in the second-order dispersion term, so on a
linear lattice or an on-momentum reference the two agree exactly. These tests use
a reference deliberately placed off momentum, where they do not.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from tests.reference_co import full_state_reference_from_twiss
from tmom_recon import ModelDetails, MomentumReference, calculate_pz
from tmom_recon.physics.pt_calculation import estimate_pt_from_model

pytestmark = pytest.mark.unit

BPMS = [f"bpm{i}" for i in range(12)]


def _twiss() -> pd.DataFrame:
    phase = np.linspace(0.0, 2.0 * np.pi, len(BPMS), endpoint=False)
    return pd.DataFrame(
        {
            "dx": 2.0 + 0.9 * np.cos(phase),
            "dy": np.zeros(len(BPMS)),
            "ddx": 1.3 + 0.5 * np.sin(phase),
        },
        index=pd.Index(BPMS, name="name"),
    )


def _turn_data(orbit: np.ndarray, turns: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": np.tile(BPMS, turns),
            "turn": np.repeat(np.arange(turns), len(BPMS)),
            "x": np.tile(orbit, turns),
            "y": np.zeros(len(BPMS) * turns),
        }
    )


def test_offset_from_is_the_difference_to_the_reference_momentum() -> None:
    reference = MomentumReference(pd.DataFrame({"x": 0.0}, index=pd.Index(BPMS)), pt=3.0e-3)
    assert reference.offset_from(4.2e-3) == pytest.approx(1.2e-3)
    assert reference.offset_from(3.0e-3) == 0.0


def test_a_reference_without_positions_is_rejected() -> None:
    with pytest.raises(ValueError, match='needs an "x" column'):
        MomentumReference(pd.DataFrame({"px": 0.0}, index=pd.Index(BPMS)))


def test_all_bpm_reconstruction_requires_a_reference() -> None:
    data = pd.DataFrame(
        {
            "name": [BPMS[0]],
            "turn": [0],
            "x": [0.0],
            "y": [0.0],
        }
    )

    with pytest.raises(ValueError, match="needs a `reference`"):
        calculate_pz(
            data,
            ModelDetails(accelerator=cast(Any, None), pt=0.0),
            reference=None,
            barrier_s=None,
        )


def test_all_bpm_reconstruction_requires_an_explicit_acd_barrier_decision() -> None:
    """A caller cannot silently transport a neighbour pair through a local kick."""
    barrier_s = inspect.signature(calculate_pz).parameters["barrier_s"]
    assert barrier_s.default is inspect.Parameter.empty


def test_full_state_reference_preserves_closed_orbit_angles() -> None:
    tws = pd.DataFrame(
        {
            "x": np.linspace(0.0, 1.0, len(BPMS)),
            "px": np.linspace(1.0, 2.0, len(BPMS)),
            "y": np.linspace(2.0, 3.0, len(BPMS)),
            "py": np.linspace(3.0, 4.0, len(BPMS)),
        },
        index=pd.Index(BPMS),
    )

    reference = full_state_reference_from_twiss(tws, pt=2e-3)

    assert list(reference.closed_orbit.columns) == ["x", "px", "y", "py"]
    pd.testing.assert_frame_equal(reference.closed_orbit, tws)
    assert reference.pt == pytest.approx(2e-3)


def test_the_estimated_momentum_is_an_offset_not_an_absolute_pt() -> None:
    """The estimator's output is in the same units the offset API expects."""
    tws = _twiss()
    reference_pt, offset = 3.0e-3, 1.2e-3
    dx, ddx = tws["dx"].to_numpy(), tws["ddx"].to_numpy()

    reference_orbit = reference_pt * dx + reference_pt**2 * ddx
    measurement_pt = reference_pt + offset
    measured_orbit = measurement_pt * dx + measurement_pt**2 * ddx

    reference = MomentumReference(
        pd.DataFrame({"x": reference_orbit}, index=pd.Index(BPMS)), pt=reference_pt
    )
    estimated = estimate_pt_from_model(
        _turn_data(measured_orbit), tws, reference=reference, info=False
    )

    # The estimate is the offset, not the absolute momentum: it is far closer to
    # `offset` than to `measurement_pt`, and closer than the residual second-order
    # gain error (~2*reference_pt*ddx/dx, a few 1e-3 relative) allows any confusion.
    assert estimated == pytest.approx(offset, rel=0.02)
    assert abs(estimated - offset) < abs(estimated - measurement_pt) / 100.0
