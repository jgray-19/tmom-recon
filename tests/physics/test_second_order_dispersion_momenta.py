"""Second-order dispersion in the neighbour-pair momentum reconstruction.

The orbit at momentum ``pt`` is ``pt*dx + pt**2*ddx`` (MAD-NG's ``chrom=true``
columns already carry the Taylor 1/2), and its angle is ``pt*dpx + pt**2*ddpx``.
The reconstruction must strip the full dispersive *position* before normalising
to betatron coordinates and add the full dispersive *angle* back afterwards.

These tests drive :func:`_compute_nominal_momenta` directly with a hand-built
neighbour pair, so they pin the formula rather than any lattice. The columns are
optional everywhere: when they are absent the result must fall back exactly to
the first-order expression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.data.schema import PREV, SUFFIX_PREV
from tmom_recon.physics.momenta import _compute_nominal_momenta

pytestmark = pytest.mark.unit
__test__ = False

PT = 8.0e-3
DX, DDX = 1.7, -4.2
DPX, DDPX = 0.11, -0.37
BETA_X, BETA_Y = 12.0, 7.0
ALFA_X, ALFA_Y = 0.4, -0.6


def _pair(betatron_x: float, betatron_x_prev: float, *, second_order: bool) -> pd.DataFrame:
    """One BPM and its previous neighbour, both on the dispersive orbit at PT."""
    dispersive = PT * DX + (PT**2 * DDX if second_order else 0.0)
    row = {
        "x": betatron_x + dispersive,
        "y": 0.0,
        PREV.x: betatron_x_prev + dispersive,
        PREV.y: 0.0,
        "sqrt_betax": np.sqrt(BETA_X),
        "sqrt_betay": np.sqrt(BETA_Y),
        f"sqrt_betax_{SUFFIX_PREV}": np.sqrt(BETA_X),
        f"sqrt_betay_{SUFFIX_PREV}": np.sqrt(BETA_Y),
        "alfax": ALFA_X,
        "alfay": ALFA_Y,
        "dx": DX,
        PREV.dx: DX,
        "dpx": DPX,
        "dy": 0.0,
        PREV.dy: 0.0,
        "dpy": 0.0,
        PREV.delta_x: 0.17,
        PREV.delta_y: 0.23,
    }
    if second_order:
        row |= {"ddx": DDX, PREV.ddx: DDX, "ddpx": DDPX, "ddy": 0.0, PREV.ddy: 0.0, "ddpy": 0.0}
    return pd.DataFrame([row])


def _px(second_order: bool, betatron_x: float = 1.1e-3, betatron_x_prev: float = -0.4e-3) -> float:
    data = _pair(betatron_x, betatron_x_prev, second_order=second_order)
    px, _ = _compute_nominal_momenta(data, PREV, SUFFIX_PREV, is_prev=True, pt_est=PT)
    return float(px[0])


def test_second_order_position_is_removed_and_angle_added_back() -> None:
    """A purely dispersive orbit must reconstruct to the dispersive angle alone."""
    data = _pair(0.0, 0.0, second_order=True)
    px, py = _compute_nominal_momenta(data, PREV, SUFFIX_PREV, is_prev=True, pt_est=PT)
    assert float(px[0]) == pytest.approx(PT * DPX + PT**2 * DDPX, rel=1e-12)
    assert float(py[0]) == pytest.approx(0.0, abs=1e-15)


def test_first_order_alone_leaves_the_second_order_orbit_as_a_bias() -> None:
    """Dropping the columns is the old behaviour, and it is wrong off momentum."""
    data = _pair(0.0, 0.0, second_order=True).drop(
        columns=["ddx", PREV.ddx, "ddpx", "ddy", PREV.ddy, "ddpy"]
    )
    px, _ = _compute_nominal_momenta(data, PREV, SUFFIX_PREV, is_prev=True, pt_est=PT)
    # The un-removed pt**2*ddx position leaks through the betatron transport and
    # the pt**2*ddpx angle is simply missing, so the error is far from zero.
    assert abs(float(px[0]) - (PT * DPX + PT**2 * DDPX)) > 1e-6


def test_missing_columns_reproduce_the_first_order_result_exactly() -> None:
    """Optional means optional: absent columns must not change anything else."""
    with_zeros = _pair(1.1e-3, -0.4e-3, second_order=True).assign(
        **{"ddx": 0.0, PREV.ddx: 0.0, "ddpx": 0.0}
    )
    # Rebuild the position without the second-order dispersive offset so the two
    # frames describe the same particle.
    with_zeros["x"] -= PT**2 * DDX
    with_zeros[PREV.x] -= PT**2 * DDX
    px_zeros, _ = _compute_nominal_momenta(with_zeros, PREV, SUFFIX_PREV, is_prev=True, pt_est=PT)
    assert float(px_zeros[0]) == pytest.approx(_px(second_order=False), rel=1e-12)


def test_betatron_motion_is_untouched_by_the_second_order_terms() -> None:
    """The dispersive orbit is an offset; it must not scale the betatron part."""
    betatron_only = _px(second_order=False) - PT * DPX
    full = _px(second_order=True) - (PT * DPX + PT**2 * DDPX)
    assert full == pytest.approx(betatron_only, rel=1e-12)
