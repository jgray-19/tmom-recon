"""Unit tests for tmom_recon.lattice.transport."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.lattice.transport import (
    PlaneTransportMatrix,
    solve_kick_4d_least_squares,
    solve_kick_from_positions,
    transport_matrix_4d_from_twiss,
    transport_matrix_from_twiss,
)


@pytest.fixture()
def simple_twiss() -> pd.DataFrame:
    """Two-element Twiss table at a quarter-wave waist in both planes.

    Both elements sit at beta=1.0, alpha=0.0.  The phase advance from
    ``"kicker"`` to ``"BPM1"`` is exactly 0.25 fractional tune units (π/2
    radians) in both planes, producing the Courant-Snyder matrix

    .. code-block:: text

        R = [[0, 1], [-1, 0]]

    which corresponds to R11=0, R12=β₀=1, R21=-1/β₀=-1, R22=0.
    """
    data = {
        "beta11": [1.0, 1.0],
        "alfa11": [0.0, 0.0],
        "mu1": [0.0, 0.25],
        "beta22": [1.0, 1.0],
        "alfa22": [0.0, 0.0],
        "mu2": [0.0, 0.25],
        "s": [0.0, 10.0],
    }
    index = pd.Index(["kicker", "BPM1"], name="name")
    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# transport_matrix_from_twiss
# ---------------------------------------------------------------------------


def test_quarter_wave_waist_horizontal(simple_twiss: pd.DataFrame) -> None:
    """At a quarter-wave waist with beta=1 and alpha=0, the 2x2 matrix is the
    symplectic rotation [[0, 1], [-1, 0]]."""
    m = transport_matrix_from_twiss(simple_twiss, source="kicker", target="BPM1", plane="x")
    assert isinstance(m, PlaneTransportMatrix)
    assert abs(m.r11) < 1e-12
    assert abs(m.r12 - 1.0) < 1e-12
    assert abs(m.r21 + 1.0) < 1e-12
    assert abs(m.r22) < 1e-12


def test_quarter_wave_waist_vertical(simple_twiss: pd.DataFrame) -> None:
    """The vertical plane should give the same result as horizontal for the
    symmetric quarter-wave fixture."""
    m = transport_matrix_from_twiss(simple_twiss, source="kicker", target="BPM1", plane="y")
    assert abs(m.r11) < 1e-12
    assert abs(m.r12 - 1.0) < 1e-12
    assert abs(m.r21 + 1.0) < 1e-12
    assert abs(m.r22) < 1e-12


@pytest.mark.parametrize(
    "beta0,alpha0,beta1,alpha1,delta_mu_frac",
    [
        (1.0, 0.0, 1.0, 0.0, 0.25),
        (2.5, 0.5, 1.8, -0.3, 0.13),
        (10.0, 1.2, 4.0, 0.8, 0.37),
    ],
)
def test_symplecticity(
    beta0: float, alpha0: float, beta1: float, alpha1: float, delta_mu_frac: float
) -> None:
    """det(R) = 1 for any valid Courant-Snyder parameters (symplectic map)."""
    data = {
        "beta11": [beta0, beta1],
        "alfa11": [alpha0, alpha1],
        "mu1": [0.0, delta_mu_frac],
        "beta22": [beta0, beta1],
        "alfa22": [alpha0, alpha1],
        "mu2": [0.0, delta_mu_frac],
        "s": [0.0, 10.0],
    }
    twiss = pd.DataFrame(data, index=pd.Index(["src", "tgt"], name="name"))
    m = transport_matrix_from_twiss(twiss, source="src", target="tgt", plane="x")
    det = m.r11 * m.r22 - m.r12 * m.r21
    assert abs(det - 1.0) < 1e-10, f"det(R) = {det} (expected 1)"


def test_unknown_plane_raises(simple_twiss: pd.DataFrame) -> None:
    """Passing an unsupported plane string must raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported plane"):
        transport_matrix_from_twiss(simple_twiss, source="kicker", target="BPM1", plane="z")


def test_missing_source_element_raises(simple_twiss: pd.DataFrame) -> None:
    """Referencing an element not in the Twiss index must raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        transport_matrix_from_twiss(simple_twiss, source="NOSUCH", target="BPM1", plane="x")


def test_missing_target_element_raises(simple_twiss: pd.DataFrame) -> None:
    """Referencing a target element not in the Twiss index must raise KeyError."""
    with pytest.raises(KeyError, match="not found"):
        transport_matrix_from_twiss(simple_twiss, source="kicker", target="NOSUCH", plane="x")


# ---------------------------------------------------------------------------
# transport_matrix_4d_from_twiss
# ---------------------------------------------------------------------------


def test_4d_matrix_block_diagonal(simple_twiss: pd.DataFrame) -> None:
    """The assembled 4x4 matrix must have zero off-diagonal 2x2 blocks."""
    mat = transport_matrix_4d_from_twiss(simple_twiss, source="kicker", target="BPM1")
    assert mat.shape == (4, 4)
    # Top-right block (rows 0-1, cols 2-3) must be zero
    assert np.allclose(mat[:2, 2:], 0.0)
    # Bottom-left block (rows 2-3, cols 0-1) must be zero
    assert np.allclose(mat[2:, :2], 0.0)


def test_4d_matrix_diagonals_match_planes(simple_twiss: pd.DataFrame) -> None:
    """The top-left and bottom-right 2x2 blocks must match the individual
    plane matrices returned by transport_matrix_from_twiss."""
    mat = transport_matrix_4d_from_twiss(simple_twiss, source="kicker", target="BPM1")
    mx = transport_matrix_from_twiss(simple_twiss, source="kicker", target="BPM1", plane="x")
    my = transport_matrix_from_twiss(simple_twiss, source="kicker", target="BPM1", plane="y")

    assert abs(mat[0, 0] - mx.r11) < 1e-15
    assert abs(mat[0, 1] - mx.r12) < 1e-15
    assert abs(mat[1, 0] - mx.r21) < 1e-15
    assert abs(mat[1, 1] - mx.r22) < 1e-15

    assert abs(mat[2, 2] - my.r11) < 1e-15
    assert abs(mat[2, 3] - my.r12) < 1e-15
    assert abs(mat[3, 2] - my.r21) < 1e-15
    assert abs(mat[3, 3] - my.r22) < 1e-15


# ---------------------------------------------------------------------------
# solve_kick_from_positions
# ---------------------------------------------------------------------------


def test_solve_kick_horizontal_only(simple_twiss: pd.DataFrame) -> None:
    """A purely horizontal kick (y_source=y_target=0) gives py=0 exactly and
    px equal to x_target/R12 (since R11=0 at quarter-wave waist)."""
    px_true = 3.7e-6
    x_target = 1.0 * px_true  # R12 = 1 at quarter-wave waist
    px, py = solve_kick_from_positions(
        simple_twiss,
        source="kicker",
        target="BPM1",
        x_source=0.0,
        y_source=0.0,
        x_target=x_target,
        y_target=0.0,
    )
    assert abs(px - px_true) < 1e-15
    assert py == 0.0


def test_solve_kick_singular_raises() -> None:
    """When R12=0 (delta_mu=0, i.e. source==target), the solve must raise."""
    data = {
        "beta11": [1.0, 1.0],
        "alfa11": [0.0, 0.0],
        "mu1": [0.0, 0.0],
        "beta22": [1.0, 1.0],
        "alfa22": [0.0, 0.0],
        "mu2": [0.0, 0.0],
        "s": [0.0, 0.0],
    }
    twiss = pd.DataFrame(data, index=pd.Index(["A", "B"], name="name"))
    with pytest.raises(ValueError, match="singular"):
        solve_kick_from_positions(
            twiss, source="A", target="B", x_source=0.0, y_source=0.0, x_target=1.0, y_target=0.0
        )


# ---------------------------------------------------------------------------
# solve_kick_4d_least_squares
# ---------------------------------------------------------------------------


def test_4d_least_squares_uncoupled_exact(simple_twiss: pd.DataFrame) -> None:
    """Single-BPM block-diagonal matrix: exact recovery of (px, py)."""
    mat = transport_matrix_4d_from_twiss(simple_twiss, source="kicker", target="BPM1")

    px_true = 5.0e-6
    py_true = -2.3e-6
    state = mat @ np.array([0.0, px_true, 0.0, py_true])
    x_meas = {"BPM1": float(state[0])}
    y_meas = {"BPM1": float(state[2])}

    px_rec, py_rec = solve_kick_4d_least_squares({"BPM1": mat}, x_meas, y_meas)
    assert abs(px_rec - px_true) < 1e-14
    assert abs(py_rec - py_true) < 1e-14


def test_4d_least_squares_coupled_matrix() -> None:
    """Non-trivial coupled 4x4 matrix: solver recovers (px, py) from synthetic
    measurements constructed by applying the matrix to a known kick state."""
    m = np.eye(4)
    m[0, 1] = 2.1
    m[0, 3] = 0.4
    m[2, 1] = 0.3
    m[2, 3] = 1.8

    px_true = 1.2e-5
    py_true = -3.4e-6
    state = m @ np.array([0.0, px_true, 0.0, py_true])
    x_meas = {"BPM_C": float(state[0])}
    y_meas = {"BPM_C": float(state[2])}

    px_rec, py_rec = solve_kick_4d_least_squares({"BPM_C": m}, x_meas, y_meas)
    assert abs(px_rec - px_true) < 1e-13
    assert abs(py_rec - py_true) < 1e-13
