"""Unit tests for tmom_recon.kicker.core (no MAD-NG, no xtrack)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tmom_recon.kicker.core import (
    check_data_index,
    find_kick,
    reconstruct_momentum_kick,
    subtract_closed_orbit,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_twiss() -> pd.DataFrame:
    """Two-element Twiss table at a quarter-wave waist in both planes.

    The kicker sits at *s* = 0 and BPM1 at *s* = 10.  The phase advance is
    0.25 fractional tune units (π/2 radians) in both planes, so the
    Courant-Snyder R12 element equals β₀ = 1.0 m in both planes.  With the
    closed orbit subtracted (x_k = y_k = 0), the BPM displacement on the kick
    turn satisfies

    .. code-block:: text

        x_BPM1 = R12_x * px_kick = px_kick
        y_BPM1 = R12_y * py_kick = py_kick

    which makes reconstruction trivially verifiable.
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


def _make_data(
    px_kick: float,
    py_kick: float,
    n_turns_free: int = 10,
    kick_turn: int = 10,
) -> pd.DataFrame:
    """Build a minimal turn-by-turn DataFrame for a single BPM.

    Pre-kick turns have x=0, y=0 (zero closed orbit, zero noise).  On the
    kick turn the displacements equal the kick momenta (since R12=1 at the
    quarter-wave waist).  Post-kick turns are set to zero for simplicity.

    Args:
        px_kick: True horizontal kick momentum [rad].
        py_kick: True vertical kick momentum [rad].
        n_turns_free: Number of pre-kick turns (closed-orbit estimation window).
        kick_turn: Turn number at which the kick occurs.

    Returns:
        DataFrame with columns ``name``, ``turn``, ``x``, ``y`` covering turns
        0 … kick_turn (inclusive), with ``name`` as the index.
    """
    rows = []
    for t in range(kick_turn + 1):
        if t < n_turns_free:
            x, y = 0.0, 0.0
        elif t == kick_turn:
            x, y = px_kick, py_kick
        else:
            x, y = 0.0, 0.0
        rows.append({"name": "BPM1", "turn": t, "x": x, "y": y})
    df = pd.DataFrame(rows).set_index("name")
    df.index.name = "name"
    return df


# ---------------------------------------------------------------------------
# check_data_index
# ---------------------------------------------------------------------------


def test_check_data_index_already_indexed() -> None:
    """Data already indexed by 'name' must be returned with the same index."""
    df = pd.DataFrame({"turn": [0], "x": [0.0], "y": [0.0]}, index=pd.Index(["BPM1"], name="name"))
    result = check_data_index(df)
    assert result.index.name == "name"
    assert list(result.index) == ["BPM1"]


def test_check_data_index_from_name_column() -> None:
    """A 'name' column (not index) must be promoted to the index."""
    df = pd.DataFrame({"name": ["BPM1"], "turn": [0], "x": [0.0], "y": [0.0]})
    result = check_data_index(df)
    assert result.index.name == "name"
    assert "name" not in result.columns

    df = pd.DataFrame({"NAME": ["BPM1"], "turn": [0], "x": [0.0], "y": [0.0]})
    result = check_data_index(df)
    assert result.index.name == "NAME"


def test_check_data_index_missing_name_raises() -> None:
    """When neither index name nor name/NAME column exists, ValueError is raised."""
    df = pd.DataFrame({"turn": [0], "x": [0.0], "y": [0.0]})
    with pytest.raises(ValueError, match="name"):
        check_data_index(df)


def test_check_data_index_wrong_index_name_raises() -> None:
    """An index with a non-name index name must raise ValueError."""
    df = pd.DataFrame(
        {"turn": [0], "x": [0.0], "y": [0.0]}, index=pd.Index(["BPM1"], name="element")
    )
    with pytest.raises(ValueError, match="named 'name'"):
        check_data_index(df)


# ---------------------------------------------------------------------------
# subtract_closed_orbit
# ---------------------------------------------------------------------------


def test_subtract_closed_orbit_pre_kick_mean_zero() -> None:
    """After subtraction, the mean of pre-kick x and y at each BPM must be ≈ 0."""
    rng = np.random.default_rng(0)
    n_turns_free = 50
    co_x, co_y = 1.2e-4, -3.5e-5
    turns = list(range(n_turns_free + 5))
    rows = [
        {
            "name": "BPM1",
            "turn": t,
            "x": co_x + rng.normal(0, 1e-6),
            "y": co_y + rng.normal(0, 1e-6),
        }
        for t in turns
    ]
    df = pd.DataFrame(rows).set_index("name")
    df.index.name = "name"

    result, _, _ = subtract_closed_orbit(df, n_turns_free=n_turns_free)
    pre_kick = result[result["turn"] < n_turns_free]
    assert abs(pre_kick["x"].mean()) < 1e-10
    assert abs(pre_kick["y"].mean()) < 1e-10


def test_subtract_closed_orbit_preserves_displacement() -> None:
    """A known displacement added after the pre-kick window must survive subtraction."""
    n_turns_free = 20
    co_x = 5e-4
    rows = []
    for t in range(n_turns_free + 1):
        extra_x = 1e-3 if t >= n_turns_free else 0.0
        rows.append({"name": "BPM1", "turn": t, "x": co_x + extra_x, "y": 0.0})
    df = pd.DataFrame(rows).set_index("name")
    df.index.name = "name"

    result, _, _ = subtract_closed_orbit(df, n_turns_free=n_turns_free)
    post_kick_x = float(result[result["turn"] == n_turns_free]["x"].iloc[0])
    assert abs(post_kick_x - 1e-3) < 1e-12


# ---------------------------------------------------------------------------
# find_kick
# ---------------------------------------------------------------------------


def test_find_kick_identifies_correct_turn() -> None:
    """find_kick must return the turn at which the kick was injected."""
    n_turns_free = 10
    kick_turn = 10
    df = _make_data(px_kick=1e-4, py_kick=0.0, n_turns_free=n_turns_free, kick_turn=kick_turn)
    _, found_turn = find_kick(df, n_turns_free=n_turns_free)
    assert found_turn == kick_turn


def test_find_kick_identifies_correct_bpm() -> None:
    """find_kick must return the BPM with the largest displacement on the kick turn."""
    n_turns_free = 10
    kick_turn = 10
    rows = []
    for t in range(kick_turn + 1):
        for bpm, x_kick in [("BPM1", 1e-4), ("BPM2", 5e-5)]:
            x = x_kick if t == kick_turn else 0.0
            rows.append({"name": bpm, "turn": t, "x": x, "y": 0.0})
    df = pd.DataFrame(rows).set_index("name")
    df.index.name = "name"

    found_bpm, _ = find_kick(df, n_turns_free=n_turns_free)
    assert found_bpm == "BPM1"


# ---------------------------------------------------------------------------
# reconstruct_momentum_kick — Twiss path (no MAD-NG)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "px_kick,py_kick",
    [
        pytest.param(1e-5, 0.0, id="horizontal"),
        pytest.param(0.0, 1e-5, id="vertical"),
        pytest.param(1e-5, 1e-5, id="diagonal"),
    ],
)
def test_reconstruct_kick_twiss_path(
    simple_twiss: pd.DataFrame,
    px_kick: float,
    py_kick: float,
) -> None:
    """Twiss-path reconstruction recovers the true kick momenta at the
    quarter-wave waist where R12_x = R12_y = 1.

    The synthetic dataset has zero closed orbit and the BPM displacement on
    the kick turn equals the kick momentum directly (R12 = 1), so the
    reconstructed values must match exactly within floating-point precision.
    """
    n_turns_free = 10
    kick_turn = 10
    df = _make_data(
        px_kick=px_kick, py_kick=py_kick, n_turns_free=n_turns_free, kick_turn=kick_turn
    )

    result = reconstruct_momentum_kick(
        df,
        twiss=simple_twiss,
        n_turns_free=n_turns_free,
        n_turns_after_kick=3,
    )

    kick_rows = result[(result["turn"] == kick_turn) & (result.index == "BPM1")]
    assert not kick_rows.empty, "Expected a kick row for BPM1 on the kick turn"
    rec_px = float(kick_rows["px"].iloc[0])
    rec_py = float(kick_rows["py"].iloc[0])

    assert abs(rec_px - px_kick) < 1e-13, f"px: expected {px_kick}, got {rec_px}"
    assert abs(rec_py - py_kick) < 1e-13, f"py: expected {py_kick}, got {rec_py}"
