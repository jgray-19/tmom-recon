import numpy as np
import pandas as pd
import pytest

from tmom_recon.physics.bpm_phases import (
    DEGENERACY_BAND,
    barrier_block_matrix,
    next_bpm_to_pi,
    next_bpm_to_pi_2,
    phase_advance_matrix_from_edges,
    phase_advance_matrix_from_tws,
    prev_bpm_to_pi,
    prev_bpm_to_pi_2,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def next_mu():
    return pd.Series(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.75],
        index=["BPM1", "BPM2", "BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM8"],
    )


@pytest.fixture
def prev_mu():
    return pd.Series(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9],
        index=["BPM1", "BPM2", "BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM8"],
    )


expected_results = {
    "next_bpm_to_pi_2": {
        "names": ["BPM3", "BPM4", "BPM5", "BPM6", "BPM7", "BPM8", "BPM8", "BPM1"],
        "deltas": [-0.05, -0.05, -0.05, -0.05, -0.05, 0.00, -0.1, 0.0],
    },
    "prev_bpm_to_pi_2": {
        "names": ["BPM7", "BPM7", "BPM1", "BPM1", "BPM2", "BPM4", "BPM6", "BPM6"],
        "deltas": [-0.05, 0.05, -0.05, 0.05, 0.05, 0.05, -0.05, 0.05],
    },
    "next_bpm_to_pi": {
        # . FROM   BPM1.   BPM2.   BPM3.   BPM4.   BPM5.   BPM6.   BPM7.   BPM8
        "names": ["BPM6", "BPM7", "BPM8", "BPM8", "BPM1", "BPM1", "BPM2", "BPM3"],
        "deltas": [0, 0, 0.05, -0.05, 0.1, 0, 0, -0.05],
    },
    "prev_bpm_to_pi": {
        "names": ["BPM4", "BPM5", "BPM6", "BPM6", "BPM1", "BPM2", "BPM4", "BPM5"],
        "deltas": [0, 0, -0.1, 0, -0.1, 0, 0, 0],
    },
}


@pytest.mark.parametrize(
    "func, mu_fixture",
    [
        (next_bpm_to_pi_2, "next_mu"),
        (prev_bpm_to_pi_2, "prev_mu"),
    ],
)
def test_bpm_to_pi_2(func, mu_fixture, request):
    mu = request.getfixturevalue(mu_fixture)
    tune = 1.0

    forward = "next" in func.__name__
    result = func(phase_advance_matrix_from_tws(mu, tune, forward=forward))

    key = "next_bpm" if forward else "prev_bpm"
    expected = expected_results[func.__name__]
    assert all(result[key] == expected["names"])
    assert np.allclose(result["delta"], expected["deltas"])


@pytest.mark.parametrize(
    "func, mu_fixture, tune",
    [
        (next_bpm_to_pi, "next_mu", 1.0),
        (prev_bpm_to_pi, "prev_mu", 0.8),
    ],
)
def test_bpm_to_pi(func, mu_fixture, tune, request):
    mu = request.getfixturevalue(mu_fixture)

    forward = "next" in func.__name__
    result = func(phase_advance_matrix_from_tws(mu, tune, forward=forward))

    key = "next_bpm" if forward else "prev_bpm"
    expected = expected_results[func.__name__]
    assert all(result[key] == expected["names"])
    assert np.allclose(result["delta"], expected["deltas"])


@pytest.mark.parametrize("forward", [True, False])
def test_edge_matrix_matches_cumulative_form(forward):
    """Accumulating mod-1 edges around the ring reproduces the cumulative-phase matrix.

    The ring closes through the final (last -> first) edge rather than a ``% tune``
    boundary, so the closing advance is the edge value itself.
    """
    mu = pd.Series(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75],
        index=[f"BPM{i}" for i in range(8)],
    )
    tune = 1.0

    # Edges in ring order, including the closing edge that returns to the first BPM.
    edge_vals = np.diff(mu.to_numpy(), append=tune + mu.to_numpy()[0])
    edges = pd.Series(edge_vals, index=mu.index)

    from_edges = phase_advance_matrix_from_edges(edges, forward=forward)
    from_tws = phase_advance_matrix_from_tws(mu, tune, forward=forward)

    pd.testing.assert_frame_equal(from_edges, from_tws)


def _assert_closest_within_oscillation(matrix, result, key, target):
    """Each match must be the BPM whose cumulative phase advance (within one
    betatron oscillation) is closest to ``target``."""
    for bpm in matrix.index:
        matched = result.loc[bpm, key]
        if pd.isna(matched):
            continue
        advance = matrix.loc[bpm].to_numpy(dtype=float)
        candidates = advance[np.isfinite(advance) & (advance > 0.0) & (advance <= 1.0)]
        assert candidates.size, f"no candidate within one oscillation for {bpm}"
        best = float(np.min(np.abs(candidates - target)))
        chosen = abs(float(matrix.at[bpm, matched]) - target)
        # Degenerate candidates (near a multiple of pi) may be skipped, so the
        # chosen match is the closest non-degenerate candidate.
        assert chosen <= best + DEGENERACY_BAND


def test_pi_2_selects_closest_phase_within_oscillation():
    """For π/2, the matched BPM has cumulative phase advance closest to 0.25 turns."""
    mu = pd.Series(
        np.linspace(0, 0.9, 50),  # 50 BPMs over nearly 1 full turn.
        index=[f"BPM{i}" for i in range(50)],
    )
    matrix = phase_advance_matrix_from_tws(mu, 1.0, forward=True)
    result = next_bpm_to_pi_2(matrix)
    _assert_closest_within_oscillation(matrix, result, "next_bpm", 0.25)


def test_pi_selects_closest_phase_within_oscillation():
    """For π, the matched BPM has cumulative phase advance closest to 0.5 turns."""
    mu = pd.Series(
        np.linspace(0, 0.95, 50),  # 50 BPMs over nearly 1 full turn
        index=[f"BPM{i}" for i in range(50)],
    )
    matrix = phase_advance_matrix_from_tws(mu, 1.0, forward=True)
    result = next_bpm_to_pi(matrix)
    _assert_closest_within_oscillation(matrix, result, "next_bpm", 0.5)


def test_barrier_block_matrix_forward_and_backward():
    """The mask flags exactly the BPM pairs whose directed segment spans the barrier."""
    # Five BPMs at integer s; barrier sits between index 1 and 2 (s = 1.5).
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    fwd = barrier_block_matrix(s, 1.5, forward=True)
    bwd = barrier_block_matrix(s, 1.5, forward=False)

    # Forward segment i -> j contains 1.5 iff it starts at/before s=1 and ends past it.
    assert fwd[1, 2] and fwd[1, 3] and fwd[0, 2]
    assert not fwd[0, 1] and not fwd[2, 4] and not fwd[2, 3]
    # A forward segment that wraps the ring (3 -> 1) does not pass 1.5.
    assert not fwd[3, 1]
    # Backward mask is the transpose relationship: segment j -> i contains 1.5.
    assert bwd[2, 1] and bwd[3, 1] and bwd[2, 0]
    assert not bwd[1, 0] and not bwd[4, 2]
    # No pair is ever blocked against itself.
    assert not np.any(np.diag(fwd)) and not np.any(np.diag(bwd))


def test_blocked_mask_drops_cross_barrier_neighbour():
    """A π/2 neighbour on the far side of a barrier is excluded, leaving the near side."""
    # BPMs every ~π/2; the natural next/prev neighbour is the immediate ring neighbour.
    mu = pd.Series(
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        index=["A", "B", "C", "D", "E", "F"],
    )
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    # Barrier between C (s=2) and D (s=3).
    fwd = barrier_block_matrix(s, 2.5, forward=True)
    bwd = barrier_block_matrix(s, 2.5, forward=False)

    fwd_matrix = phase_advance_matrix_from_tws(mu, 1.5, forward=True)
    bwd_matrix = phase_advance_matrix_from_tws(mu, 1.5, forward=False)

    # Without masking, C pairs forward with D and D pairs backward with C.
    assert next_bpm_to_pi_2(fwd_matrix).loc["C", "next_bpm"] == "D"
    assert prev_bpm_to_pi_2(bwd_matrix).loc["D", "prev_bpm"] == "C"

    # With masking, those cross-barrier matches are dropped (no near-side π/2
    # candidate exists in this minimal ring), while distant BPMs are unaffected.
    blocked_next = next_bpm_to_pi_2(fwd_matrix, blocked=fwd)
    blocked_prev = prev_bpm_to_pi_2(bwd_matrix, blocked=bwd)
    assert pd.isna(blocked_next.loc["C", "next_bpm"])
    assert pd.isna(blocked_prev.loc["D", "prev_bpm"])
    assert blocked_next.loc["A", "next_bpm"] == "B"
