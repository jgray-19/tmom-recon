import numpy as np
import pandas as pd
import pytest

from tmom_recon.physics.bpm_phases import (
    DEGENERACY_BAND,
    next_bpm_to_pi,
    next_bpm_to_pi_2,
    phase_advance_matrix_from_edges,
    phase_advance_matrix_from_tws,
    prev_bpm_to_pi,
    prev_bpm_to_pi_2,
)


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
