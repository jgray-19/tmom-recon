import logging

import numpy as np
import numpy.typing as npt
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Half-width (turns) of the band around multiples of pi within which a
# candidate phase advance is considered degenerate for momentum reconstruction.
DEGENERACY_BAND = 0.05


def phase_advance_difference(
    mu_from: npt.ArrayLike | float, mu_to: npt.ArrayLike | float, tune: float
) -> np.ndarray:
    """Return the forward phase advance from ``mu_from`` to ``mu_to`` in turns.

    Accumulated around the ring, so the closing step from the last BPM back to the
    first is just another phase advance (its value is ``tune`` minus the spanned
    phase). For BPMs within one oscillation the result is ``< 1`` turn, i.e. the
    natural mod-1 phase advance.
    """
    return (np.asarray(mu_to, dtype=float) - np.asarray(mu_from, dtype=float)) % tune


def _accumulate_phase_matrix(
    cumulative: np.ndarray, total: float, index: pd.Index, *, forward: bool
) -> pd.DataFrame:
    """Accumulate edge advances into the BPM-to-BPM phase-advance matrix.

    ``cumulative`` is the running phase of each BPM around the ring (the prefix sum of
    the edge advances, first BPM at 0) and ``total`` is the full phase around the ring
    (the sum of every edge, i.e. the tune). Entry ``[i, j]`` is the phase accumulated
    stepping from BPM ``i`` to BPM ``j`` in the requested direction; the ring closes
    through the accumulation itself, so there is no tune boundary to special-case.
    """
    n = len(cumulative)
    a = cumulative.reshape(n, 1)
    b = cumulative.reshape(1, n)
    diff = (b - a + total) % total if forward else (a - b + total) % total
    np.fill_diagonal(diff, np.nan)
    return pd.DataFrame(diff, index=index, columns=index)


def phase_advance_matrix_from_edges(edges: pd.Series, *, forward: bool) -> pd.DataFrame:
    """Build the BPM-to-BPM phase-advance matrix from adjacent (edge) advances.

    ``edges`` holds the mod-1 phase advance between consecutive BPMs in ring order:
    ``edges.iloc[k]`` is the forward advance from BPM ``k`` to BPM ``k + 1``, and the
    final entry is the closing edge from the last BPM back to the first. The ring
    closes naturally through that edge -- the advance from the last BPM to the first is
    simply its measured phase advance, with no tune boundary.

    Entry ``[i, j]`` is the forward advance from BPM ``i`` to BPM ``j`` (backward if
    ``forward`` is False). Advances within one betatron oscillation (the ones the
    neighbour search keeps) are ``< 1`` turn.
    """
    edge_vals = edges.to_numpy(float)
    total = float(edge_vals.sum())
    # Running phase of each BPM relative to the first (first BPM at 0); the last edge
    # (closing the ring) lifts the total to ``tune`` but is not a per-BPM position.
    cumulative = np.concatenate(([0.0], np.cumsum(edge_vals)[:-1]))
    return _accumulate_phase_matrix(cumulative, total, edges.index, forward=forward)


def phase_advance_matrix_from_tws(mu: pd.Series, tune: float, *, forward: bool) -> pd.DataFrame:
    """Build the phase-advance matrix from a model cumulative-phase column (turns).

    The model provides the cumulative phase ``mu`` directly; ``tune`` is the total
    betatron phase around the ring (the sum of all adjacent edge advances), i.e. the
    value of the single closing edge that takes the last BPM back to the first. This is
    the cumulative-phase form of :func:`phase_advance_matrix_from_edges`: ``mu`` is the
    prefix sum of the edges, so no edges are reconstructed here.
    """
    return _accumulate_phase_matrix(mu.to_numpy(float), float(tune), mu.index, forward=forward)


def _phase_pair_var_forward(
    var_mu: np.ndarray, i: np.ndarray, j: np.ndarray, total_var: float
) -> np.ndarray:
    """
    Variance of forward phase advance from i -> j (turns^2), with wrap-around.
    i, j are integer arrays of equal length.
    """
    out = np.empty_like(var_mu[i], dtype=float)
    mask = j >= i
    out[mask] = var_mu[j[mask]] - var_mu[i[mask]]
    out[~mask] = (total_var - var_mu[i[~mask]]) + var_mu[j[~mask]]
    # Numerical safety (should not be negative, but floating noise can happen)
    return np.maximum(out, 0.0)


def _find_bpm_phase(
    phase_matrix: pd.DataFrame,
    target: float,
    forward: bool,
    name: str,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    Find BPM pairs with phase advance closest to a target value.

    For each BPM_i, finds the BPM_j in the specified direction (forward or
    backward) within one betatron oscillation (phase advance <= 1 turn) whose
    phase advance is closest to the target. Candidates whose phase advance is
    degenerate (within DEGENERACY_BAND of a multiple of pi, where sin(dphi) -> 0
    and the momentum reconstruction is singular) are avoided unless no other
    candidate exists.

    Args:
        phase_matrix: DataFrame of BPM-to-BPM phase advances,
        with BPM names as index and columns.
        target: Target phase advance (turns), e.g., 0.25 for π/2, 0.5 for π
        forward: If True, search forward; if False, search backward
        name: Column name for the matched BPM in output DataFrame
        mu_var: Optional phase variance for each BPM (rotations²)
        total_var: Optional total phase variance around ring (rotations²)

    Returns:
        DataFrame with columns:
            - {name}: Name of matched BPM
            - delta: Signed phase error (actual - target) in rotations
            - delta_err: Phase error uncertainty (rotations), if variance provided
    """
    n = len(phase_matrix.index)

    # Multiples of pi (in turns) where sin(dphi) -> 0 and the two-BPM momentum
    # reconstruction becomes singular; avoided unless no alternative exists.
    degenerate_points = [p for p in (0.0, 0.5, 1.0) if not np.isclose(p, target)]

    idx = np.full(n, -1, dtype=int)

    for i in range(n):
        row = phase_matrix.iloc[i].to_numpy(dtype=float)

        # Candidates within one betatron oscillation in the requested direction
        candidates = np.where(np.isfinite(row) & (row > 0.0) & (row <= 1.0))[0]
        if len(candidates) == 0:
            continue

        advance = row[candidates]
        degenerate = np.zeros(len(candidates), dtype=bool)
        for point in degenerate_points:
            degenerate |= np.abs(advance - point) <= DEGENERACY_BAND
        if not np.all(degenerate):
            candidates = candidates[~degenerate]
            advance = advance[~degenerate]

        idx[i] = candidates[np.argmin(np.abs(advance - target))]

    # Build output DataFrame
    delta = np.full(n, np.nan)
    names = np.full(n, None, dtype=object)
    for i in range(n):
        if idx[i] != -1:
            delta[i] = phase_matrix.iloc[i, idx[i]] - target
            names[i] = phase_matrix.index[idx[i]]

    out = pd.DataFrame({name: names, "delta": delta}, index=phase_matrix.index)

    # Add uncertainty if variance information provided
    if mu_var is not None and total_var is not None:
        var_arr = mu_var.to_numpy(float)
        i_arr = np.arange(n, dtype=int)
        j_arr = idx.astype(int)

        if forward:
            pair_var = _phase_pair_var_forward(var_arr, i_arr, j_arr, float(total_var))
        else:
            # Backward from i to j is forward from j to i
            pair_var = _phase_pair_var_forward(var_arr, j_arr, i_arr, float(total_var))

        out["delta_err"] = np.sqrt(pair_var)

    return out


def prev_bpm_to_pi_2(
    phase_matrix: pd.DataFrame,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    Find previous BPM at π/2 phase advance.

    For each BPM_i, finds the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to π/2 (0.25 turns).

    Returns:
        DataFrame with columns:
            - prev_bpm: Name of matched previous BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
    """
    return _find_bpm_phase(
        phase_matrix,
        0.25,
        forward=False,
        name="prev_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )


def next_bpm_to_pi_2(
    phase_matrix: pd.DataFrame,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    Find next BPM at π/2 phase advance.

    For each BPM_i, finds the next BPM_j whose forward phase advance
    (mu_j - mu_i) is closest to π/2 (0.25 turns).

    Returns:
        DataFrame with columns:
            - next_bpm: Name of matched next BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
    """
    return _find_bpm_phase(
        phase_matrix,
        0.25,
        forward=True,
        name="next_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )


def prev_bpm_to_pi(
    phase_matrix: pd.DataFrame,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    Find previous BPM at π phase advance.

    For each BPM_i, finds the previous BPM_j whose backward phase advance
    (mu_i - mu_j) is closest to π (0.5 turns).

    Returns:
        DataFrame with columns:
            - prev_bpm: Name of matched previous BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
    """
    return _find_bpm_phase(
        phase_matrix,
        0.5,
        forward=False,
        name="prev_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )


def next_bpm_to_pi(
    phase_matrix: pd.DataFrame,
    *,
    mu_var: pd.Series | None = None,
    total_var: float | None = None,
) -> pd.DataFrame:
    """
    Find next BPM at π phase advance.

    For each BPM_i, finds the next BPM_j whose forward phase advance
    (mu_j - mu_i) is closest to π (0.5 turns).

    Returns:
        DataFrame with columns:
            - next_bpm: Name of matched next BPM
            - delta: Phase error (turns)
            - delta_err: Phase error uncertainty (turns), if variance provided
    """
    return _find_bpm_phase(
        phase_matrix,
        0.5,
        forward=True,
        name="next_bpm",
        mu_var=mu_var,
        total_var=total_var,
    )
