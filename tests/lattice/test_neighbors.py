import numpy as np
import pandas as pd

from tmom_recon.data.schema import NEXT, PREV
from tmom_recon.lattice.neighbors import build_lattice_neighbor_tables


def _ring_twiss(*, include_s: bool = True):
    """Six BPMs spaced ~π/2 apart over a single turn, with integer s positions."""
    names = ["A", "B", "C", "D", "E", "F"]
    mu = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    columns = {
        "beta11": 1.0,
        "beta22": 1.0,
        "alfa11": 0.0,
        "alfa22": 0.0,
        "mu1": mu,
        "mu2": mu,
    }
    if include_s:
        columns = {"s": np.arange(len(names), dtype=float), **columns}
    tws = pd.DataFrame(columns, index=names)
    # The neighbour builder reads the tunes off the frame as attributes; set them
    # last so they survive (a copy would drop them).
    tws.q1 = 1.5
    tws.q2 = 1.5
    return tws


def test_barrier_s_excludes_cross_barrier_neighbours():
    """A barrier between C and D stops C/D from pairing across it."""
    tws = _ring_twiss()

    # No barrier: C's forward neighbour is D and D's backward neighbour is C.
    prev_x, _, next_x, _ = build_lattice_neighbor_tables(tws, include_errors=False)
    assert next_x.loc["C", NEXT.bpm_x] == "D"
    assert prev_x.loc["D", PREV.bpm_x] == "C"

    # Barrier at s = 2.5 (between C and D): those cross-barrier pairs are dropped
    # in both planes, while a BPM far from the barrier keeps its neighbour.
    prev_x, prev_y, next_x, next_y = build_lattice_neighbor_tables(
        tws, include_errors=False, barrier_s=2.5
    )
    assert pd.isna(next_x.loc["C", NEXT.bpm_x])
    assert pd.isna(next_y.loc["C", NEXT.bpm_y])
    assert pd.isna(prev_x.loc["D", PREV.bpm_x])
    assert pd.isna(prev_y.loc["D", PREV.bpm_y])
    assert next_x.loc["A", NEXT.bpm_x] == "B"


def test_missing_s_column_skips_barrier(caplog):
    """Without an 's' column the barrier is ignored (with a warning), not an error."""
    tws = _ring_twiss(include_s=False)

    with caplog.at_level("WARNING"):
        _, _, next_x, _ = build_lattice_neighbor_tables(tws, include_errors=False, barrier_s=2.5)

    # Falls back to the unbarriered behaviour.
    assert next_x.loc["C", NEXT.bpm_x] == "D"
    assert any("'s' column" in record.message for record in caplog.records)
