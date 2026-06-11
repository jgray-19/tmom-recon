from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tmom_recon.measurements.twiss_from_measurement import (
    _compute_cumulative_phase,
    _ordered_phase_edges,
    _reconstruct_ring_tune,
    build_twiss_from_measurements,
)

NAMES = [f"BPM{i}" for i in range(5)]

# A real OMC3 measurement (PSB ring 3, hio kick, 0 Hz): an open 16-BPM phase chain with
# fractional-only Q headers (Q1=0.170, Q2=0.230; true tunes ~4.17 / ~4.23).
PSB_MEAS_DIR = Path(__file__).parent.parent / "data" / "measurements" / "psb_hio_0Hz"


@pytest.fixture
def phase_df():
    """A small NAME -> NAME2 phase chain closing back on the first BPM.

    Advances are the measured (mod-1) FFT phase advances between adjacent BPMs.
    """
    return pd.DataFrame(
        {
            "NAME2": NAMES[1:] + NAMES[:1],  # BPM0->BPM1, ..., BPM4->BPM0 (closure)
            "PHASEX": [0.30, 0.30, 0.30, 0.30, 0.30],
            "ERRPHASEX": [0.01, 0.02, 0.03, 0.01, 0.05],
        },
        index=pd.Index(NAMES, name="NAME"),
    )


def test_cumulative_phase_accumulates_measured_edges(phase_df):
    result = _compute_cumulative_phase(phase_df, "PHASEX")

    assert list(result.index) == NAMES
    # mu is the running sum of the four inter-BPM advances, starting at 0; it grows past
    # one turn so a downstream "within one oscillation" cap can discriminate near vs far.
    np.testing.assert_allclose([result.mu[b] for b in NAMES], [0.0, 0.30, 0.60, 0.90, 1.20])

    expected_var = np.cumsum([0.0, 0.01**2, 0.02**2, 0.03**2, 0.01**2])
    np.testing.assert_allclose([result.var[b] for b in NAMES], expected_var)

    # total_var sums *all* edges around the ring, including the closure edge.
    expected_total = sum(e**2 for e in [0.01, 0.02, 0.03, 0.01, 0.05])
    assert result.total_var == pytest.approx(expected_total)


def test_cumulative_phase_reverse(phase_df):
    result = _compute_cumulative_phase(phase_df, "PHASEX", reverse=True)

    assert list(result.index) == NAMES[::-1]
    # Reverse accumulation traverses the same edges backwards; magnitudes are unchanged.
    np.testing.assert_allclose([result.mu[b] for b in NAMES[::-1]], [0.0, 0.30, 0.60, 0.90, 1.20])
    expected_total = sum(e**2 for e in [0.01, 0.02, 0.03, 0.01, 0.05])
    assert result.total_var == pytest.approx(expected_total)


def test_ordered_phase_edges_returns_measured_differences(phase_df):
    order, diffs, variances, total_var = _ordered_phase_edges(phase_df, "PHASEX", reverse=False)
    assert order == NAMES
    np.testing.assert_allclose(diffs, [0.30, 0.30, 0.30, 0.30])
    np.testing.assert_allclose(variances, [0.01**2, 0.02**2, 0.03**2, 0.01**2])
    assert total_var == pytest.approx(sum(e**2 for e in [0.01, 0.02, 0.03, 0.01, 0.05]))


def _open_chain_phase_data(rows):
    """PhaseData for an open chain (no closing edge), one PHASEX per ``rows`` entry.

    Like a real OMC3 phase file, the chain runs first BPM -> ... -> last BPM and does not
    close. Every measured edge is kept (including the one into the final BPM), so the chain
    has ``len(rows) + 1`` BPMs and the cumulative phase reaches ``mu_last = sum(rows)``.
    """
    n = len(rows)
    names = [f"BPM{i}" for i in range(n + 1)]
    phase_df = pd.DataFrame(
        {
            "NAME2": names[1:],  # BPM_k -> BPM_{k+1}; no row closes back to BPM0
            "PHASEX": list(rows),
            "ERRPHASEX": [0.0] * n,
        },
        index=pd.Index(names[:n], name="NAME"),
    )
    return _compute_cumulative_phase(phase_df, "PHASEX")


def test_open_chain_keeps_final_bpm():
    # 5 measured edges connect 6 BPMs; none is dropped.
    pd_x = _open_chain_phase_data([0.30, 0.30, 0.30, 0.30, 0.30])
    assert list(pd_x.index) == [f"BPM{i}" for i in range(6)]
    np.testing.assert_allclose([pd_x.mu[b] for b in pd_x.index], [0.0, 0.3, 0.6, 0.9, 1.2, 1.5])


def test_reconstruct_ring_tune_adds_closing_edge_from_fractional_header():
    # Open chain -> mu_last = 1.5 (integer part 1); fractional tune 0.17.
    pd_x = _open_chain_phase_data([0.30, 0.30, 0.30, 0.30, 0.30])
    # closing = (0.17 - 1.5) mod 1 = 0.67; full tune = 1.5 + 0.67 = 2.17.
    assert _reconstruct_ring_tune(pd_x, 0.17) == pytest.approx(2.17)
    # A full-tune header is normalised to its fractional part, so the result is unchanged.
    assert _reconstruct_ring_tune(pd_x, 4.17) == pytest.approx(2.17)


def test_reconstruct_ring_tune_falls_back_to_cumulative_without_header():
    pd_x = _open_chain_phase_data([0.30, 0.30, 0.30, 0.30, 0.30])
    assert _reconstruct_ring_tune(pd_x, None) == pytest.approx(1.5)


def test_empty_phase_df_raises():
    empty = pd.DataFrame({"NAME2": [], "PHASEX": [], "ERRPHASEX": []})
    with pytest.raises(ValueError, match="empty"):
        _compute_cumulative_phase(empty, "PHASEX")


def test_nan_phase_raises(phase_df):
    phase_df.loc["BPM2", "PHASEX"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _compute_cumulative_phase(phase_df, "PHASEX")


def test_build_twiss_from_real_open_chain_measurement():
    tws, _ = build_twiss_from_measurements(PSB_MEAS_DIR, include_errors=True)
    print(tws[["MUX", "MUY"]])

    # The open 16-BPM phase chain keeps every BPM, including the final one.
    assert len(tws) == 16
    assert "BR3.BPM16L3" in tws.index

    # Headers are reconstructed to the *full* tunes (integer part from the accumulated
    # phase, fractional from the OMC3 header's 0.170 / 0.230).
    assert tws.headers["Q1"] == pytest.approx(4.170040429688)
    assert tws.headers["Q2"] == pytest.approx(4.229992187500, abs=1e-9)
