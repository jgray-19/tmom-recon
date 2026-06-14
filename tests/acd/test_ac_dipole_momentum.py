"""Full-pipeline AC-dipole momentum reconstruction test for the LHC.

A single integration test (parametrised over the measurement noise level) that
tracks one AC-dipole excitation through the LHC, reconstructs the per-turn
momenta with :func:`calculate_ac_dipole_momentum`, and checks that the
reconstruction is excellent: the harmonic kick fit has a very good R^2 against
the tracked truth and the reconstructed momenta agree with the truth at the
upstream/downstream BPMs and at the ``<acd>_before`` / ``<acd>_after`` markers
that bracket the AC-dipole kick in the tracking line.
"""

from __future__ import annotations

import numpy as np
import pytest

from tmom_recon import inject_noise_xy
from tmom_recon.acd.reconstruction import calculate_ac_dipole_momentum

from .acd_test_helpers import (
    AC_DIPOLE_ELEMENT,
    _ac_dipole_segment_around_element,
    _get_driver,
    acd_state_marker_names,
    assert_acd_momenta_match_truth,
)

SEQ_FILE = "lhcb1.seq"
DRIVEN_TUNES = (0.27, 0.322)


@pytest.mark.slow
@pytest.mark.parametrize("noise_std", [0.0, 1e-5], ids=["clean", "noise_1e-5"])
def test_ac_dipole_momentum_reconstruction(data_dir, acd_tracking_setup, noise_std: float) -> None:
    clean = noise_std == 0.0
    setup = acd_tracking_setup(SEQ_FILE, data_dir, flattop_turns=200, state_markers=True)
    tracking_df, tws = setup["tracking_df"], setup["tws"]

    model = _get_driver(data_dir / "sequences" / SEQ_FILE, debug=False)
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        model.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )

    # Feed only the BPM rows to the reconstruction (optionally noised); the
    # `<acd>_before` / `<acd>_after` marker rows are kept aside purely as truth.
    before_marker, after_marker = acd_state_marker_names(model)
    bpm_df = tracking_df.loc[~tracking_df["name"].isin([before_marker, after_marker])].copy()
    if not clean:
        bpm_df = inject_noise_xy(bpm_df, np.random.default_rng(42), noise_std)

    result = calculate_ac_dipole_momentum(
        bpm_df,
        tws,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        model=model,
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        inject_noise=False,
    )

    assert_acd_momenta_match_truth(
        result,
        tracking_df,
        model,
        clean=clean,
        kick_r2_min=0.999 if clean else 0.99,
        bpm_r2_min=0.999,
        marker_r2_min=0.999,
    )
