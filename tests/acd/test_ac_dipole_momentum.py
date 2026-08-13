"""Full-pipeline AC-dipole momentum reconstruction test for the LHC.

A single integration test (parametrised over the measurement noise level, the
sequence/optics, and the beam momentum offset) that tracks one AC-dipole
excitation through the LHC, reconstructs the per-turn momenta with
:func:`calculate_ac_dipole_momentum`, and checks that the reconstruction is
excellent: the harmonic kick fit has a very good R^2 against the tracked truth
and the reconstructed momenta agree with the truth at the upstream/downstream
BPMs and at the ``<acd>_before`` / ``<acd>_after`` markers that bracket the
AC-dipole kick in the tracking line.

Both the on-momentum (δp=0) and off-momentum (dispersive) cases are covered; for
the off-momentum case the MAD-NG model is built at the matching ``pt`` so the
reconstruction is dispersion-aware.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymadng_utils.accelerators import LHC

from tmom_recon import inject_noise_xy
from tmom_recon.acd.reconstruction import calculate_ac_dipole_momentum

from .acd_test_helpers import (
    AC_DIPOLE_ELEMENT,
    _ac_dipole_segment_around_element,
    _get_driver,
    acd_state_marker_names,
    assert_acd_momenta_match_truth,
)

DRIVEN_TUNES = (0.27, 0.322)
OFF_MOMENTUM_DELTA_P = 4e-4

# The 120 cm crossing optics carry a closed orbit, coupling and dispersion
# through the AC-dipole region, so the marker momentum reconstruction is
# marginally less perfect there than for the round, flat lhcb1 optics. The
# reconstruction is still excellent (R^2 > 0.9995), so the crossing sequence
# gets a slightly looser marker threshold. Off-momentum reconstruction folds in
# the dispersion model, which loosens the achievable R^2 a little further.
MARKER_R2_MIN = {
    ("lhcb1.seq", True): 0.9998,
    ("lhcb1.seq", False): 0.999,
    ("b1_120cm_crossing.seq", True): 0.9995,
    ("b1_120cm_crossing.seq", False): 0.999,
}


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
@pytest.mark.parametrize(
    "delta_p", [0.0, OFF_MOMENTUM_DELTA_P], ids=["on_momentum", "off_momentum"]
)
@pytest.mark.parametrize("noise_std", [0.0, 1e-5], ids=["no_noise", "noise_1e-5"])
def test_ac_dipole_momentum_reconstruction(
    data_dir, acd_tracking_setup, noise_std: float, delta_p: float, seq_file: str
) -> None:
    noisy = noise_std != 0.0
    on_momentum = delta_p == 0.0
    setup = acd_tracking_setup(
        seq_file, data_dir, delta_p=delta_p, flattop_turns=200, state_markers=True
    )
    tracking_df, tws = setup["tracking_df"], setup["tws"]

    seq = data_dir / "sequences" / seq_file
    pt = LHC(beam=1, sequence_file=seq, kinetic_energy=6800).dp2pt(delta_p)
    model = _get_driver(seq, pt=pt, debug=False)
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        model.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )

    # Feed only the BPM rows to the reconstruction (optionally noised); the
    # `<acd>_before` / `<acd>_after` marker rows are kept aside purely as truth.
    before_marker, after_marker = acd_state_marker_names(model)
    bpm_df = tracking_df.loc[~tracking_df["name"].isin([before_marker, after_marker])].copy()
    if noisy:
        bpm_df = inject_noise_xy(bpm_df, np.random.default_rng(42), noise_std)

    result = calculate_ac_dipole_momentum(
        bpm_df,
        tws,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        model=model,
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        closed_orbit_tws=tws,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
    )

    raw_markers = result.attrs["raw_marker_states"]
    assert set(raw_markers["name"].str.upper()) == {before_marker, after_marker}
    assert len(raw_markers) == 2 * len(result.attrs["summary"])
    assert raw_markers[["x", "px", "y", "py"]].notna().all().all()

    marker_r2_min = MARKER_R2_MIN[seq_file, on_momentum]
    kick_r2_min = 0.998 if noisy else (0.9999 if on_momentum else 0.999)
    assert_acd_momenta_match_truth(
        result,
        tracking_df,
        model,
        clean=not noisy,
        kick_r2_min=kick_r2_min,
        bpm_r2_min=0.9998 if on_momentum else 0.999,
        marker_r2_min=marker_r2_min,
        marker_pos_r2_min=marker_r2_min,
    )
