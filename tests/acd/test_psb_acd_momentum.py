"""Standalone full-pipeline AC-dipole momentum reconstruction test for the PSB.

Mirrors the LHC test ([test_ac_dipole_momentum.py]) for Proton Synchrotron
Booster ring 3, end to end:

1. build the PSB ring-3 xsuite line and track one AC-dipole excitation through a
   long ramp followed by a flat top, seeded on the (off-)momentum closed orbit,
   with ``<acd>_before`` / ``<acd>_after`` markers bracketing the kick so the
   tracked data carries the true pre/post-kick momenta;
2. reconstruct the per-turn momenta with :func:`calculate_ac_dipole_momentum`
   against a MAD-NG model of the same ring at the same pt; and
3. check that the harmonic kick fit has a very good R^2 against the tracked truth
   and that the reconstructed momenta agree with the truth at the BPMs and at the
   ``<acd>_before`` / ``<acd>_after`` markers.

The test is parametrised over the momentum offset ``delta_p`` and uses the orbit
MAD-NG solves at ``pt``; the first-order ``pt * D`` orbit model is not accurate
enough at ``delta_p = 1e-2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES
from tmom_recon import inject_noise_xy
from tmom_recon.acd.reconstruction import calculate_ac_dipole_momentum

from .acd_test_helpers import acd_state_marker_names, assert_acd_momenta_match_truth

pytestmark = [pytest.mark.psb, pytest.mark.integration]
__test__ = False

ORBIT_COLUMNS = ("x", "px", "y", "py")


@pytest.fixture(scope="module", params=[0.0, 1e-2], ids=["on_momentum", "off_momentum"])
def psb_acd_setup(request, psb_tracking_setup):
    """Track one PSB AC-dipole excitation seeded on the ``delta_p`` closed orbit."""
    return psb_tracking_setup(float(request.param))


@pytest.mark.slow
@pytest.mark.parametrize("noise_std", [0.0, 1e-5], ids=["clean", "noise_1e-5"])
def test_psb_ac_dipole_momentum_reconstruction(psb_acd_setup, noise_std: float) -> None:
    clean = noise_std == 0.0
    tracking_df = psb_acd_setup.measurement.data
    tws = psb_acd_setup.machine.madng_twiss
    model = psb_acd_setup.machine.madng_model

    # Feed only the BPM rows to the reconstruction (optionally noised); the
    # `<acd>_before` / `<acd>_after` marker rows are kept aside purely as truth.
    before_marker, after_marker = acd_state_marker_names(model)
    bpm_df = tracking_df.loc[~tracking_df["name"].isin([before_marker, after_marker])].copy()
    if not clean:
        bpm_df = inject_noise_xy(bpm_df, np.random.default_rng(42), noise_std)

    # `tws` is solved around the on-momentum orbit (x == 0 for this error-free
    # lattice), so using it as the closed-orbit reference models the dispersive
    # orbit as the first-order `pt * D`. At delta_p = 1e-2 that is 2.3e-5 m short
    # of the true orbit -- 10% of the driven amplitude -- so instead take the
    # orbit MAD-NG solves at `pt`, which matches the tracked orbit to 2.4e-7 m,
    # and tell the betatron stage not to add `pt * D` on top of it.
    closed_orbit_tws = model.run_twiss(observe=1, coupling=True, pt=model.pt)

    result = calculate_ac_dipole_momentum(
        bpm_df,
        tws,
        ac_dipole_marker=ACD_ELEMENT,
        model=model,
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        closed_orbit_tws=closed_orbit_tws,
        dispersion_tws=tws,
    )

    assert_acd_momenta_match_truth(
        result,
        tracking_df,
        model,
        clean=clean,
        kick_r2_min=0.99 if clean else 0.99,
        bpm_r2_min=0.99 if clean else 0.99,
        marker_r2_min=0.998,
        marker_pos_r2_min=0.998,
    )
