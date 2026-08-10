"""AC-dipole reconstruction with a non-zero (dipole-error) closed orbit.

This reproduces the real PSB-measurement failure path in ``../psb_md`` where the
MAD-NG model and its twiss both carry a distorted closed orbit (from uncorrected
dipole errors), and the reconstruction's internal round-trip consistency check
(:func:`tmom_recon.acd.reconstruction._check_bpm_state_consistency`) rejects the
reconstructed BPM state, e.g.::

    ValueError: Reconstructed x at BPM BR3.BPM2L3 does not match the predicted
    value within absolute tolerance 1.0e-04 (max|residual|=7.698e-04).

To decide whether that is a reconstruction bug or a physics/model-mismatch issue,
this test drives the same pipeline in a fully controlled simulation:

* a 0.08% RMS relative field error is added to every powered PSB main bend, in a
  *matched* way across the xsuite tracking line and the MAD-NG reconstruction
  model (see :func:`tests.psb_tracking.build_psb_tracking_setup`), so the tracked
  data and the model share the *same* distorted closed orbit. The two codes agree
  to ~1e-9 at the BPMs (the xsuite/MAD-NG floor) — four orders of magnitude below
  the reconstruction's 1e-4 consistency tolerance;
* the model twiss therefore has a genuine non-zero closed orbit that matches the
  model, so :func:`_check_has_zero_closed_orbit` takes the closed-orbit-removal
  branch — exactly like the real measurement;
* the reconstruction is then required to recover the tracked truth.

Because the tracked data and the model share the closed orbit to ~1e-9 here, a
failure at ``_check_bpm_state_consistency`` (observed residual ~1e-4 in x) *cannot*
be a data/model mismatch — it isolates a bug in the reconstruction's closed-orbit
handling in ``reconstruct_from_prepared``. If this test fails at that check, that is
the signal to debug the CO handling there.

Both a dipole *field* error and a quadrupole *gradient* error are applied, again
matched between the tracking line and the model. The gradient error does not kick
the orbit by itself; it perturbs beta, tune and dispersion, so the distorted orbit
samples the perturbed gradients off-axis. That is what makes the error orbit and
the dispersive orbit non-separable — measured on this setup, ``CO(err, 0) +
CO(0, dp)`` differs from ``CO(err, dp)`` by ~6e-4 in ``x``, far above the 1e-4
consistency tolerance — so no scheme that adds an error-orbit term to a dispersion
term can be correct here.

The test is parametrised over ``delta_p``:

* ``delta_p == 0``: a pure *dipole-error* closed orbit (orbit distortion at
  ``pt == 0``);
* ``delta_p != 0``: the dipole-error orbit *plus* a dispersive closed orbit, which
  is what the real PSB measurement actually carries (the beam is both off-orbit and
  off-momentum). This reproduces a separate ``x`` failure at
  ``_check_bpm_state_consistency`` (max|residual| ~3e-4) that the on-momentum case
  does not, isolating the dispersive branch of the closed-orbit handling.

and over the two ways ``pt`` can enter the closed-orbit reference
(``ACDipoleConfig.dispersive_closed_orbit``):

* ``linear``: the ``dp/p=0`` twiss closed orbit plus a first-order ``pt * D``
  dispersive orbit. Correct only to first order — the neglected ``pt**2 * D2``
  term is a constant per-BPM offset, so the reconstructed momenta keep their
  shape but acquire a bias, which is why it shows up as a mildly degraded R^2
  rather than as noise. Off momentum this is expected to fail, and is marked
  ``xfail(strict=True)`` so that fixing it is noticed rather than silently
  absorbed.
* ``exact``: the closed orbit MAD-NG solves at the model's ``pt``, exact to all
  orders and carrying the magnet errors the model knows about, with the
  ``pt * D`` correction switched off. Required to hold to the same strict
  thresholds on and off momentum.
"""

from __future__ import annotations

import pytest

from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES, build_psb_tracking_setup
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_pz

from .acd_test_helpers import acd_state_marker_names, assert_acd_momenta_match_truth

# 0.08% relative bend error, matching the orbit scale seen in the real measurement.
BEND_ERROR_RMS = 8e-4
BEND_ERROR_SEED = 7
# 0.1% relative quadrupole gradient error: perturbs the optics (beta beating,
# tune, dispersion) without kicking the orbit directly.
QUAD_ERROR_RMS = 1e-3
QUAD_ERROR_SEED = 11
ACD_DRIVEN_TUNES = (0.18, DRIVEN_TUNES[1])
OFF_MOMENTUM_DELTA_P = 8.0e-3


@pytest.mark.slow
@pytest.mark.parametrize(
    ("delta_p", "dispersive_closed_orbit"),
    [
        pytest.param(0.0, False, id="on_momentum-linear"),
        pytest.param(0.0, True, id="on_momentum-exact"),
        pytest.param(
            OFF_MOMENTUM_DELTA_P,
            False,
            id="off_momentum-linear",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "First-order pt*D dispersive orbit: the neglected pt**2*D2 term is a "
                    "constant per-BPM offset (~1e-5 rad in px against ~1e-4 rad of driven "
                    "px), amplified by the error orbit feeding down through the perturbed "
                    "quadrupoles. Use dispersive_closed_orbit=True instead."
                ),
            ),
        ),
        pytest.param(OFF_MOMENTUM_DELTA_P, True, id="off_momentum-exact"),
    ],
)
def test_psb_acd_reconstruction_with_dipole_closed_orbit(
    delta_p, dispersive_closed_orbit, data_dir
) -> None:
    setup = build_psb_tracking_setup(
        data_dir,
        delta_p=delta_p,
        driven_tunes=ACD_DRIVEN_TUNES,
        bend_error_rms=BEND_ERROR_RMS,
        bend_error_seed=BEND_ERROR_SEED,
        quad_error_rms=QUAD_ERROR_RMS,
        quad_error_seed=QUAD_ERROR_SEED,
    )
    tracking_df = setup["tracking_df"]
    tws = setup["tws"]
    model = setup["model"]
    # The regenerated model inside `calculate_pz` must carry the same errors as the
    # tracked line, otherwise its closed orbit is the wrong lattice's and neither
    # pt method can be correct.
    magnet_strengths = {f"{name.upper()}.k0": value for name, value in setup["bend_k0"].items()}
    magnet_strengths.update(
        {f"{name.upper()}.k1": value for name, value in setup["quad_k1"].items()}
    )

    # The dipole errors (plus dispersion when off-momentum) must actually distort
    # the closed orbit, otherwise this test would not exercise the non-zero-CO
    # branch it is meant to check.
    assert float(tws["x"].abs().max()) > 1e-3, "expected a distorted closed orbit"

    # Feed only the BPM rows to the reconstruction; the `<acd>_before` /
    # `<acd>_after` marker rows are kept aside purely as truth.
    before_marker, after_marker = acd_state_marker_names(model)
    bpm_df = tracking_df.loc[~tracking_df["name"].isin([before_marker, after_marker])].copy()

    # If the closed-orbit handling is broken, this raises inside
    # `_check_bpm_state_consistency`; that is the failure we want to catch here.
    result = calculate_pz(
        bpm_df,
        model_details=ModelDetails(
            accelerator=model.accelerator,
            pt=model.pt,
            magnet_strengths=magnet_strengths,
        ),
        acd=ACDipoleConfig(
            ac_dipole_marker=ACD_ELEMENT,
            driven_tunes=ACD_DRIVEN_TUNES,
            dispersive_closed_orbit=dispersive_closed_orbit,
        ),
        acd_only=True,
    )

    assert_acd_momenta_match_truth(
        result,
        tracking_df,
        model,
        clean=True,
        kick_r2_min=0.999,
        bpm_r2_min=0.999,
        marker_r2_min=0.998,
        marker_pos_r2_min=0.998,
    )
