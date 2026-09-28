"""Contract for deriving ``D'`` from measured position dispersion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymadng_utils.accelerators import PSB
from pymadng_utils.mad import AcceleratorMadInterface

from tmom_recon.momentum_dispersion import (
    MIN_SIN_PHASE_ADVANCE,
    derive_momentum_dispersion,
)

RING = 3
KINETIC_ENERGY_GEV = 0.16


@pytest.fixture(scope="module")
def psb_twiss(seq_psb3) -> pd.DataFrame:
    """A real PSB ring-3 Twiss at the BPMs, with bends between them."""
    accelerator = PSB(sequence_file=str(seq_psb3), ring=RING, kinetic_energy=KINETIC_ENERGY_GEV)
    interface = AcceleratorMadInterface(accelerator)
    interface.observe()
    # betx/alfx only exist in a coupled twiss, where they equal beta11/alfa11.
    tws = interface.run_twiss(coupling=True)
    interface.close()
    tws.index = tws.index.astype(str).str.upper()
    return tws


def _tunes(tws: pd.DataFrame) -> tuple[float, float]:
    return float(tws.headers["q1"]), float(tws.headers["q2"])


def test_round_trip_recovers_the_model_momentum_dispersion(psb_twiss):
    """Feeding the model's own D back in must return the model's own D'.

    This is the whole formula under test at once: the transfer matrix, the
    ``m13`` bend term and the wrap of the final BPM pair around the ring. Any
    of them wrong and the identity fails, because a lattice's ``(D, D')`` is
    exactly the pair the propagation relates.
    """
    derived = derive_momentum_dispersion(psb_twiss, psb_twiss, tunes=_tunes(psb_twiss))

    for column in ("dpx", "dpy"):
        np.testing.assert_allclose(
            derived[column], psb_twiss[column].to_numpy(dtype=float), atol=1e-12
        )


def test_follows_the_measurement_rather_than_the_model(psb_twiss):
    """A measured D differing from the model must move the derived D'.

    The round-trip above is also passed by simply copying the model's ``D'``,
    so this is the test that separates a derivation from a copy. ``D'`` is
    affine in the measured ``D``, so perturbing ``D`` by ``k`` times a fixed
    shape must move ``D'`` by exactly ``k`` times a fixed response -- checked
    here without restating the formula under test.
    """
    tunes = _tunes(psb_twiss)
    baseline = derive_momentum_dispersion(psb_twiss, psb_twiss, tunes=tunes)

    # Additive, not multiplicative: this model's vertical dispersion is
    # identically zero, so scaling it would perturb nothing and the vertical
    # half of this test would pass vacuously.
    shape = 1e-2 * np.cos(np.linspace(0.0, 2.0 * np.pi, len(psb_twiss), endpoint=False))

    def derived_at(scale: float) -> dict[str, np.ndarray]:
        perturbed = psb_twiss.copy()
        for column in ("dx", "dy"):
            perturbed[column] = psb_twiss[column].to_numpy(dtype=float) + scale * shape
        return derive_momentum_dispersion(perturbed, psb_twiss, tunes=tunes)

    one, two = derived_at(1.0), derived_at(2.0)
    for column in ("dpx", "dpy"):
        first = one[column] - baseline[column]
        second = two[column] - baseline[column]
        # Doubling the perturbation doubles the response: affine, and driven by
        # the measurement.
        np.testing.assert_allclose(second, 2.0 * first, rtol=1e-9)
        # And the response is real, not a rounding-level wobble around a copy.
        assert np.abs(first).max() > 1e-3


def test_does_not_inherit_the_omc3_index_alignment_bug(psb_twiss):
    """The beta ratio between the two BPMs must actually be applied.

    omc3's ``_calculate_dp`` writes this factor as
    ``np.sqrt(df.loc[shifted, "BETX"] / df.loc[:, "BETX"])`` where both operands
    are Series over the *same* label set, so pandas aligns them and the ratio
    collapses to exactly 1. The factor is silently dropped. It cancels out of
    the final ratio when the measurement equals the model, which is why the two
    implementations still agree there, but not otherwise -- and on a machine
    whose BPM betas really differ it is not a small term.

    This pins the factor down so a future rewrite cannot reintroduce the same
    silent collapse.
    """
    # Under that collapse both m11 and m12 lose every reference to the *next*
    # BPM's beta, so D' at one BPM becomes entirely independent of its
    # successor's optics. Perturbing only the successor is therefore the sharp
    # discriminator: a correct implementation must move, a collapsed one cannot.
    measured = psb_twiss.copy()
    measured["dx"] = psb_twiss["dx"].to_numpy(dtype=float) + 1e-2

    index = 5
    perturbed_model = psb_twiss.copy()
    beta = perturbed_model["betx"].to_numpy(dtype=float).copy()
    beta[index + 1] *= 1.3
    perturbed_model["betx"] = beta

    baseline = derive_momentum_dispersion(measured, psb_twiss, tunes=_tunes(psb_twiss))
    perturbed = derive_momentum_dispersion(measured, perturbed_model, tunes=_tunes(psb_twiss))
    assert abs(perturbed["dpx"][index] - baseline["dpx"][index]) > 1e-6


def test_propagates_measurement_errors(psb_twiss):
    """The uncertainty must come from the two BPMs the answer is built from."""
    measured = psb_twiss.copy()
    measured["dx_err"] = 1e-3
    measured["dy_err"] = 1e-3
    derived = derive_momentum_dispersion(measured, psb_twiss, tunes=_tunes(psb_twiss))

    for column in ("dpx_err", "dpy_err"):
        assert np.all(derived[column] > 0.0)
        assert np.all(np.isfinite(derived[column]))

    # Errors scale linearly with the input error, since D enters linearly.
    doubled = measured.copy()
    doubled["dx_err"] = 2e-3
    doubled["dy_err"] = 2e-3
    scaled = derive_momentum_dispersion(doubled, psb_twiss, tunes=_tunes(psb_twiss))
    np.testing.assert_allclose(scaled["dpx_err"], 2.0 * derived["dpx_err"], rtol=1e-12)


def test_absent_error_column_gives_zero_uncertainty(psb_twiss):
    derived = derive_momentum_dispersion(psb_twiss, psb_twiss, tunes=_tunes(psb_twiss))
    np.testing.assert_allclose(derived["dpx_err"], 0.0, atol=0.0)


def test_rejects_a_degenerate_phase_advance(psb_twiss):
    """A pair near a half-integer advance must raise, not return a huge number."""
    model = psb_twiss.copy()
    # Put the second BPM at the same phase as the first, so sin(dphi) = 0.
    mu1 = model["mu1"].to_numpy(dtype=float).copy()
    mu1[1] = mu1[0]
    model["mu1"] = mu1

    with pytest.raises(ValueError, match="too close to a multiple of pi"):
        derive_momentum_dispersion(model, model, tunes=_tunes(psb_twiss))

    assert MIN_SIN_PHASE_ADVANCE > 0.0


def test_rejects_a_frame_that_is_not_in_ring_order(psb_twiss):
    shuffled = psb_twiss.iloc[::-1]
    with pytest.raises(ValueError, match="ring-ordered"):
        derive_momentum_dispersion(shuffled, shuffled, tunes=_tunes(psb_twiss))


def test_rejects_misaligned_frames(psb_twiss):
    with pytest.raises(ValueError, match="share one ring-ordered index"):
        derive_momentum_dispersion(psb_twiss.iloc[:-1], psb_twiss, tunes=_tunes(psb_twiss))


def test_rejects_a_single_bpm(psb_twiss):
    with pytest.raises(ValueError, match="at least two BPMs"):
        derive_momentum_dispersion(psb_twiss.iloc[:1], psb_twiss.iloc[:1], tunes=_tunes(psb_twiss))
