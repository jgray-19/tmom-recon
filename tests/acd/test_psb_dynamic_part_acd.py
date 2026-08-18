"""Does the reconstruction recover the *dynamic* (betatron) part on its own?

Motivation
----------
When fitting quadrupole errors from an AC-dipole excitation, the useful signal is
the **driven betatron oscillation** — its amplitude and phase at each BPM are set
by the optics, i.e. by the quadrupole gradients. The *static* closed orbit is set
by dipole errors and quadrupole misalignments, which are a different (and, on the
PSB, separately correctable) error family.

Those two parts do not enter an optimiser on an equal footing. In the ``xy``
scenario below the static closed-orbit angle at the downstream BPM is ``1.7e-3``
rad while the driven oscillation is only ``8.7e-5`` rad RMS — a factor ~19. Any
chi-square built on the *absolute* momenta is therefore dominated by the static
part, and a fit will happily trade a large optics error for a small orbit
improvement. Hence a "dynamic part only" mode: strip the closed orbit from both
planes and fit what is left.

What is tested
--------------
The machine (xsuite tracking line) is given a distorted closed orbit — bend field
errors in ``x``, vertical quadrupole misalignments in ``y`` — while the MAD-NG
reconstruction model stays **nominal**, so its twiss closed orbit is ~zero. This
is the real PSB situation: the orbit is in the data and not in the model.

The quantity asserted on is the *dynamic part*, defined as the turn-series with
its own flat-top turn-mean removed::

    dynamic(a) = a - mean(a)

applied identically to the tracked truth and to the reconstruction. Note this is
**not** the "betatron R^2" diagnostic: that mean-removes only the truth and
compares it against a reconstruction that still carries its DC term, so it
reports large negative R^2 even when the dynamic part is perfect. Removing the
mean from both sides is the comparison that matters for a dynamic-part fit,
because a dynamic-part objective would itself mean-remove.

What is checked
---------------
1. :func:`test_dynamic_part_is_invariant_to_closed_orbit_handling` — the dynamic
   part is *bit-identical* whether or not the closed orbit is removed from the
   data, because both the removal and the betatron reconstruction are linear in
   the data. This is the guarantee the whole mode rests on.
2. :func:`test_dynamic_part_is_robust_to_a_wrong_pt` — ``pt`` is the one quantity
   the mode cannot subtract away, so how accurately it must be known.
3. :func:`test_sextupole_feed_down_couples_static_orbit_into_dynamic_part` — the
   boundary condition: with sextupoles powered the orbit genuinely changes the
   driven optics and no reconstruction choice can undo it.
4. :func:`test_single_plane_handling_leaves_other_plane_static_dominated` — the
   "both planes or neither" claim: handling only ``x`` leaves the *vertical*
   static orbit in the result, and it dominates a combined objective.
5. :func:`test_acd_kick_dc_offset_measures_the_unmodelled_closed_orbit` — the
   fitted DC kick is zero in truth and is a calibrated measure of unmodelled
   closed orbit in the absolute frame.
6. :func:`test_dynamic_part_survives_the_ac_dipole_cleaning` — the pipeline runs
   with the harmonic cleaning on, and the marker rows the optimiser consumes are
   *always* the cleaned ones, so the dynamic part is checked through that path
   too, not just through the raw BPM-pair reconstruction.
7. :func:`test_dc_offset_collapses_in_dynamic_part_mode` — what (5) becomes once
   the orbit is removed from the data: it collapses, and therefore stops being a
   bend diagnostic. That is the cost of the mode, so it is pinned down.

One closed-orbit mechanism appears below: ``psb_md``'s ``--orbit-mode dynamic``
subtracts the orbit from the *input* data and never restores it, giving a purely
dynamic state. That is the ``remove_planes`` argument of :func:`_run`; leaving it
unset references the reconstruction to the (nominal) model twiss orbit instead.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES, build_psb_tracking_setup
from tests.reference_co import zero_momentum_reference
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_pz
from tmom_recon.physics.closed_orbit import estimate_closed_orbit

from .acd_test_helpers import acd_state_marker_names

LOGGER = logging.getLogger(__name__)

BEND_ERROR_RMS = 8e-4
BEND_ERROR_SEED = 7
QUAD_MISALIGN_Y_RMS = 2e-4  # 0.2 mm
QUAD_MISALIGN_SEED = 11
ACD_DRIVEN_TUNES = (0.18, DRIVEN_TUNES[1])
# Integrated sextupole strength for the feed-down check. The saved sequence has
# every chromaticity sextupole at zero (no-multipole campaign), so any non-zero
# value makes the lattice non-linear; this one is large enough for the feed-down
# to be unambiguous.
SEXTUPOLE_K2L = 5.0

# Relative RMS error tolerated on the BPM dynamic part. The wrong-pt sweep
# reaches 7.8e-4 at a deliberately large 30% momentum error; retain modest
# headroom while keeping static-orbit leakage detectable.
BPM_DYNAMIC_TOL = 8.5e-4
MARKER_DYNAMIC_TOL = 1e-3

# mode -> (bend_error_rms, quad_misalign_y_rms, perturbed planes)
_MODES = {
    "x": (BEND_ERROR_RMS, 0.0, "x"),
    "y": (0.0, QUAD_MISALIGN_Y_RMS, "y"),
    "xy": (BEND_ERROR_RMS, QUAD_MISALIGN_Y_RMS, "xy"),
}

# Off-momentum offset used for the dispersive variants. Matches
# [test_psb_closed_orbit_acd.py]; at ``Dx ~ -2.89 m`` it puts an ~8.7 mm dispersive
# orbit on top of the ~5.5 mm bend-error orbit, i.e. the dispersive part *dominates*.
OFF_MOMENTUM_DELTA_P = 3.0e-3


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


def _dynamic(values) -> np.ndarray:
    """Turn-series with its own mean removed — the driven (betatron) part.

    The AC-dipole flat-top oscillation has ~zero mean over the flat-top turns, so
    the mean is the static closed-orbit value and the remainder is the dynamic
    part. Applied to truth and reconstruction alike so the comparison is between
    two quantities of the same kind.
    """
    array = np.asarray(values, dtype=float)
    return array - float(np.mean(array))


def _dynamic_error(true, pred) -> float:
    """Relative RMS error of the dynamic part of *pred* against that of *true*."""
    true_dynamic = _dynamic(true)
    return _rms(true_dynamic - _dynamic(pred)) / _rms(true_dynamic)


def _build_setup(mode: str, data_dir, *, delta_p: float = 0.0):
    """Track *mode*'s perturbations at *delta_p*, with a model that ignores them.

    The model twiss is on-momentum in both cases — the dispersive orbit is *not*
    in the twiss ``x``/``y`` columns, it is carried as ``pt * d`` and subtracted
    from the data mean by
    :func:`tmom_recon.physics.closed_orbit.estimate_closed_orbit`. So off momentum
    the static orbit in the *data* grows by the dispersive contribution (~8.7 mm
    at ``Dx = -2.89 m``, well above the ~5.5 mm bend-error orbit) while the model
    accounts for it through a single scalar ``pt``. That split is the point of the
    off-momentum variants: the static part is now much larger and only partly
    modelled, and the driven optics are chromatically shifted as well.
    """
    bend_rms, quad_rms, planes = _MODES[mode]
    setup = build_psb_tracking_setup(
        data_dir,
        delta_p=delta_p,
        driven_tunes=ACD_DRIVEN_TUNES,
        bend_error_rms=bend_rms,
        bend_error_seed=BEND_ERROR_SEED,
        apply_bend_errors_to_model=False,
        quad_misalign_y_rms=quad_rms,
        quad_misalign_seed=QUAD_MISALIGN_SEED,
    )
    # The model must not know about the perturbations: this is the whole point of
    # the scenario, and if the model twiss ever acquired the orbit the test would
    # silently stop exercising the data-mean path.
    assert float(setup["tws"]["x"].abs().max()) < 1e-5
    assert float(setup["tws"]["y"].abs().max()) < 1e-5
    # Off momentum the *data* must actually carry the dispersive orbit, otherwise
    # the off-momentum variants would be duplicates of the on-momentum ones.
    tracking_df = setup["tracking_df"]
    data_orbit = float(tracking_df.groupby("name", observed=True)["x"].mean().abs().max())
    LOGGER.info("[mode=%s dp=%.1e] max |data mean x| = %.3f mm", mode, delta_p, 1e3 * data_orbit)
    if delta_p != 0.0:
        # The dispersive orbit alone is ~4.5 mm; with the bend errors on top it
        # reaches ~10 mm. Bound below the smaller of the two.
        assert data_orbit > 4e-3, "expected a large dispersive orbit in the data"
    return setup, planes


def _run(
    setup,
    *,
    pt: float | None = None,
    remove_planes: str | None = None,
):
    """Reconstruct, optionally in ``psb_md``'s dynamic-part frame.

    With *remove_planes* unset the closed orbit comes from the model twiss, which
    here is nominal — so the machine's orbit stays in the data and the output is
    an absolute state carrying it.

    *remove_planes* is what ``psb_md`` does for ``--orbit-mode dynamic``: the
    orbit is subtracted from the *input* data and never restored, so the output
    is purely dynamic. Reproduced here exactly as
    ``psb_md.orbit_frame.ClosedOrbitFrame.turn_mean(...).subtract_from`` does it
    -- same ``estimate_closed_orbit`` estimator, dispersive ``pt*d`` preserved.
    """
    model = setup["model"]
    before_marker, after_marker = acd_state_marker_names(model)
    bpm_df = setup["tracking_df"]
    bpm_df = bpm_df.loc[~bpm_df["name"].isin([before_marker, after_marker])].copy()
    effective_pt = model.pt if pt is None else pt
    if remove_planes:
        closed_orbit = estimate_closed_orbit(bpm_df, setup["tws"], pt_est=effective_pt)
        for plane in remove_planes:
            bpm_df[plane] = bpm_df[plane] - bpm_df["name"].map(closed_orbit[plane]).astype(
                float
            ).fillna(0.0)
    return calculate_pz(
        bpm_df,
        reference=zero_momentum_reference(bpm_df),
        model_details=ModelDetails(accelerator=model.accelerator, pt=effective_pt),
        use_dispersion=True,
        acd=ACDipoleConfig(
            ac_dipole_marker=ACD_ELEMENT,
            driven_tunes=ACD_DRIVEN_TUNES,
        ),
        acd_only=True,
    )


def _bpm_dynamic_errors(result, tracking_df, *, cleaned: bool) -> dict[str, float]:
    """Relative dynamic-part error of the reconstructed BPM momenta, per plane/side."""
    summary = result.attrs["summary"]
    suffix = "_cleaned" if cleaned else ""
    errors: dict[str, float] = {}
    for side in ("upstream", "downstream"):
        bpm = result.attrs[f"bpm_{side}"]
        truth = (
            tracking_df.loc[tracking_df["name"] == bpm, ["turn", "px", "py"]]
            .sort_values("turn")
            .rename(columns={"px": "px_true", "py": "py_true"})
        )
        merged = summary.merge(truth, on="turn", how="inner")
        for plane in ("px", "py"):
            errors[f"{plane}_{side}"] = _dynamic_error(
                merged[f"{plane}_true"].to_numpy(dtype=float),
                merged[f"{plane}_bpm_{side}{suffix}"].to_numpy(dtype=float),
            )
    return errors


def _marker_dynamic_errors(result, tracking_df, model) -> dict[str, float]:
    """Relative dynamic-part error of the marker states — the optimiser's ICs."""
    state_rows = result.assign(name=result["name"].astype(str).str.upper())
    errors: dict[str, float] = {}
    for marker in acd_state_marker_names(model):
        truth = (
            tracking_df.loc[tracking_df["name"] == marker, ["turn", "x", "px", "y", "py"]]
            .sort_values("turn")
            .rename(columns={c: f"{c}_true" for c in ("x", "px", "y", "py")})
        )
        rows = state_rows.loc[state_rows["name"] == marker].merge(truth, on="turn", how="inner")
        assert len(rows) == len(truth), f"{marker}: lost turns merging against truth"
        label = "before" if marker.endswith("_BEFORE") else "after"
        for coord in ("x", "px", "y", "py"):
            errors[f"{coord}_{label}"] = _dynamic_error(
                rows[f"{coord}_true"].to_numpy(dtype=float),
                rows[coord].to_numpy(dtype=float),
            )
    return errors


def _static_vs_dynamic(tracking_df, bpm: str) -> dict[str, tuple[float, float]]:
    """``{plane: (|static angle|, dynamic RMS)}`` in the tracked truth at *bpm*."""
    rows = tracking_df.loc[tracking_df["name"] == bpm].sort_values("turn")
    out = {}
    for plane in ("px", "py"):
        values = rows[plane].to_numpy(dtype=float)
        out[plane] = (abs(float(np.mean(values))), _rms(_dynamic(values)))
    return out


@pytest.mark.slow
@pytest.mark.parametrize(
    "delta_p",
    [
        pytest.param(0.0, id="on_momentum"),
        pytest.param(OFF_MOMENTUM_DELTA_P, id="off_momentum"),
    ],
)
def test_dynamic_part_is_invariant_to_closed_orbit_handling(delta_p, data_dir) -> None:
    """The dynamic part does not depend on how the closed orbit is handled.

    Both the closed-orbit removal/restoration and the BPM-pair momentum
    reconstruction are *linear* in the data, so a static (DC) offset in the input
    maps to a static offset in the output and cannot reach the AC part. Removing
    the closed orbit from the data therefore cancels exactly on the dynamic part.

    This is the guarantee a dynamic-part optimisation rests on: the driven
    oscillation it fits is the same whether or not the orbit was handled, so an
    unmodelled orbit cannot bias the optics fit. It is asserted to machine
    precision because anything looser would not distinguish "cancels exactly"
    from "cancels well" — and only the former makes the mode safe.

    Note this holds because the lattice is linear (the PSB sextupoles are
    unpowered here, matching the no-multipole campaign). See
    :func:`test_sextupole_feed_down_couples_static_orbit_into_dynamic_part` for
    what happens when it is not.

    Off momentum the two configurations are *further* apart than on momentum, so
    this is the sharper version of the same check. Leaving the orbit in place
    references the reconstruction to the model twiss orbit, which off momentum is
    the ~8.7 mm dispersive orbit; removing it from the data references it to the
    dispersive orbit *plus* the bend-error and misalignment orbit. Two genuinely
    different, both large, static references — and the dynamic part must still
    come out identical.
    """
    setup, planes = _build_setup("xy", data_dir, delta_p=delta_p)
    handled = _bpm_dynamic_errors(
        _run(setup, remove_planes=planes), setup["tracking_df"], cleaned=False
    )
    unhandled = _bpm_dynamic_errors(_run(setup), setup["tracking_df"], cleaned=False)
    LOGGER.info("[dp=%.1e] closed orbit removed from the data: %s", delta_p, _fmt(handled))
    LOGGER.info("[dp=%.1e] closed orbit from model twiss: %s", delta_p, _fmt(unhandled))

    assert handled.keys() == unhandled.keys()
    for key, error in handled.items():
        assert error == pytest.approx(unhandled[key], rel=1e-9), (
            f"{key}: dynamic-part error changed with closed-orbit handling "
            f"({error:.12e} vs {unhandled[key]:.12e}); the betatron reconstruction "
            "is supposed to be exactly linear in the data"
        )


@pytest.mark.slow
def test_dynamic_part_is_robust_to_a_wrong_pt(data_dir) -> None:
    """How well does ``pt`` have to be known for a dynamic-part fit off momentum?

    The one thing a dynamic-part mode cannot subtract away is ``pt`` itself: it
    sets the dispersive orbit removed from the data mean *and* the momentum the
    reconstruction model is built at, so a wrong ``pt`` shifts the chromatic
    optics as well as the static reference. Since ``pt`` comes from the RF rather
    than from the fit, the question is how much error is tolerable.

    Tracking is at ``delta_p = 3e-3``; the reconstruction is handed a ``pt`` that
    is wrong by up to 30%, i.e. up to ``9e-4`` in ``dp/p`` — far worse than an RF
    measurement. The dynamic part is compared against the same tracked truth in
    every case.
    """
    setup, planes = _build_setup("xy", data_dir, delta_p=OFF_MOMENTUM_DELTA_P)
    accelerator = setup["model"].accelerator
    tracking_df = setup["tracking_df"]

    errors: dict[float, dict[str, float]] = {}
    for fraction in (0.0, 0.1, 0.3):
        pt = accelerator.dp2pt(OFF_MOMENTUM_DELTA_P * (1.0 + fraction))
        errors[fraction] = _bpm_dynamic_errors(
            _run(setup, pt=pt, remove_planes=planes), tracking_df, cleaned=False
        )
        LOGGER.info(
            "pt error %+.0f%% (dp/p off by %+.1e): %s",
            100 * fraction,
            OFF_MOMENTUM_DELTA_P * fraction,
            _fmt(errors[fraction]),
        )

    for fraction, dynamic_errors in errors.items():
        for key, error in dynamic_errors.items():
            assert error < BPM_DYNAMIC_TOL, (
                f"{key} dynamic error {error:.3e} with pt wrong by {100 * fraction:.0f}%"
            )


@pytest.mark.slow
def test_sextupole_feed_down_couples_static_orbit_into_dynamic_part(data_dir) -> None:
    """With sextupoles powered, the static orbit *physically* alters the dynamic part.

    The invariance above is a property of a linear lattice. Power the
    chromaticity sextupoles and a static horizontal orbit ``x_co`` through them
    acts as a gradient error ``dk1l = k2l * x_co``, changing the driven optics
    themselves. The dynamic part then genuinely depends on the orbit — no
    reconstruction choice can undo that, because the machine really did change.

    This compares two *tracked* runs directly (no reconstruction, no model), so
    it isolates the physics: same sextupole setting, with and without the bend
    errors that create the orbit.

    The sextupoles-off baseline is small but **not** zero (~7e-3): a relative
    bend field error changes ``k0``, and with exact bend maps that perturbs the
    weak focusing as well as the orbit. That residual is the bends' own optics
    change, not feed-down, and it bounds how cleanly this scenario can separate
    the two. The assertion is therefore that the sextupole-on effect *dominates*
    it by a wide margin, not that the baseline vanishes.

    The consequence for a dynamic-part fit: it is clean only while the
    sextupoles are off. With them on, the orbit must be either corrected or
    included in the model before the dynamic part means anything about the
    quadrupoles — the static orbit is then not a nuisance to be subtracted but a
    parameter the optics genuinely depends on.
    """

    def dynamic_at_bpms(*, k2l: float, bend_rms: float) -> dict[str, np.ndarray]:
        setup = build_psb_tracking_setup(
            data_dir,
            delta_p=0.0,
            driven_tunes=ACD_DRIVEN_TUNES,
            bend_error_rms=bend_rms,
            bend_error_seed=BEND_ERROR_SEED,
            apply_bend_errors_to_model=False,
            sextupole_k2l=k2l,
        )
        tracking_df = setup["tracking_df"]
        before, after = acd_state_marker_names(setup["model"])
        bpms = sorted(
            set(tracking_df["name"].unique()) - {before, after},
            key=str,
        )
        return {
            bpm: _dynamic(
                tracking_df.loc[tracking_df["name"] == bpm]
                .sort_values("turn")["x"]
                .to_numpy(dtype=float)
            )
            for bpm in bpms
        }

    def relative_change(k2l: float) -> float:
        flat = dynamic_at_bpms(k2l=k2l, bend_rms=0.0)
        distorted = dynamic_at_bpms(k2l=k2l, bend_rms=BEND_ERROR_RMS)
        return max(
            _rms(distorted[bpm] - flat[bpm]) / _rms(flat[bpm]) for bpm in flat if bpm in distorted
        )

    linear = relative_change(0.0)
    non_linear = relative_change(SEXTUPOLE_K2L)
    LOGGER.info(
        "dynamic-part change from the static orbit: sextupoles off=%.3e, on (k2l=%.1f)=%.3e",
        linear,
        SEXTUPOLE_K2L,
        non_linear,
    )

    assert linear < 0.02, (
        f"the sextupoles-off baseline is {linear:.3e}, larger than the ~7e-3 expected "
        "from the bend errors' own weak-focusing change; the scenario no longer "
        "separates feed-down from direct optics perturbation"
    )
    assert non_linear > 10 * max(linear, 1e-12), (
        f"powering the sextupoles did not couple the static orbit into the dynamic "
        f"part ({non_linear:.3e} vs {linear:.3e}); either k2l is too small to matter "
        "or the sextupoles are not actually being powered"
    )


@pytest.mark.slow
def test_single_plane_handling_leaves_other_plane_static_dominated(data_dir) -> None:
    """Handling one plane only is not a coherent objective for the other plane.

    Both planes are perturbed but only ``x`` is given the data-mean closed orbit.
    The horizontal dynamic part comes out fine — the planes are uncoupled, so
    there is no leakage — but the *vertical* result still carries its full static
    angle. An optimiser summing residuals over both planes would then be driven by
    the vertical static orbit, which is exactly the failure mode a dynamic-part fit
    is meant to avoid. Hence: dynamic in both planes, or absolute in both planes.
    """
    setup, _ = _build_setup("xy", data_dir)
    result = _run(setup, remove_planes="x")
    tracking_df = setup["tracking_df"]
    summary = result.attrs["summary"]

    dynamic_errors = _bpm_dynamic_errors(result, tracking_df, cleaned=False)
    LOGGER.info("x-only handling, dynamic errors: %s", _fmt(dynamic_errors))

    # Horizontal is unharmed by the untreated vertical plane (no coupling).
    for side in ("upstream", "downstream"):
        assert dynamic_errors[f"px_{side}"] < BPM_DYNAMIC_TOL, (
            f"px_{side} dynamic error {dynamic_errors[f'px_{side}']:.3e} — the "
            "untreated vertical plane should not affect the horizontal one"
        )

    # But the vertical absolute residual is static-dominated: its mean (the DC
    # error) is large compared with the dynamic scatter about that mean.
    for side in ("upstream", "downstream"):
        bpm = result.attrs[f"bpm_{side}"]
        truth = (
            tracking_df.loc[tracking_df["name"] == bpm, ["turn", "py"]]
            .sort_values("turn")
            .rename(columns={"py": "py_true"})
        )
        merged = summary.merge(truth, on="turn", how="inner")
        residual = merged["py_true"].to_numpy(dtype=float) - merged[f"py_bpm_{side}"].to_numpy(
            dtype=float
        )
        static_error = abs(float(np.mean(residual)))
        dynamic_error = _rms(_dynamic(residual))
        LOGGER.info(
            "x-only handling, py_%s residual: static=%.3e dynamic=%.3e (ratio %.1f)",
            side,
            static_error,
            dynamic_error,
            static_error / max(dynamic_error, 1e-30),
        )
        assert static_error > dynamic_error, (
            f"py_{side}: expected the untreated vertical plane to be dominated by "
            f"its static error (static {static_error:.3e}, dynamic {dynamic_error:.3e})"
        )


@pytest.mark.slow
def test_acd_kick_dc_offset_measures_the_unmodelled_closed_orbit(data_dir) -> None:
    """The DC term of the AC-dipole kick fit should be zero, and is not if the orbit is.

    An AC dipole imparts a purely oscillating kick, so the true
    ``dpx = px_after - px_before`` has *exactly* zero mean — confirmed here
    against the tracked markers at the 1e-21 level, for every error setting.

    A non-zero fitted DC term is therefore never physical. It appears when the
    model fails to reproduce the machine's closed orbit *between the two
    markers*: the static ``px`` inferred on each side of the AC dipole is then
    biased by a different amount, and the difference survives as a DC kick. Two
    controls pin that down — with no errors at all, and with the model given the
    same bend errors as the machine, the DC term returns to ~1e-9.

    Crucially the DC term is **linear in the closed orbit** (asserted below), so
    it is a calibrated, free diagnostic: it says how much orbit the model is
    still missing, and it should collapse toward zero as the bend fit improves.
    In the horizontal plane it is also *large*: at 0.08% RMS bend errors the DC
    term is ~43x the AC kick amplitude it sits on. That is the same asymmetry
    that motivates fitting the dynamic part.
    """

    def dc_offset(bend_rms: float, *, match_model: bool) -> tuple[float, float, float, float]:
        """``(fitted dpx DC, fitted dpx amplitude, max |x closed orbit|, true DC)``."""
        setup = build_psb_tracking_setup(
            data_dir,
            delta_p=0.0,
            driven_tunes=ACD_DRIVEN_TUNES,
            bend_error_rms=bend_rms,
            bend_error_seed=BEND_ERROR_SEED,
            apply_bend_errors_to_model=match_model,
        )
        model = setup["model"]
        tracking_df = setup["tracking_df"]
        before, after = acd_state_marker_names(model)
        bpm_df = tracking_df.loc[~tracking_df["name"].isin([before, after])].copy()
        orbit = float(bpm_df.groupby("name", observed=True)["x"].mean().abs().max())
        strengths = (
            {f"{name.upper()}.k0": value for name, value in setup["bend_k0"].items()}
            if (setup["bend_k0"] and match_model)
            else None
        )
        result = calculate_pz(
            bpm_df,
            reference=zero_momentum_reference(bpm_df),
            model_details=ModelDetails(
                accelerator=model.accelerator, pt=model.pt, magnet_strengths=strengths
            ),
            acd=ACDipoleConfig(
                ac_dipole_marker=ACD_ELEMENT,
                driven_tunes=ACD_DRIVEN_TUNES,
            ),
            acd_only=True,
        )

        # The true kick is a difference of the two tracked marker momenta.
        def at(name):
            return tracking_df.loc[tracking_df["name"] == name].sort_values("turn")

        true_dc = float(np.mean(at(after)["px"].to_numpy(float) - at(before)["px"].to_numpy(float)))
        return result.attrs["dpx_offset"], result.attrs["dpx_amplitude"], orbit, true_dc

    # The AC dipole cannot produce a DC kick, whatever the machine errors are.
    for bend_rms, match_model in ((0.0, False), (BEND_ERROR_RMS, False), (BEND_ERROR_RMS, True)):
        *_, true_dc = dc_offset(bend_rms, match_model=match_model)
        assert abs(true_dc) < 1e-15, (
            f"the tracked truth has a DC kick of {true_dc:.3e} rad at bend_rms="
            f"{bend_rms:.1e}; the AC dipole is supposed to be purely oscillating"
        )

    # Controls: no errors, and a model that matches the machine, both give ~0 —
    # provided the closed orbit is taken from the (correct) twiss.
    nominal_dc, nominal_amp, _, _ = dc_offset(0.0, match_model=False)
    matched_dc, matched_amp, _, _ = dc_offset(BEND_ERROR_RMS, match_model=True)
    LOGGER.info(
        "dpx DC offset: no errors=%.3e (amp %.3e), model matched to errors=%.3e (amp %.3e)",
        nominal_dc,
        nominal_amp,
        matched_dc,
        matched_amp,
    )
    assert abs(nominal_dc) < 1e-8, (
        f"an error-free machine produced a DC kick term of {nominal_dc:.3e} rad"
    )
    assert abs(matched_dc) < 1e-8, (
        f"the model was given the machine's own bend errors, so its closed orbit "
        f"matches and the DC term should vanish; got {matched_dc:.3e} rad"
    )

    # Linear in the closed orbit, so it is usable as a calibrated diagnostic.
    scale = []
    for bend_rms in (2e-4, 4e-4, 8e-4):
        dc, amp, orbit, _ = dc_offset(bend_rms, match_model=False)
        LOGGER.info(
            "bend_rms=%.1e -> max|x_co|=%.3f mm, dpx DC=%+.3e rad (%.1fx the AC amplitude)",
            bend_rms,
            orbit * 1e3,
            dc,
            abs(dc) / amp,
        )
        scale.append(abs(dc) / orbit)
    assert scale[0] == pytest.approx(scale[1], rel=1e-3)
    assert scale[0] == pytest.approx(scale[2], rel=1e-3), (
        f"the DC term is not proportional to the closed orbit (rad/m: {scale}); it "
        "cannot be read as a linear measure of the unmodelled orbit"
    )
    # And it dominates the AC kick it sits on at realistic PSB bend errors.
    dc, amp, _, _ = dc_offset(BEND_ERROR_RMS, match_model=False)
    assert abs(dc) > 10 * amp, (
        f"expected the DC term ({dc:.3e}) to dwarf the AC kick amplitude ({amp:.3e})"
    )


def _fmt(errors: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.3e}" for key, value in sorted(errors.items()))


@pytest.mark.slow
@pytest.mark.parametrize(
    "delta_p",
    [
        pytest.param(0.0, id="on_momentum"),
        pytest.param(OFF_MOMENTUM_DELTA_P, id="off_momentum"),
    ],
)
def test_dynamic_part_survives_the_ac_dipole_cleaning(delta_p, data_dir) -> None:
    """The dynamic part must survive the AC-dipole harmonic cleaning, not just the
    raw BPM-pair reconstruction.

    The other tests here assert on the *raw* ``px_bpm_*`` columns, which is the
    plain neighbour-pair reconstruction. That is not what the pipeline actually
    consumes: ``psb_md`` runs with ``use_acd_cleaning=True``, and the AC-dipole
    marker rows written to the parquet -- the initial conditions the optimiser
    tracks from -- are built from the *cleaned* marker states
    (``fit.cleaned_upstream`` / ``fit.cleaned_downstream``), never the raw ones.
    So the cleaning sits directly in the dynamic-part path and has to be checked
    there.

    The cleaning replaces the measured turn series with a single fitted harmonic
    at the driven tune, which is exactly a model of the dynamic part; the concern
    is whether it distorts amplitude or phase once a large unmodelled static orbit
    is present. Both raw and cleaned are compared against the same tracked truth,
    in the true dynamic-part frame (``remove_planes``), so the comparison is like
    for like.
    """
    setup, planes = _build_setup("xy", data_dir, delta_p=delta_p)
    result = _run(setup, remove_planes=planes)
    tracking_df = setup["tracking_df"]

    raw = _bpm_dynamic_errors(result, tracking_df, cleaned=False)
    cleaned = _bpm_dynamic_errors(result, tracking_df, cleaned=True)
    markers = _marker_dynamic_errors(result, tracking_df, setup["model"])
    LOGGER.info("[dp=%.1e] BPM dynamic errors, raw:     %s", delta_p, _fmt(raw))
    LOGGER.info("[dp=%.1e] BPM dynamic errors, cleaned: %s", delta_p, _fmt(cleaned))
    LOGGER.info("[dp=%.1e] marker dynamic errors (always cleaned): %s", delta_p, _fmt(markers))

    for key, error in cleaned.items():
        assert error < BPM_DYNAMIC_TOL, (
            f"cleaned BPM {key} dynamic error {error:.3e} (raw {raw[key]:.3e}); "
            "the harmonic cleaning is distorting the driven motion it is supposed "
            "to isolate"
        )
    for key, error in markers.items():
        assert error < MARKER_DYNAMIC_TOL, f"marker {key} dynamic error {error:.3e}"

    # The kick fit itself must still describe the data: a poor R^2 would mean the
    # cleaned harmonic is not the motion that is actually there.
    assert result.attrs["dpx_r2"] > 0.99
    assert result.attrs["dpy_r2"] > 0.99


@pytest.mark.slow
@pytest.mark.parametrize(
    "delta_p",
    [
        pytest.param(0.0, id="on_momentum"),
        pytest.param(OFF_MOMENTUM_DELTA_P, id="off_momentum"),
    ],
)
def test_dc_offset_collapses_in_dynamic_part_mode(delta_p, data_dir) -> None:
    """What the AC-dipole fit's DC offset means once the orbit is removed.

    :func:`test_acd_kick_dc_offset_measures_the_unmodelled_closed_orbit` shows the
    fitted ``dpx`` DC term is a calibrated measure of unmodelled closed orbit:
    ~6e-5 rad per mm, tens of times the AC amplitude at realistic bend errors.

    That diagnostic works because the closed orbit is *in* the reconstruction.
    ``--orbit-mode dynamic`` removes it from the input data and never restores it,
    so there is no static orbit left to be mis-modelled and the offset must
    collapse toward zero.

    This is the honest cost of the mode: **in dynamic-part mode the DC offset
    stops being a bend diagnostic.** A near-zero offset there says only that the
    orbit was successfully removed; it says nothing about the bends. To read the
    bend quality, reconstruct once without the flag. Asserted both ways here so
    neither reading can be lost.
    """
    setup, planes = _build_setup("xy", data_dir, delta_p=delta_p)

    absolute = _run(setup)
    dynamic = _run(setup, remove_planes=planes)

    absolute_dc = float(absolute.attrs["dpx_offset"])
    dynamic_dc = float(dynamic.attrs["dpx_offset"])
    amplitude = float(dynamic.attrs["dpx_amplitude"])
    LOGGER.info(
        "[dp=%.1e] dpx DC offset: absolute frame=%+.3e rad (%.1fx amplitude), "
        "dynamic-part frame=%+.3e rad (%.2fx amplitude); amplitude=%.3e",
        delta_p,
        absolute_dc,
        abs(absolute_dc) / amplitude,
        dynamic_dc,
        abs(dynamic_dc) / amplitude,
        amplitude,
    )

    # In the absolute frame the offset is the orbit diagnostic: large.
    assert abs(absolute_dc) > 10 * amplitude, (
        f"expected the absolute-frame DC offset ({absolute_dc:.3e}) to dwarf the "
        f"AC kick amplitude ({amplitude:.3e}); it is the unmodelled-orbit measure"
    )
    # In the dynamic-part frame it collapses: nothing static is left to mis-model.
    assert abs(dynamic_dc) < abs(absolute_dc) / 100, (
        f"the DC offset did not collapse when the orbit was removed from the data "
        f"({dynamic_dc:.3e} vs {absolute_dc:.3e}); either the removal is leaving a "
        "static orbit behind or it is introducing one of its own"
    )
    # And it is small against the signal it sits on, so it cannot bias the fit.
    assert abs(dynamic_dc) < amplitude, (
        f"dynamic-part DC offset {dynamic_dc:.3e} is comparable to the AC "
        f"amplitude {amplitude:.3e}; the removed orbit is not clean"
    )


@pytest.mark.slow
def test_dynamic_part_frame_is_insensitive_to_the_orbit_size(data_dir) -> None:
    """Sweeping the *size* of the removed orbit must not move the dynamic part.

    [test_dc_offset_collapses_in_dynamic_part_mode] shows the DC offset collapses
    at one orbit size. That leaves a real worry open, and it is the natural
    explanation for a consistency-guard failure in dynamic-part mode: removing the
    closed orbit from the *data* does not remove it from the *physics*. The
    betatron motion physically happens about the machine's true closed orbit,
    while the reconstruction transports it between the BPMs and the AC-dipole
    marker through a model whose reference orbit is the design (zero) orbit. The
    PSB model uses exact bend maps, so that transport is genuinely non-linear in
    the offset, and the linearity argument the whole mode rests on would break
    down at some orbit amplitude.

    This sweeps the tracked bend errors over a 16x range -- 0.7 mm to 11 mm of
    horizontal closed orbit, the latter well beyond anything the real machine
    shows -- with the model kept nominal throughout, and measures where that
    breakdown sets in.

    It does not set in. Measured: the dynamic-part DC offset is -9.02e-10 rad at
    *every* orbit size (sixth digit unchanged over the sweep) and the driven
    amplitude moves by ~1%, while over the same sweep the absolute-frame offset
    runs 4.1e-5 -> 6.6e-4 rad, exactly proportional to the orbit at 6.001e-5
    rad/mm. So the separation is exact at any orbit the PSB can produce, and a
    guard failure in this frame must be explained by something else.
    """

    def frames(bend_rms: float) -> tuple[float, float, float, float]:
        """``(absolute DC, dynamic DC, dynamic amplitude, max |x closed orbit|)``."""
        setup = build_psb_tracking_setup(
            data_dir,
            delta_p=0.0,
            driven_tunes=ACD_DRIVEN_TUNES,
            bend_error_rms=bend_rms,
            bend_error_seed=BEND_ERROR_SEED,
            apply_bend_errors_to_model=False,
        )
        before, after = acd_state_marker_names(setup["model"])
        bpm_df = setup["tracking_df"]
        orbit = float(
            bpm_df.loc[~bpm_df["name"].isin([before, after])]
            .groupby("name", observed=True)["x"]
            .mean()
            .abs()
            .max()
        )
        absolute = _run(setup)
        dynamic = _run(setup, remove_planes="xy")
        return (
            float(absolute.attrs["dpx_offset"]),
            float(dynamic.attrs["dpx_offset"]),
            float(dynamic.attrs["dpx_amplitude"]),
            orbit,
        )

    # 1e-4 .. 1.6e-3 RMS relative bend error -> ~0.7 .. ~11 mm of closed orbit.
    measured = [frames(bend_rms) for bend_rms in (1e-4, 4e-4, 1.6e-3)]
    for absolute_dc, dynamic_dc, amplitude, orbit in measured:
        LOGGER.info(
            "max|x_co|=%6.3f mm -> absolute dpx DC=%+.3e rad (%.3e rad/mm), "
            "dynamic dpx DC=%+.3e rad (%.2e x amplitude), amplitude=%.3e",
            1e3 * orbit,
            absolute_dc,
            absolute_dc / (1e3 * orbit),
            dynamic_dc,
            abs(dynamic_dc) / amplitude,
            amplitude,
        )

    orbits = [row[3] for row in measured]
    assert orbits[-1] / orbits[0] > 10, (
        f"the sweep no longer spans a wide range of closed orbits ({orbits}); it "
        "cannot show insensitivity to something that barely varies"
    )

    # The control: in the absolute frame the offset tracks the orbit exactly, so
    # the sweep demonstrably *does* change the thing being removed.
    per_mm = [row[0] / row[3] for row in measured]
    assert per_mm[0] == pytest.approx(per_mm[1], rel=1e-3)
    assert per_mm[0] == pytest.approx(per_mm[2], rel=1e-3), (
        f"the absolute-frame DC offset is no longer proportional to the closed "
        f"orbit (rad/m: {per_mm}), so this test's control has gone"
    )

    # The claim: in the dynamic-part frame it does not move at all. Compared
    # against the *absolute* frame's own variation, which is the scale that a
    # leaked orbit would appear on.
    dynamic_dcs = [row[1] for row in measured]
    absolute_spread = abs(measured[-1][0] - measured[0][0])
    dynamic_spread = max(dynamic_dcs) - min(dynamic_dcs)
    assert dynamic_spread < absolute_spread / 1e5, (
        f"the dynamic-part DC offset varied by {dynamic_spread:.3e} rad across a "
        f"{1e3 * orbits[0]:.2f}-{1e3 * orbits[-1]:.2f} mm orbit sweep, against "
        f"{absolute_spread:.3e} rad in the absolute frame. Some of the removed "
        "orbit is reaching the dynamic part -- most likely the BPM<->marker "
        "transport becoming non-linear in the offset"
    )
    # And it stays negligible against the signal, at every orbit size.
    for absolute_dc, dynamic_dc, amplitude, orbit in measured:
        assert abs(dynamic_dc) < 1e-3 * amplitude, (
            f"at {1e3 * orbit:.2f} mm of closed orbit the dynamic-part DC offset "
            f"is {dynamic_dc:.3e} rad against an amplitude of {amplitude:.3e}"
        )

    # The driven amplitude itself must also be orbit-independent: a DC offset
    # that stays put while the amplitude drifts would still corrupt an optics fit.
    amplitudes = [row[2] for row in measured]
    assert amplitudes[0] == pytest.approx(amplitudes[-1], rel=2e-2), (
        f"the driven amplitude moved from {amplitudes[0]:.3e} to "
        f"{amplitudes[-1]:.3e} rad across the orbit sweep; the dynamic part is "
        "not independent of the orbit after all"
    )
