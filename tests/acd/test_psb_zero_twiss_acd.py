"""AC-dipole reconstruction with a distorted data orbit but a *zero-CO* twiss.

This is the counterpart to [test_psb_closed_orbit_acd.py], where the MAD-NG model
*shares* the machine's distorted closed orbit. Here the tracked data carries a
distorted closed orbit (from bend field errors in ``x`` and/or vertical quadrupole
misalignments in ``y``) but the reconstruction model is left **nominal**, so its
twiss closed orbit is ~zero — it does *not* represent the machine orbit. This is
the "a twiss on zero" scenario of the real PSB measurement.

Removing a zero closed orbit would leave the ~mm orbit in the data, biasing the
betatron reconstruction and tripping ``_check_bpm_state_consistency``. Instead the
reconstruction opts in to the *data-mean* closed orbit
(``ACDipoleConfig.data_mean_closed_orbit_planes``), per plane: the per-BPM
turn-mean of the data is used as the closed-orbit position reference (valid
because the AC-dipole oscillation has ~zero mean), and the matching closed-orbit
*angle* is inferred from those positions with the model optics, so the orbit
removed before the betatron reconstruction is restored afterwards as a
consistent state.

The test characterises how good the reconstruction is under that limitation, with
**no measurement noise**, comparing the momenta reconstructed *without* the
harmonic cleaning (raw ``*_bpm_*`` columns) against *with* cleaning
(``*_bpm_*_cleaned``).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES, build_psb_tracking_setup
from tests.reference_co import zero_reference_co
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_pz

from .acd_test_helpers import acd_state_marker_names, r_squared

LOGGER = logging.getLogger(__name__)

BEND_ERROR_RMS = 8e-4
BEND_ERROR_SEED = 7
QUAD_MISALIGN_Y_RMS = 2e-4  # 0.2 mm
QUAD_MISALIGN_SEED = 11
ACD_DRIVEN_TUNES = (0.18, DRIVEN_TUNES[1])

# mode -> (bend_error_rms, quad_misalign_y_rms, data_mean_closed_orbit_planes)
_MODES = {
    "x": (BEND_ERROR_RMS, 0.0, "x"),
    "y": (0.0, QUAD_MISALIGN_Y_RMS, "y"),
    "xy": (BEND_ERROR_RMS, QUAD_MISALIGN_Y_RMS, "yx"),
}


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _bpm_mean_orbit(tracking_df, bpm_names, plane) -> float:
    """Max |per-BPM turn-mean| of *plane* over *bpm_names* in the tracking data."""
    rows = tracking_df[tracking_df["name"].isin(bpm_names)]
    means = rows.groupby("name", observed=True)[plane].mean().to_numpy(dtype=float)
    return float(np.abs(means).max())


@pytest.mark.slow
@pytest.mark.parametrize("mode", list(_MODES))
def test_psb_acd_reconstruction_zero_twiss_data_mean_co(mode, data_dir) -> None:
    bend_rms, quad_rms, co_planes = _MODES[mode]
    setup = build_psb_tracking_setup(
        data_dir,
        delta_p=0.0,
        driven_tunes=ACD_DRIVEN_TUNES,
        bend_error_rms=bend_rms,
        bend_error_seed=BEND_ERROR_SEED,
        apply_bend_errors_to_model=False,
        quad_misalign_y_rms=quad_rms,
        quad_misalign_seed=QUAD_MISALIGN_SEED,
    )
    tracking_df = setup["tracking_df"]
    tws = setup["tws"]
    model = setup["model"]

    # The model twiss must be ~on-zero (it does not know about the perturbations),
    # while the tracked data must carry a distorted orbit in the perturbed plane(s).
    assert float(tws["x"].abs().max()) < 1e-5
    assert float(tws["y"].abs().max()) < 1e-5

    before_marker, after_marker = acd_state_marker_names(model)
    bpm_df = tracking_df.loc[~tracking_df["name"].isin([before_marker, after_marker])].copy()
    bpm_names = list(bpm_df["name"].unique())
    if "x" in co_planes:
        assert _bpm_mean_orbit(tracking_df, bpm_names, "x") > 1e-4, "expected distorted x orbit"
    if "y" in co_planes:
        assert _bpm_mean_orbit(tracking_df, bpm_names, "y") > 1e-4, "expected distorted y orbit"

    result = calculate_pz(
        bpm_df,
        reference_co=zero_reference_co(bpm_df),
        model_details=ModelDetails(accelerator=model.accelerator, pt=model.pt),
        acd=ACDipoleConfig(
            ac_dipole_marker=ACD_ELEMENT,
            driven_tunes=ACD_DRIVEN_TUNES,
            data_mean_closed_orbit_planes=co_planes,
        ),
        acd_only=True,
    )

    summary = result.attrs["summary"]
    bpm_upstream = result.attrs["bpm_upstream"]
    bpm_downstream = result.attrs["bpm_downstream"]

    # The harmonic kick fit should still reconstruct the true kick well: the kick is
    # a difference of the two marker momenta, so the (constant) closed-orbit angle
    # largely cancels even though it is missing from the absolute momenta.
    assert result.attrs["dpx_r2"] > 0.99
    assert result.attrs["dpy_r2"] > 0.99

    # Compare reconstructed BPM momenta against the tracked truth, without cleaning
    # (raw `*_bpm_*`) and with cleaning (`*_bpm_*_cleaned`). The data-mean closed
    # orbit is restored with an angle inferred from its own positions, so the
    # reconstructed momentum is now absolute up to the first-order error of that
    # inference. We therefore report both:
    #   - "absolute": reconstructed vs the raw tracked momentum (how bad the missing
    #     closed-orbit angle makes the absolute momentum), and
    #   - "betatron": reconstructed vs the tracked momentum with its turn-mean removed
    #     (the closed-orbit angle ~= the turn-mean of the driven momentum), which
    #     isolates the quality of the betatron reconstruction itself.
    truth_cols = ["turn", "px", "py"]
    for side, bpm in (("upstream", bpm_upstream), ("downstream", bpm_downstream)):
        truth = (
            tracking_df.loc[tracking_df["name"] == bpm, truth_cols]
            .sort_values("turn")
            .rename(columns={"px": "px_true", "py": "py_true"})
        )
        merged = summary.merge(truth, on="turn", how="inner")
        for plane in ("px", "py"):
            true = merged[f"{plane}_true"].to_numpy(dtype=float)
            true_betatron = true - float(np.mean(true))
            # The betatron diagnostics below mean-remove *both* sides. Comparing a
            # mean-removed truth against a reconstruction that still carries its DC
            # term reports a large negative R^2 no matter how good the betatron
            # reconstruction is, which is not a useful signal. See
            # [test_psb_dynamic_part_acd.py], which asserts on this quantity.
            raw = merged[f"{plane}_bpm_{side}"].to_numpy(dtype=float)
            cleaned = merged[f"{plane}_bpm_{side}_cleaned"].to_numpy(dtype=float)
            LOGGER.info(
                "[mode=%s] %s_%s | absolute: raw RMS=%.3e (R2=%.3f) cleaned RMS=%.3e "
                "(R2=%.3f) | betatron: raw R2=%.4f cleaned R2=%.4f | "
                "co-angle=%.3e osc-RMS=%.3e",
                mode,
                plane,
                side,
                _rms(true - raw),
                r_squared(true, raw),
                _rms(true - cleaned),
                r_squared(true, cleaned),
                r_squared(true_betatron, raw - float(np.mean(raw))),
                r_squared(true_betatron, cleaned - float(np.mean(cleaned))),
                float(np.mean(true)),
                _rms(true_betatron),
            )
            # Assert on the closed-orbit angle itself rather than on R^2. The
            # restored angle is inferred from the data-mean positions with the
            # *nominal* model optics, so it is only first-order correct;
            # restoring nothing (the old betatron-only behaviour) would leave
            # 100% of `co_angle` in the residual. The bound is relative to the
            # angle where that dominates and to the oscillation amplitude where
            # the angle is small — it characterises the residual, it is not a
            # target. Only the raw momenta are asserted on: the harmonic
            # cleaning attenuates the DC term (see the logged cleaned values),
            # which is a separate, pre-existing property of the cleaner.
            co_angle = abs(float(np.mean(true)))
            residual = abs(float(np.mean(true - raw)))
            assert residual < max(0.2 * co_angle, 0.4 * _rms(true_betatron)), (
                f"{plane}_{side} raw: closed-orbit angle recovered only to "
                f"{residual:.3e} of {co_angle:.3e}"
            )
