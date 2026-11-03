"""The complete end-to-end contract matrix for supported reconstruction APIs."""

from __future__ import annotations

import logging
from dataclasses import replace

import numpy as np
import pytest

from tests.acd.acd_test_helpers import assert_acd_momenta_match_truth, r_squared
from tests.contracts.conftest import truth_and_reconstruction_for_plane
from tests.psb_tracking import ACD_ELEMENT
from tests.support.acd_barrier import acd_barrier_s
from tests.support.assertions import rmse
from tests.support.external_strengths import load_external_strength_fixture
from tests.support.lhc import lhc_acd_barrier_s, setup_xsuite_simulation
from tests.support.scenarios import MATCHED_BEND_AND_QUAD_ERRORS
from tmom_recon import (
    ModelDetails,
    calculate_acd_pz,
    calculate_pz,
    inject_noise_xy,
)
from tmom_recon.acd.integration import resolve_ac_dipole_config
from tmom_recon.model import resolve_model_details
from tmom_recon.physics.pt_calculation import estimate_closed_orbit_pt

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]
LOGGER = logging.getLogger(__name__)

# PSB clean: worst achieved over all cells (2026-09-28, after the zero-pt
# dispersion expansion) px 6.6e-10, py 8.6e-10 rad; limits ~4.5x that. LHC not
# re-measured.
_RMSE_MAX = {
    "psb": {"x": 3.0e-9, "y": 4.0e-9},
    "lhcb1": {"x": 1.4e-7, "y": 8.7e-8},
    "b1_120cm_crossing": {"x": 6.6e-8, "y": 5.8e-8},
}
# The noisy budgets are the propagated 1e-5 m BPM noise, not a round number.
# A neighbour pair turns a position error into an angle error of order
# sigma_x / beta, so the same noise costs PSB (beta ~ 5 m) an order of magnitude
# more than the LHC (beta ~ 100-8000 m).  Measured worst case per machine, over
# both planes, both frames and all three momenta:
#
#     psb      px 3.42e-6  py 3.90e-6
#     lhcb1    px 2.88e-7  py 2.61e-7
#     crossing px 2.57e-7  py 2.55e-7
#
# The limits below are those values with 25% headroom.  No SVD cleaning is
# applied first: cleaning is optional preprocessing, not part of the
# reconstruction, and rank-truncating this data would reduce the PSB error 2.6x
# (3.42e-6 -> 1.32e-6) purely by discarding noise modes.  Budgeting on cleaned
# data would let a real reconstruction error hide behind the filter.
_NOISY_RMSE_MAX = {
    "psb": {"x": 4.3e-6, "y": 4.9e-6},
    "lhcb1": {"x": 3.6e-7, "y": 3.3e-7},
    "b1_120cm_crossing": {"x": 3.2e-7, "y": 3.2e-7},
}
# Static (mean) momentum error of the clean all-BPM reconstruction; an RMSE hides
# a constant offset. PSB worst achieved 2.2e-10 rad.
_CLEAN_MEAN_MAX = {"psb": 1.0e-9}
# BPM-momentum R^2 floor by (machine, condition). PSB clean worst 1 - R^2 is
# 2.4e-10 and noisy 8.9e-4 (cleaned); LHC not re-measured.
_ACD_R2_MIN = {
    ("psb", "clean"): 1.0 - 1.0e-9,
    ("psb", "bpm_noise"): 0.9990,
    ("lhcb1", "clean"): 0.99982,
    ("lhcb1", "bpm_noise"): 0.99982,
    ("b1_120cm_crossing", "clean"): 0.99982,
    ("b1_120cm_crossing", "bpm_noise"): 0.99982,
}
# The kick is a *difference* of two BPM momenta, so with noise its fit is bounded
# by 1 - (sigma_fit / sigma_kick)**2.  The LHC scenarios track 100 turns, where a
# four-parameter harmonic fit leaves about 7e-8 rad on a ~1.5e-6 rad kick: R^2 =
# 0.9979 measured, against 0.9995+ for PSB, whose kick is larger and whose fit
# averages 1000 turns.  The floor keeps a factor 2.4 of margin in (1 - R^2) for
# seed-to-seed scatter; the clean floor stays where it was.
_KICK_R2_MIN = {"clean": 0.999, "bpm_noise": 0.995}
# PSB clean, all 1 - R^2 worst achieved: kick 2.4e-8, internal fit 6.6e-8,
# markers 4.3e-10; static marker error 3.5e-10 rad (px, py), 3.0e-9 m (x, y).
_PSB_CLEAN_KICK_R2_MIN = 1.0 - 1.0e-7
_PSB_CLEAN_FIT_R2_MIN = 1.0 - 3.0e-7
_PSB_CLEAN_MARKER_R2_MIN = 1.0 - 2.0e-9
_PSB_CLEAN_MARKER_MEAN_MAX = {"x": 1.5e-8, "px": 2.0e-9, "y": 1.5e-8, "py": 2.0e-9}
_MAGNETIC_ERROR_RMSE_MAX = {
    "psb": {"x": 1.2e-7, "y": 1.2e-7},
    "lhcb1": {"x": 1.4e-7, "y": 1.4e-7},
    "b1_120cm_crossing": {"x": 1.0e-7, "y": 1.0e-7},
}


def _data(scenario):
    if scenario.condition == "clean":
        return scenario.data
    if scenario.condition == "bpm_noise":
        return inject_noise_xy(
            scenario.data.copy(deep=True), np.random.default_rng(42), scenario.noise_std
        )
    raise AssertionError(f"Unsupported reconstruction condition {scenario.condition!r}")


def _assert_all_bpm(scenario, data, *, acd) -> None:
    """Assert the all-BPM reconstruction on free (``acd=None``) or driven optics.

    Both cells reconstruct the same driven data against the same budget: the
    optics the neighbour pairs are transported through is the only difference,
    so a divergence between them is a statement about the driven twiss and
    nothing else.
    """
    label = "all" if acd is None else "driven_all"
    result = calculate_pz(
        data,
        scenario.model_details,
        closed_orbit_at_zero=scenario.reference,
        orbit_mode=scenario.orbit_mode,
        optics=scenario.optics,
        acd=acd,
        barrier_s=scenario.barrier_s,
        info=False,
    )
    failures = []
    for plane in ("x", "y"):
        truth, reconstructed = truth_and_reconstruction_for_plane(scenario.data, result, plane)
        error = rmse(truth.to_numpy(), reconstructed.to_numpy())
        budgets = _NOISY_RMSE_MAX if scenario.condition == "bpm_noise" else _RMSE_MAX
        limit = budgets[scenario.machine][plane]
        LOGGER.info(
            "%s %s dp=%+.3e %s: p%s RMSE=%.3e limit=%.3e",
            label,
            scenario.machine,
            scenario.delta_p,
            scenario.condition,
            plane,
            error,
            limit,
        )
        if error >= limit:
            failures.append(f"p{plane} RMSE={error:.3e} limit={limit:.3e}")
        mean_limit = _CLEAN_MEAN_MAX.get(scenario.machine)
        if scenario.condition == "clean" and mean_limit is not None:
            mean = float(np.mean(reconstructed.to_numpy() - truth.to_numpy()))
            if abs(mean) >= mean_limit:
                failures.append(f"p{plane} static error {mean:.3e} limit={mean_limit:.1e}")
    assert not failures, (
        f"{label} {scenario.machine} dp={scenario.delta_p:+.3e} {scenario.condition}: "
        + "; ".join(failures)
    )


def _assert_all(scenario, data) -> None:
    _assert_all_bpm(scenario, data, acd=None)


def _assert_driven_all(scenario, data) -> None:
    """``calculate_pz(acd=...)``: all-BPM momenta on the *driven* optics.

    This is not the AC-dipole kick fit (``calculate_acd_pz``, the ``acd``
    workflow). It is the production path used downstream whenever the
    acquisition was taken with the dipole driving, and the only cell in this
    suite that passes ``acd=`` to ``calculate_pz`` at all.

    Against *exact model* optics the driven and free reconstructions agree to
    round-off (1.3e-14 rad at dp=0, 1.1e-9 at dp=1e-3 for PSB) even though the
    driven twiss differs by 9% in beta and 1.8% in phase: a neighbour pair is
    transported through the same lattice either way, and the driven twiss only
    reparametrises it. So this cell is a regression guard on the plumbing --
    that ``acd=`` is accepted, resolves the driven twiss, honours ``barrier_s``
    and still reconstructs correctly -- not a discriminating test of driven
    optics. Discriminating it needs measured optics, and therefore a budget
    derived from phase estimator uncertainty rather than the exact-model one.
    """
    _assert_all_bpm(scenario, data, acd=scenario.acd)


def _assert_acd(scenario, data) -> None:
    result = calculate_acd_pz(
        data,
        scenario.model_details,
        scenario.acd,
        closed_orbit_at_zero=scenario.reference,
        orbit_mode=scenario.orbit_mode,
        optics=scenario.optics,
    )
    summary = result.attrs["summary"]
    failures = []
    for side in ("upstream", "downstream"):
        bpm = str(result.attrs[f"bpm_{side}"])
        for plane in ("px", "py"):
            truth = scenario.data.loc[
                scenario.data["name"].astype(str).str.upper() == bpm.upper(), ["turn", plane]
            ].rename(columns={plane: "truth"})
            reconstructed = summary[["turn", f"{plane}_bpm_{side}_cleaned"]].rename(
                columns={f"{plane}_bpm_{side}_cleaned": plane}
            )
            merged = truth.merge(reconstructed, on="turn", validate="one_to_one")
            r2 = r_squared(merged["truth"], merged[plane])
            limit = _ACD_R2_MIN[scenario.machine, scenario.condition]
            if r2 <= limit:
                failures.append(f"{side} {plane} R2={r2:.6f} limit={limit:.6f}")
    assert not failures, (
        f"acd {scenario.machine} dp={scenario.delta_p:+.3e} {scenario.condition}: "
        + "; ".join(failures)
    )
    # The summary fit alone can absorb a static error.  Compare the complete
    # reconstructed state at both BPMs and at the physical before/after marker
    # states to the independently tracked truth.
    resolved = resolve_ac_dipole_config(scenario.model_details, scenario.acd)
    psb_clean = scenario.machine == "psb" and scenario.condition == "clean"
    assert_acd_momenta_match_truth(
        result,
        scenario.marker_truth,
        resolved.model,
        clean=scenario.condition == "clean",
        kick_r2_min=_PSB_CLEAN_KICK_R2_MIN if psb_clean else _KICK_R2_MIN[scenario.condition],
        bpm_r2_min=_ACD_R2_MIN[scenario.machine, scenario.condition],
        marker_r2_min=_PSB_CLEAN_MARKER_R2_MIN if psb_clean else 0.998,
        marker_pos_r2_min=_PSB_CLEAN_MARKER_R2_MIN if psb_clean else 0.998,
        fit_r2_min=_PSB_CLEAN_FIT_R2_MIN if psb_clean else 0.999,
        marker_mean_max=_PSB_CLEAN_MARKER_MEAN_MAX if psb_clean else None,
    )


_CONTRACTS = {"all": _assert_all, "acd": _assert_acd, "driven_all": _assert_driven_all}


def assert_reconstruction(contract_scenario) -> None:
    """Each machine/workflow/momentum/condition is one complete contract cell."""
    _CONTRACTS[contract_scenario.workflow](contract_scenario, _data(contract_scenario))


def assert_model_pt_estimate_matches_tracked_offset(contract_scenario) -> None:
    """The model-dispersion estimator must recover the measurement pt.

    Dynamic OMC3 scenarios use this same assertion after their measured optics
    have been produced; keeping it independent of momentum reconstruction
    makes a wrong energy estimate immediately diagnosable.

    The estimator differences the mean orbit of the data against the frame's
    origin, so unlike the reconstruction it needs an origin *measured the same
    way* -- an on-momentum acquisition, not the exact closed orbit. Both carry
    the same driven-mean bias, and only their difference enters the fit.
    """
    # Estimate the offset from the nominal reconstruction optics.
    estimate = estimate_closed_orbit_pt(
        contract_scenario.data,
        replace(contract_scenario.model_details, pt=0.0),
        closed_orbit_at_zero=contract_scenario.measured_orbit_zero,
    )
    assert estimate == pytest.approx(contract_scenario.pt, abs=2.5e-6), (
        f"{contract_scenario.machine} dp={contract_scenario.delta_p:+.3e}: "
        f"pt estimate {estimate:.3e} differs from {contract_scenario.pt:.3e}"
    )


def _psb_runtime_strengths(setup) -> dict[str, float]:
    strengths = {f"{name.upper()}.k0": value for name, value in setup.bend_strengths.items()}
    strengths.update({f"{name.upper()}.k1": value for name, value in setup.quad_strengths.items()})
    return strengths


def assert_all_bpm_with_matched_magnetic_errors(
    machine,
    data_dir,
    psb_scenarios,
    tmp_path_factory,
    xsuite_json_path,
) -> None:
    """Externally supplied strengths and reference angles recover both planes."""
    external = load_external_strength_fixture(data_dir / "external_strengths" / f"{machine}.json")

    if machine == "psb":
        setup = psb_scenarios(delta_p=1e-3, errors=MATCHED_BEND_AND_QUAD_ERRORS)
        data = setup.measurement.data.loc[
            setup.measurement.data["name"].isin(setup.measurement.bpm_names)
        ].copy()
        assert _psb_runtime_strengths(setup) == pytest.approx(external.strengths, rel=0.0, abs=0.0)
        details = ModelDetails(
            accelerator=setup.machine.accelerator,
            pt=setup.measurement.pt,
            magnet_strengths=dict(external.strengths),
        )
        barrier_s = acd_barrier_s(setup.machine.madng_model, ACD_ELEMENT)
    else:
        sequence_file = data_dir / "sequences" / f"{machine}.seq"
        data, _truth, details, _tws, _line = setup_xsuite_simulation(
            0.0,
            "all",
            12,
            xsuite_json_path(sequence_file.name),
            sequence_file,
            tmp_path_factory.mktemp(f"magnetic-errors-{machine}"),
            f"magnetic_errors_{machine}",
        )
        assert details.magnet_strengths == pytest.approx(external.strengths, rel=0.0, abs=0.0)
        details = replace(details, magnet_strengths=dict(external.strengths))
        barrier_s = lhc_acd_barrier_s(details.accelerator, details.pt)

    # The setting-zero orbit is recorded *on momentum*, in the same errored
    # machine.  Taking the model twiss at `pt` instead would remove the
    # dispersive orbit here and again inside the reconstruction, which for the
    # PSB at dp/p = 1e-3 is a 1e-4 rad double subtraction.
    origin = resolve_model_details(replace(details, pt=0.0)).tws
    result = calculate_pz(
        data,
        details,
        closed_orbit_at_zero=origin[["x", "y"]],
        orbit_mode="absolute",
        barrier_s=barrier_s,
        info=False,
    )
    failures = []
    for plane in ("x", "y"):
        truth, reconstructed = truth_and_reconstruction_for_plane(data, result, plane)
        error = rmse(truth.to_numpy(), reconstructed.to_numpy())
        limit = _MAGNETIC_ERROR_RMSE_MAX[machine][plane]
        if error >= limit:
            failures.append(f"p{plane} RMSE={error:.3e} limit={limit:.3e}")
    assert not failures, f"magnetic_errors {machine}: " + "; ".join(failures)
