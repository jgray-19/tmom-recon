"""Global driven-optics ACD markers (``calculate_acd_global_pz``) against PSB tracking truth."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from pymadng_utils.mad import AcceleratorMadInterface

from tests.acd.acd_test_helpers import acd_state_marker_names
from tests.contracts.conftest import ScenarioSpec
from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES
from tests.support.scenarios import MagnetErrors
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_acd_global_pz, calculate_acd_pz
from tmom_recon.acd.global_markers import GLOBAL_SUMMARY_ATTR_NAME, harmonic_lines
from tmom_recon.acd.integration import resolve_ac_dipole_config

LOGGER = logging.getLogger(__name__)
pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow, pytest.mark.psb]

# Exact model. Relative drive-line error |Z_rec - Z_true| / |Z_true|, worst
# measured 2026-09-28 over dp/p = 0, 1e-3: markers 1.5e-4 (px before, dp/p=1e-3),
# kick 1.8e-3 (px, dp/p=1e-3); BPM residual 5.3e-5. Limits ~2x.
_MARKER_LINE_MAX = 3e-4
_KICK_LINE_MAX = 4e-3
_BPM_RESIDUAL_MAX = 1e-3
# Unmodelled quad errors, tune-matched nominal model. Kick drive-line error
# measured 2026-09-28 (global / tracked, px and py):
#   quad_rms 1e-3: 0.46% / 3.2%, 1.2% / 2.6%
#   quad_rms 3e-3: 1.6% / 16%,   3.8% / 13%
#   quad_rms 1e-2: 4.8% / 31%,   13% / 59%   (not a test cell)
# Limits ~1.6x the worst plane; the global kick must also beat the tracked one.
_ROBUST_KICK_MAX = {1e-3: 2e-2, 3e-3: 6e-2}
_ROBUST_TRACKED_RATIO_MAX = 0.7


def _line(turns, values, tune) -> complex:
    return complex(harmonic_lines(turns, values, [tune])[0][0])


@pytest.mark.parametrize(
    "contract_scenario",
    [
        pytest.param(ScenarioSpec("psb", "acd", dp, "clean"), id=f"dp_{dp:+.0e}")
        for dp in (0.0, 1e-3)
    ],
    indirect=True,
)
def test_global_markers_match_tracked_truth(contract_scenario) -> None:
    """Exact model: the marker drive lines and the kick match truth; the static part stays tracked."""
    sc = contract_scenario
    kwargs = {
        "closed_orbit_at_zero": sc.reference,
        "orbit_mode": sc.orbit_mode,
        "optics": sc.optics,
    }
    result = calculate_acd_global_pz(sc.data, sc.model_details, sc.acd, **kwargs)
    tracked = calculate_acd_pz(sc.data, sc.model_details, sc.acd, **kwargs)
    names = acd_state_marker_names(resolve_ac_dipole_config(sc.model_details, sc.acd).model)
    fits = result.attrs[GLOBAL_SUMMARY_ATTR_NAME]
    for fit in fits:
        assert fit.median_residual < _BPM_RESIDUAL_MAX, (
            f"mode {fit.mode} BPM residual {fit.median_residual:.2e}"
        )

    truth = sc.marker_truth.assign(name=sc.marker_truth["name"].astype(str).str.upper())
    rec = result.assign(name=result["name"].astype(str).str.upper())
    trk = tracked.assign(name=tracked["name"].astype(str).str.upper())
    lines = {}
    for name in names:
        t = truth[truth["name"] == name].sort_values("turn")
        r = rec[rec["name"] == name].set_index("turn").loc[t["turn"]]
        k = trk[trk["name"] == name].set_index("turn").loc[t["turn"]]
        turns = t["turn"].to_numpy(float)
        for col, q in (
            ("x", sc.acd.driven_tunes[0]),
            ("px", sc.acd.driven_tunes[0]),
            ("y", sc.acd.driven_tunes[1]),
            ("py", sc.acd.driven_tunes[1]),
        ):
            z_true, z_rec = (_line(turns, frame[col].to_numpy(float), q) for frame in (t, r))
            lines[name, col] = z_true, z_rec
            err = abs(z_rec - z_true) / abs(z_true)
            LOGGER.info("%s %s marker drive-line error %.2e", name, col, err)
            assert err < _MARKER_LINE_MAX, f"{name} {col}: drive line off by {err:.2e}"
            # Only the drive line is replaced: the static part stays the tracked one.
            np.testing.assert_allclose(r[col].mean(), k[col].mean(), rtol=0, atol=1e-9)

    before, after = names
    for mom in ("px", "py"):
        kick_true = lines[after, mom][0] - lines[before, mom][0]
        kick_rec = lines[after, mom][1] - lines[before, mom][1]
        err = abs(kick_rec - kick_true) / abs(kick_true)
        LOGGER.info("%s kick drive-line error %.2e", mom, err)
        assert err < _KICK_LINE_MAX, f"{mom} kick drive line off by {err:.2e}"


def _kick_line_errors(result, truth_data, names, tunes) -> dict[str, float]:
    """Relative drive-line error of the reconstructed px/py kick against truth."""
    truth = truth_data.assign(name=truth_data["name"].astype(str).str.upper())
    rec = result.assign(name=result["name"].astype(str).str.upper())
    before, after = names
    errors = {}
    for mom, q in (("px", tunes[0]), ("py", tunes[1])):
        z = {}
        for label, frame in (("truth", truth), ("rec", rec)):
            for name in names:
                rows = frame[frame["name"] == name].sort_values("turn")
                z[label, name] = _line(rows["turn"].to_numpy(float), rows[mom].to_numpy(float), q)
        kick_true = z["truth", after] - z["truth", before]
        errors[mom] = abs(z["rec", after] - z["rec", before] - kick_true) / abs(kick_true)
    return errors


@pytest.mark.parametrize("quad_rms", sorted(_ROBUST_KICK_MAX), ids=lambda v: f"quad_rms_{v:.0e}")
def test_global_kick_is_robust_to_unmodelled_quad_errors(psb_scenarios, quad_rms) -> None:
    """Nominal model, errored machine: the global kick beats the tracked one.

    The tracked kick is a difference of two tracked momenta, so the local model
    error between the two BPMs next to the AC dipole enters it at full size. The
    global kick follows the driven alpha jump and one ring-averaged amplitude.

    The model must sit on the machine's natural tunes (as production matches it
    to the measured ones): the driven optics depend on Q_nat - Q_drive (0.01
    here), and an unmatched ~2e-3 tune miss alone costs the global kick ~20%.
    """
    setup = psb_scenarios(delta_p=0.0, errors=MagnetErrors(quad_rms=quad_rms, quad_seed=11))
    data_all = setup.measurement.data
    data = data_all.loc[data_all["name"].isin(setup.measurement.bpm_names)].copy()
    headers = {str(k).lower(): v for k, v in setup.machine.madng_twiss.headers.items()}
    natural = (float(headers["q1"]) % 1, float(headers["q2"]) % 1)
    knobs = AcceleratorMadInterface(setup.machine.accelerator).match_tunes(*natural)
    details = ModelDetails(
        accelerator=setup.machine.accelerator, pt=setup.measurement.pt, tune_knobs=knobs
    )
    config = ACDipoleConfig(ac_dipole_marker=ACD_ELEMENT, driven_tunes=DRIVEN_TUNES)
    origin = data.groupby("name", sort=False)[["x", "y"]].mean()
    kwargs = {"closed_orbit_at_zero": origin, "orbit_mode": "dynamic"}
    names = acd_state_marker_names(resolve_ac_dipole_config(details, config).model)
    errors = {
        label: _kick_line_errors(
            method(data, details, config, **kwargs), data_all, names, DRIVEN_TUNES
        )
        for label, method in (("global", calculate_acd_global_pz), ("tracked", calculate_acd_pz))
    }
    LOGGER.info("quad_rms %.0e, natural tunes %s: kick errors %s", quad_rms, natural, errors)
    for mom in ("px", "py"):
        glob, trk = errors["global"][mom], errors["tracked"][mom]
        assert glob < _ROBUST_KICK_MAX[quad_rms], f"{mom} global kick error {glob:.2e}"
        assert glob < _ROBUST_TRACKED_RATIO_MAX * trk, (
            f"{mom} global {glob:.2e} vs tracked {trk:.2e}"
        )


# 2 mrad rms quad rolls: 15-20% cross-plane lines at the AC dipole (psb_sim's
# truth machine: ~1%), a hard test of the coupled forced response. Measured
# 2026-09-28, exact model, relative line errors at the markers:
#   global:  own line 9.2e-5, cross line 6.4e-4 (worst coordinate)
#   tracked: own line 8.2e-4, cross line 6.5e-2
#   global with the driven-twiss lens model instead (superseded): 1.0e-2, 2.0e-1
_TILT_RMS = 2e-3
_COUPLED_OWN_MAX = 3e-4
_COUPLED_CROSS_MAX = 1.5e-3


@pytest.fixture(scope="module")
def tilted_psb(psb_model_dir):
    from tests.psb_tracking import build_psb_tracking_setup

    setup = build_psb_tracking_setup(psb_model_dir, 0.0, quad_tilt_rms=_TILT_RMS, quad_tilt_seed=5)
    tilts = {f"{name.upper()}.tilt": value for name, value in setup.quad_tilts.items()}
    details = ModelDetails(
        accelerator=setup.machine.accelerator, pt=setup.measurement.pt, magnet_strengths=tilts
    )
    return setup, details


def _mode_lines(frame, name, tunes) -> dict[str, np.ndarray]:
    rows = frame[frame["name"] == name].sort_values("turn")
    turns = rows["turn"].to_numpy(float)
    return {
        c: harmonic_lines(turns, rows[c].to_numpy(float), tunes)[0] for c in ("x", "px", "y", "py")
    }


def test_coupled_markers_match_tracked_truth(tilted_psb) -> None:
    """Coupled machine, exact model: both drive lines in all four marker coordinates match truth.

    The per-plane tracked markers miss the cross-plane line by ~5%; the forced
    response carries it.
    """
    setup, details = tilted_psb
    data_all = setup.measurement.data
    data = data_all.loc[data_all["name"].isin(setup.measurement.bpm_names)].copy()
    config = ACDipoleConfig(ac_dipole_marker=ACD_ELEMENT, driven_tunes=DRIVEN_TUNES)
    origin = data.groupby("name", sort=False)[["x", "y"]].mean()
    kwargs = {"closed_orbit_at_zero": origin, "orbit_mode": "dynamic"}
    names = acd_state_marker_names(resolve_ac_dipole_config(details, config).model)
    results = {
        "global": calculate_acd_global_pz(data, details, config, **kwargs),
        "tracked": calculate_acd_pz(data, details, config, **kwargs),
    }
    truth = data_all.assign(name=data_all["name"].astype(str).str.upper())
    for name in names:
        z_true = _mode_lines(truth, name, DRIVEN_TUNES)
        for label, result in results.items():
            rec = result.assign(name=result["name"].astype(str).str.upper())
            z_rec = _mode_lines(rec, name, DRIVEN_TUNES)
            for col, own in (("x", 0), ("px", 0), ("y", 1), ("py", 1)):
                cross = 1 - own
                ratio = abs(z_true[col][cross]) / abs(z_true[col][own])
                own_err = abs(z_rec[col][own] - z_true[col][own]) / abs(z_true[col][own])
                cross_err = abs(z_rec[col][cross] - z_true[col][cross]) / abs(z_true[col][cross])
                LOGGER.info(
                    "%s %s %s: cross/own %.2e, own-line err %.2e, cross-line err %.2e",
                    label,
                    name,
                    col,
                    ratio,
                    own_err,
                    cross_err,
                )
                assert ratio > 0.1, f"{name} {col}: truth barely coupled ({ratio:.2e})"
                if label == "global":
                    assert own_err < _COUPLED_OWN_MAX, f"{name} {col} own line off {own_err:.2e}"
                    assert cross_err < _COUPLED_CROSS_MAX, (
                        f"{name} {col} cross line off {cross_err:.2e}"
                    )


def test_mode_vectors_match_madng_eigenmode_columns(tilted_psb) -> None:
    """A-matrix mode vectors agree with MAD-NG's coupled columns (beta12/21, alfa12/21, ev1/ev2, kpa).

    Pins the conventions both ways: mode 1 in y is sqrt(beta21) conj(ev1) exp(2 pi i mu1),
    its py part -(alfa21 - i u)/sqrt(beta21) (same phase) with u = 1 - kpa, and
    symmetrically for mode 2 in x.
    """
    from tmom_recon.acd.global_markers import normal_mode_responses

    _setup, details = tilted_psb
    config = ACDipoleConfig(ac_dipole_marker=ACD_ELEMENT, driven_tunes=DRIVEN_TUNES)
    model = resolve_ac_dipole_config(details, config).model
    tws = model.run_twiss(observe=1, coupling=True, pt=model.pt)
    tws = tws.set_axis(tws.index.astype(str).str.upper())
    vectors, _ = normal_mode_responses(model, model.pt)
    assert max(tws["beta12"].max(), tws["beta21"].max()) > 1e-3, "lattice is not coupled"
    for name, v in vectors.items():
        r = tws.loc[name]
        e1, e2 = np.exp(2j * np.pi * r["mu1"]), np.exp(2j * np.pi * r["mu2"])
        u = 1 - r["kpa"]
        expected = {
            (0, 0): np.sqrt(r["beta11"]) * e1,
            (2, 1): np.sqrt(r["beta22"]) * e2,
            (2, 0): np.sqrt(r["beta21"]) * np.conj(r["ev1"]) * e1,
            (3, 0): -(r["alfa21"] - 1j * u) / np.sqrt(r["beta21"]) * np.conj(r["ev1"]) * e1,
            (0, 1): np.sqrt(r["beta12"]) * np.conj(r["ev2"]) * e2,
            (1, 1): -(r["alfa12"] - 1j * u) / np.sqrt(r["beta12"]) * np.conj(r["ev2"]) * e2,
        }
        scale = np.abs(v).max()
        for (i, k), value in expected.items():
            assert abs(v[i, k] - value) < 1e-9 * scale, (
                f"{name} v[{i},{k}]: A {v[i, k]} vs columns {value}"
            )


# Under-coupled model (the start of a fit, and every refresh short of the
# fitted coupling): the tilted machine above, reconstructed with the model's
# tilts scaled by ``model_coupling``, tune-matched to the machine. Worst marker
# line errors measured 2026-09-29 (cross from model / from data):
#   tilts x0:   cross line 1.00 / 0.204, own line 1.43e-2 / 1.43e-2, kick 0.252 / 0.252
#   tilts x0.5: cross line 0.496 / 0.094, own line 2.15e-2 / 1.64e-2, kick 0.282 / 0.274
# The kick error is the under-coupled model's own (2 mrad rolls; ~0.2% at
# psb_sim's coupling) and the cross-mode fit leaves it alone.
_CROSS_FROM_DATA_RATIO_MAX = 0.35


@pytest.mark.parametrize("model_coupling", [0.0, 0.5], ids=lambda v: f"model_tilts_x{v:g}")
def test_cross_mode_fit_recovers_coupling_missing_from_model(tilted_psb, model_coupling) -> None:
    """Markers' cross-plane lines come from the data, not the model's coupling."""
    setup, _ = tilted_psb
    data_all = setup.measurement.data
    data = data_all.loc[data_all["name"].isin(setup.measurement.bpm_names)].copy()
    headers = {str(k).lower(): v for k, v in setup.machine.madng_twiss.headers.items()}
    natural = (float(headers["q1"]) % 1, float(headers["q2"]) % 1)
    tilts = {
        f"{name.upper()}.tilt": model_coupling * value for name, value in setup.quad_tilts.items()
    }
    interface = AcceleratorMadInterface(setup.machine.accelerator)
    interface.set_magnet_strengths(tilts)
    knobs = interface.match_tunes(*natural)
    details = ModelDetails(
        accelerator=setup.machine.accelerator,
        pt=setup.measurement.pt,
        magnet_strengths=tilts,
        tune_knobs=knobs,
    )
    config = ACDipoleConfig(ac_dipole_marker=ACD_ELEMENT, driven_tunes=DRIVEN_TUNES)
    origin = data.groupby("name", sort=False)[["x", "y"]].mean()
    kwargs = {"closed_orbit_at_zero": origin, "orbit_mode": "dynamic"}
    names = acd_state_marker_names(resolve_ac_dipole_config(details, config).model)
    results = {
        label: calculate_acd_global_pz(data, details, config, fit_cross_modes=fit, **kwargs)
        for label, fit in (("model", False), ("data", True))
    }
    truth = data_all.assign(name=data_all["name"].astype(str).str.upper())
    errors = {}
    for label, result in results.items():
        rec = result.assign(name=result["name"].astype(str).str.upper())
        own_worst = cross_worst = 0.0
        for name in names:
            z_true, z_rec = (
                _mode_lines(truth, name, DRIVEN_TUNES),
                _mode_lines(rec, name, DRIVEN_TUNES),
            )
            for col, own in (("x", 0), ("px", 0), ("y", 1), ("py", 1)):
                cross = 1 - own
                own_worst = max(own_worst, abs(z_rec[col][own] / z_true[col][own] - 1))
                cross_worst = max(cross_worst, abs(z_rec[col][cross] / z_true[col][cross] - 1))
        kick = _kick_line_errors(result, data_all, names, DRIVEN_TUNES)
        errors[label] = own_worst, cross_worst, max(kick.values())
        LOGGER.info(
            "model tilts x%g, cross from %s: own-line %.2e, cross-line %.2e, kick %.2e",
            model_coupling,
            label,
            *errors[label],
        )
    (own_m, cross_m, kick_m), (own_d, cross_d, kick_d) = errors["model"], errors["data"]
    assert cross_d < _CROSS_FROM_DATA_RATIO_MAX * cross_m, (
        f"cross line {cross_d:.2e} vs {cross_m:.2e}"
    )
    assert own_d < 1.05 * own_m, f"own line {own_d:.2e} vs {own_m:.2e}"
    assert kick_d < 1.05 * kick_m, f"kick {kick_d:.2e} vs {kick_m:.2e}"
