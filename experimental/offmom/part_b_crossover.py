"""Part B: does the pt method still matter once the model error is reduced?

Runs the full AC-dipole reconstruction on a PSB ring-3 machine carrying bend AND
quad errors that the *tracking line* has and the *model* does not, then hands the
model a ladder of increasingly good error estimates (from Part A) via
``ModelDetails.magnet_strengths``, and compares the two closed-orbit references
(linear ``pt*D`` vs ``dispersive_closed_orbit=True``) at each rung.

The rungs give a controlled sweep of residual model closed-orbit error, which is
the x-axis of the crossover.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from co_common import bpm_names, build_line, closed_orbit, nominal_k0  # noqa: E402

from tests.acd.acd_test_helpers import _truth_at, acd_state_marker_names, r_squared  # noqa: E402
from tests.psb_tracking import ACD_ELEMENT, DRIVEN_TUNES, build_psb_tracking_setup  # noqa: E402
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_pz  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
BEND_RMS, BEND_SEED = 8e-4, 7
QUAD_RMS, QUAD_SEED = 1e-3, 11
ACD_DRIVEN_TUNES = (0.18, DRIVEN_TUNES[1])
DPS = [float(x) for x in (sys.argv[1:] or ["8e-3"])]


def strengths_from(dk0, dq, bends, quads, line):
    out = {}
    for n, d in zip(bends, dk0):
        out[f"{n.upper()}.k0"] = nominal_k0(line, n) + float(d)
    for n, d in zip(quads, dq):
        out[f"{n.upper()}.k1"] = float(line[n].k1) * (1 + float(d))
    return out


def model_variants():
    """(label, magnet_strengths, model-vs-machine |CO error| at each dp)."""
    nominal, _, _ = build_line()
    machine, bend_true, quad_true = build_line(
        bend_rms=BEND_RMS, bend_seed=BEND_SEED, quad_rms=QUAD_RMS, quad_seed=QUAD_SEED
    )
    bpms = bpm_names(nominal)
    co_mach = {dp: closed_orbit(machine, dp, bpms) for dp in DPS}

    variants = [("nominal (knows nothing)", {}, nominal)]
    for tag, npz in (
        ("bend+quad fit q8", "fitted_errors_q8.npz"),
        ("bend+quad fit q12", "fitted_errors.npz"),
        ("bend+quad fit q16", "fitted_errors_q16.npz"),
    ):
        d = np.load(Path(__file__).with_name(npz), allow_pickle=True)
        bends, quads = [str(x) for x in d["bends"]], [str(x) for x in d["quads"]]
        line, _, _ = build_line()
        for n, dk in zip(bends, d["dk0"]):
            line[n].k0 = nominal_k0(line, n) + float(dk)
        for n, dqq in zip(quads, d["dq"]):
            line[n].k1 = float(line[n].k1) * (1 + float(dqq))
        variants.append((tag, strengths_from(d["dk0"], d["dq"], bends, quads, nominal), line))
        if tag == "bend+quad fit q12":  # bend-only rung, same fit with quads dropped
            line2, _, _ = build_line()
            for n, dk in zip(bends, d["dk0"]):
                line2[n].k0 = nominal_k0(line2, n) + float(dk)
            variants.insert(
                1,
                (
                    "bend-only fit",
                    strengths_from(d["dk0"], np.zeros(len(quads)), bends, quads, nominal),
                    line2,
                ),
            )
    truth_strengths = {f"{n.upper()}.k0": v for n, v in bend_true.items()}
    truth_strengths.update({f"{n.upper()}.k1": v for n, v in quad_true.items()})
    variants.append(("TRUE errors (control)", truth_strengths, machine))

    out = []
    for label, strengths, line in variants:
        co_err = {dp: float(np.abs(closed_orbit(line, dp, bpms) - co_mach[dp]).max()) for dp in DPS}
        out.append((label, strengths, co_err))
    return out


def run(setup, strengths, dispersive):
    model = setup["model"]
    before, after = acd_state_marker_names(model)
    df = setup["tracking_df"]
    bpm_df = df.loc[~df["name"].isin([before, after])].copy()
    result = calculate_pz(
        bpm_df,
        model_details=ModelDetails(
            accelerator=model.accelerator, pt=model.pt, magnet_strengths=strengths
        ),
        acd=ACDipoleConfig(
            ac_dipole_marker=ACD_ELEMENT,
            driven_tunes=ACD_DRIVEN_TUNES,
            dispersive_closed_orbit=dispersive,
        ),
        acd_only=True,
    )
    summary = result.attrs["summary"]
    r2 = {}
    for side in ("upstream", "downstream"):
        bpm = result.attrs[f"bpm_{side}"]
        m = summary.merge(_truth_at(df, bpm), on="turn", how="inner")
        for plane in ("px", "py"):
            r2[f"{plane}_{side}"] = r_squared(
                m[f"{plane}_true"].to_numpy(), m[f"{plane}_bpm_{side}"].to_numpy()
            )
    k = _truth_at(df, after).merge(_truth_at(df, before), on="turn", suffixes=("_a", "_b"))
    k["dpx_true"] = k["px_true_a"] - k["px_true_b"]
    kk = summary.merge(k[["turn", "dpx_true"]], on="turn", how="inner")
    r2["kick_dpx"] = r_squared(kk["dpx_true"], kk["dpx_fit_rad"])
    return r2


def main():
    warnings.filterwarnings("ignore")
    variants = model_variants()
    for dp in DPS:
        setup = build_psb_tracking_setup(
            DATA_DIR,
            delta_p=dp,
            driven_tunes=ACD_DRIVEN_TUNES,
            bend_error_rms=BEND_RMS,
            bend_error_seed=BEND_SEED,
            apply_bend_errors_to_model=False,
            quad_error_rms=QUAD_RMS,
            quad_error_seed=QUAD_SEED,
            apply_quad_errors_to_model=False,
        )
        print(f"\n===== delta_p = {dp:+.1e} =====", flush=True)
        print(
            f"{'model':<24} {'|CO err|':>9} {'method':<8} {'kick dpx':>9} "
            f"{'px up':>10} {'px down':>10} {'py down':>10}",
            flush=True,
        )
        for label, strengths, co_err in variants:
            for dispersive in (False, True):
                try:
                    r2 = run(setup, strengths, dispersive)
                    print(
                        f"{label:<24} {co_err[dp]:9.2e} "
                        f"{'exact' if dispersive else 'linear':<8} "
                        f"{r2['kick_dpx']:9.6f} {r2['px_upstream']:10.6f} "
                        f"{r2['px_downstream']:10.6f} {r2['py_downstream']:10.6f}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"{label:<24} {co_err[dp]:9.2e} "
                        f"{'exact' if dispersive else 'linear':<8} "
                        f"FAILED: {type(exc).__name__}: {str(exc)[:90]}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
