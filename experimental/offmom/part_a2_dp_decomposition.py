"""Part A2: separate the dp-independent (bend-error) and dp-dependent (dispersion)
parts of the measured closed orbit by polynomial fit in delta_p, per BPM.

This is what measuring the closed orbit at several delta_p actually buys: the
constant term feeds the bend-error response-matrix fit, and the linear/quadratic
terms are a *measured* dispersion and second-order dispersion that the model does
not have (because it does not know the quad gradient errors).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from co_common import bend_names, bpm_names, build_line, closed_orbit, nominal_k0  # noqa: E402
from part_a_bend_fit import BEND_RMS, QUAD_RMS, response_matrix, svd_solve  # noqa: E402

DPS = np.array([0.0, 1e-3, -1e-3, 3e-3, -3e-3, 8e-3, -8e-3])
N_SV = 17


def main():
    machine, bend_k0_true, _ = build_line(bend_rms=BEND_RMS, quad_rms=QUAD_RMS)
    model, _, _ = build_line()
    bpms, bends = bpm_names(model), bend_names(model)

    co_mach = np.array([closed_orbit(machine, dp, bpms) for dp in DPS])
    co_model = np.array([closed_orbit(model, dp, bpms) for dp in DPS])
    resid = co_mach - co_model  # (n_dp, 2*n_bpm)

    for order in (0, 1, 2):
        coef = np.polynomial.polynomial.polyfit(DPS, resid, order)  # (order+1, n_obs)
        pred = np.polynomial.polynomial.polyval(DPS, coef).T
        rms = np.sqrt(((resid - pred) ** 2).mean())
        print(f"poly order {order}: residual-of-fit rms {rms:.3e}")
    coef = np.polynomial.polynomial.polyfit(DPS, resid, 2)
    const, lin, quad = coef
    print(
        f"\nconstant  term (bend-error orbit)  rms {np.sqrt((const**2).mean()):.3e} max {np.abs(const).max():.3e}"
    )
    print(
        f"linear    term (dispersion error)  rms {np.sqrt((lin**2).mean()):.3e} max {np.abs(lin).max():.3e}"
        f"   -> {np.abs(lin).max() * 8e-3:.3e} m of orbit at dp=8e-3"
    )
    print(
        f"quadratic term (2nd-order disp err) rms {np.sqrt((quad**2).mean()):.3e} max {np.abs(quad).max():.3e}"
        f"   -> {np.abs(quad).max() * 64e-6:.3e} m of orbit at dp=8e-3"
    )

    R = response_matrix(model, bends, bpms)  # noqa: N806
    dk0_true = np.array([bend_k0_true[n] for n in bends]) - np.array(
        [nominal_k0(model, n) for n in bends]
    )

    def eval_fit(tag, dk0, disp_corr=None):
        corrected, _, _ = build_line()
        for n, dk in zip(bends, dk0):
            corrected[n].k0 = nominal_k0(corrected, n) + float(dk)
        print(f"\n== {tag}   dk0 rms err {np.sqrt(((dk0 - dk0_true) ** 2).mean()):.3e}")
        print("   dp       |CO err| max      rms")
        for i, dp in enumerate(DPS):
            pred = closed_orbit(corrected, dp, bpms)
            if disp_corr is not None:
                pred = pred + disp_corr[0] * dp + disp_corr[1] * dp**2
            res = np.abs(pred - co_mach[i])
            print(f"   {dp:+.0e}  {res.max():.3e}   {np.sqrt((res**2).mean()):.3e}")

    dk0_const = svd_solve(R, const, N_SV)
    dk0_raw0 = svd_solve(R, resid[0], N_SV)
    eval_fit("bends from dp=0 orbit only", dk0_raw0)
    eval_fit("bends from dp-fit CONSTANT term", dk0_const)

    # Second stage: the bend correction also changes the *model's* dispersion, so
    # the leftover dp-dependent term must be re-measured against the corrected
    # model rather than against the nominal one.
    corrected, _, _ = build_line()
    for n, dk in zip(bends, dk0_const):
        corrected[n].k0 = nominal_k0(corrected, n) + float(dk)
    resid2 = co_mach - np.array([closed_orbit(corrected, dp, bpms) for dp in DPS])
    c2, l2, q2 = np.polynomial.polynomial.polyfit(DPS, resid2, 2)
    print(
        f"\nleftover after bend fit: const rms {np.sqrt((c2**2).mean()):.3e}  "
        f"linear rms {np.sqrt((l2**2).mean()):.3e}  quad rms {np.sqrt((q2**2).mean()):.3e}"
    )
    eval_fit("constant-term bends + re-measured dispersion correction", dk0_const, (l2, q2))
    eval_fit(
        "constant-term bends + re-measured LINEAR term only", dk0_const, (l2, np.zeros_like(q2))
    )


if __name__ == "__main__":
    main()
