"""Part A: estimate bend field errors from closed orbits measured at several delta_p.

Machine  = xsuite line with bend field errors AND quad gradient errors.
Model    = nominal xsuite line (knows neither).
Estimator: linear orbit response matrix R (d(orbit at BPMs) / d(k0 of each powered
half-bend), computed on the *model* by finite difference at dp=0), inverted with a
truncated SVD against the measured-minus-model orbit residual, stacked over the
measured delta_p values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from co_common import bend_names, bpm_names, build_line, closed_orbit, nominal_k0  # noqa: E402

BEND_RMS = 8e-4
QUAD_RMS = 1e-3
DPS = (0.0, 1e-3, -1e-3, 3e-3, -3e-3, 8e-3, -8e-3)


def response_matrix(line, bends, bpms, dp=0.0, step=1e-5):
    """d(orbit)/d(k0) at *bpms*, one column per bend, by central difference."""
    cols = []
    for name in bends:
        k0 = nominal_k0(line, name)
        line[name].k0 = k0 + step
        plus = closed_orbit(line, dp, bpms)
        line[name].k0 = k0 - step
        minus = closed_orbit(line, dp, bpms)
        line[name].k0 = k0
        cols.append((plus - minus) / (2 * step))
    return np.column_stack(cols)


def svd_solve(R, b, n_sv):  # noqa: N803  (R: response matrix)
    u, s, vt = np.linalg.svd(R, full_matrices=False)
    inv = np.zeros_like(s)
    inv[:n_sv] = 1.0 / s[:n_sv]
    return vt.T @ (inv * (u.T @ b))


def main():
    machine, bend_k0_true, _ = build_line(bend_rms=BEND_RMS, quad_rms=QUAD_RMS)
    machine_bendonly, _, _ = build_line(bend_rms=BEND_RMS, quad_rms=0.0)
    model, _, _ = build_line()

    bpms = bpm_names(model)
    bends = bend_names(model)
    print(f"{len(bpms)} BPMs -> {2 * len(bpms)} observations per dp; {len(bends)} bend unknowns")

    k0_nom = np.array([nominal_k0(model, n) for n in bends])
    dk0_true = np.array([bend_k0_true[n] for n in bends]) - k0_nom

    print("building response matrix ...")
    R = response_matrix(model, bends, bpms)  # noqa: N806
    sv = np.linalg.svd(R, compute_uv=False)
    print("singular values:", np.array2string(sv, precision=3, max_line_width=200))

    co_model = {dp: closed_orbit(model, dp, bpms) for dp in DPS}
    co_mach = {dp: closed_orbit(machine, dp, bpms) for dp in DPS}
    co_mach_bo = {dp: closed_orbit(machine_bendonly, dp, bpms) for dp in DPS}

    def fit(dps, meas, n_sv):
        Rs = np.vstack([R] * len(dps))  # noqa: N806
        bs = np.concatenate([meas[dp] - co_model[dp] for dp in dps])
        return svd_solve(Rs, bs, n_sv)

    def report(tag, dk0, target=None):
        target = co_mach if target is None else target
        corrected, _, _ = build_line()
        for n, dk in zip(bends, dk0):
            corrected[n].k0 = nominal_k0(corrected, n) + float(dk)
        rows = []
        for dp in DPS:
            res = np.abs(closed_orbit(corrected, dp, bpms) - target[dp])
            rows.append((dp, res.max(), np.sqrt((res**2).mean())))
        kerr = np.sqrt(((dk0 - dk0_true) ** 2).mean())
        print(f"\n== {tag}")
        print(
            f"  dk0 rms error {kerr:.3e} (true dk0 rms {dk0_true.std():.3e}), "
            f"recovered fraction {1 - kerr / dk0_true.std():+.3f}"
        )
        print(f"  per-bend corr = {np.corrcoef(dk0, dk0_true)[0, 1]:.4f}")
        print("   dp        |CO err| max     rms")
        for dp, mx, rms in rows:
            print(f"   {dp:+.0e}   {mx:.3e}   {rms:.3e}")
        return corrected

    print("\n### uncorrected model")
    for dp in DPS:
        res = np.abs(co_model[dp] - co_mach[dp])
        print(f"   {dp:+.0e}   {res.max():.3e}   {np.sqrt((res**2).mean()):.3e}")

    for n_sv in (6, 8, 10, 12, 14, 16, 17):
        fit_all = fit(DPS, co_mach, n_sv)
        report(f"all dp, n_sv={n_sv}", fit_all)

    report("dp=0 only, n_sv=16", fit((0.0,), co_mach, 16))
    report("dp=+-8e-3 only, n_sv=16", fit((8e-3, -8e-3), co_mach, 16))
    report("NO QUAD ERRORS (control), all dp, n_sv=16", fit(DPS, co_mach_bo, 16), target=co_mach_bo)
    report(
        "NO QUAD ERRORS (control), dp=0 only, n_sv=16",
        fit((0.0,), co_mach_bo, 16),
        target=co_mach_bo,
    )


if __name__ == "__main__":
    main()
