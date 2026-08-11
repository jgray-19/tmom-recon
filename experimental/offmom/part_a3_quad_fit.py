"""Part A3: also fit the quadrupole gradient errors, from the *dispersion* residual.

The bend fit (Part A/A2) leaves a dp-dependent residual because the model's
dispersion is wrong: the machine's quad gradient errors are unknown. Part A2 could
only correct that as a per-BPM empirical table of (x, y) — which gives no closed-orbit
*momenta*, the quantity the reconstruction actually needs. Fitting the residual to a
quadrupole gradient response instead produces a genuine lattice, so px/py follow.

Unknowns: relative k1 error per ring quadrupole. Observations: the per-BPM linear and
quadratic dp coefficients of the orbit residual, plus the dp=0 orbit (bends).
Solved by alternating the two SVD fits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from co_common import bend_names, bpm_names, build_line, closed_orbit, nominal_k0  # noqa: E402
from part_a_bend_fit import BEND_RMS, QUAD_RMS, response_matrix, svd_solve  # noqa: E402

DPS = np.array([0.0, 1e-3, -1e-3, 3e-3, -3e-3, 8e-3, -8e-3])
N_SV_BEND = 17
FIT_DPS = (8e-3, -8e-3)
N_SV_QUAD = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def quad_names(line):
    out = []
    for n in line.element_names:
        if not str(n).lower().startswith("br.q"):
            continue
        k1 = getattr(line[n], "k1", None)
        if k1 is None or float(k1) == 0.0:
            continue
        out.append(n)
    return out


def quad_response(line, quads, bpms, dps, step=1e-4):
    """d(orbit at *dps*, stacked)/d(relative k1 error), one column per quad."""
    cols = []
    for name in quads:
        k1 = float(line[name].k1)
        line[name].k1 = k1 * (1 + step)
        plus = np.concatenate([closed_orbit(line, dp, bpms) for dp in dps])
        line[name].k1 = k1 * (1 - step)
        minus = np.concatenate([closed_orbit(line, dp, bpms) for dp in dps])
        line[name].k1 = k1
        cols.append((plus - minus) / (2 * step))
    return np.column_stack(cols)


def main():
    machine, _, _ = build_line(bend_rms=BEND_RMS, quad_rms=QUAD_RMS)
    model, _, _ = build_line()
    bpms, bends, quads = bpm_names(model), bend_names(model), quad_names(model)
    print(f"{len(bends)} bends, {len(quads)} quads, {len(bpms)} BPMs")

    co_mach = {dp: closed_orbit(machine, dp, bpms) for dp in DPS}
    Rb = response_matrix(model, bends, bpms)  # noqa: N806
    Rq = quad_response(model, quads, bpms, FIT_DPS)  # noqa: N806
    print(
        "quad-response singular values:",
        np.array2string(np.linalg.svd(Rq, compute_uv=False)[:12], precision=2),
    )

    dk0 = np.zeros(len(bends))
    dq = np.zeros(len(quads))

    def make(dk0, dq):
        ln, _, _ = build_line()
        for n, d in zip(bends, dk0):
            ln[n].k0 = nominal_k0(ln, n) + float(d)
        for n, d in zip(quads, dq):
            ln[n].k1 = float(ln[n].k1) * (1 + float(d))
        return ln

    for it in range(4):
        cur = make(dk0, dq)
        dk0 = dk0 + svd_solve(Rb, co_mach[0.0] - closed_orbit(cur, 0.0, bpms), N_SV_BEND)
        cur = make(dk0, dq)
        b = np.concatenate([co_mach[dp] - closed_orbit(cur, dp, bpms) for dp in FIT_DPS])
        dq = dq + svd_solve(Rq, b, N_SV_QUAD)
        cur = make(dk0, dq)
        errs = {dp: np.abs(closed_orbit(cur, dp, bpms) - co_mach[dp]) for dp in DPS}
        print(
            f"\nn_sv_quad={N_SV_QUAD} iter {it}: |CO err| by dp  "
            + "  ".join(f"{dp:+.0e}:{errs[dp].max():.2e}" for dp in DPS)
        )

    np.savez(
        Path(__file__).with_name(f"fitted_errors_q{N_SV_QUAD}.npz"),
        bends=np.array(bends),
        dk0=dk0,
        quads=np.array(quads),
        dq=dq,
    )
    print("\nsaved fitted_errors.npz")


if __name__ == "__main__":
    main()
