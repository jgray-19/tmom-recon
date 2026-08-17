"""Why the Part B reconstruction plateaus: the closed-orbit fit does not constrain
the optics. Quantifies beta-beating and phase error of each fitted model against
the machine, alongside the closed-orbit error it was fitted to."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from co_common import bpm_names, build_line, closed_orbit, nominal_k0  # noqa: E402
from part_a_bend_fit import BEND_RMS, QUAD_RMS  # noqa: E402


def load(npz, quads=True):
    d = np.load(Path(__file__).with_name(npz), allow_pickle=True)
    line, _, _ = build_line()
    for n, dk in zip([str(x) for x in d["bends"]], d["dk0"]):
        line[n].k0 = nominal_k0(line, n) + float(dk)
    if quads:
        for n, dq in zip([str(x) for x in d["quads"]], d["dq"]):
            line[n].k1 = float(line[n].k1) * (1 + float(dq))
    return line


def main():
    machine, _, _ = build_line(bend_rms=BEND_RMS, quad_rms=QUAD_RMS)
    bpms = bpm_names(machine)
    tm = machine.twiss(method="4d")
    rm = tm.rows[bpms]
    co_m = closed_orbit(machine, 8e-3, bpms)
    px_m = np.asarray(machine.twiss(method="4d", delta0=8e-3).rows[bpms].px)

    print(f"machine tunes qx={tm.qx:.5f} qy={tm.qy:.5f}\n")
    print(
        f"{'model':<22} {'|CO err|@8e-3':>13} {'dqx':>9} {'dqy':>9} "
        f"{'max|dbetx/betx|':>15} {'max|dmux| [2pi]':>15} {'max|px_co err|':>14}"
    )
    models = [
        ("nominal", build_line()[0]),
        ("bend-only fit", load("fitted_errors.npz", quads=False)),
        ("bend+quad q8", load("fitted_errors_q8.npz")),
        ("bend+quad q12", load("fitted_errors.npz")),
        ("bend+quad q16", load("fitted_errors_q16.npz")),
        ("TRUE errors", machine),
    ]
    for label, line in models:
        t = line.twiss(method="4d")
        r = t.rows[bpms]
        dbet = np.abs(np.asarray(r.betx) / np.asarray(rm.betx) - 1).max()
        dmu = np.asarray(r.mux) - np.asarray(rm.mux)
        dmu = np.abs(dmu - dmu[0]).max()
        co = np.abs(closed_orbit(line, 8e-3, bpms) - co_m).max()
        tw8 = line.twiss(method="4d", delta0=8e-3).rows[bpms]
        dpx_co = np.abs(np.asarray(tw8.px) - px_m).max()
        print(
            f"{label:<22} {co:13.2e} {t.qx - tm.qx:+9.5f} {t.qy - tm.qy:+9.5f} "
            f"{dbet:15.3e} {dmu:15.3e} {dpx_co:14.3e}"
        )


if __name__ == "__main__":
    main()
