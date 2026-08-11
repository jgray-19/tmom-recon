"""Part C: establish the convention of MAD-NG's second-order dispersion columns
(``ddx``/``ddpx``) and measure what a second-order closed-orbit model buys.

Method: take the exact closed orbit MAD-NG solves at several ``pt`` and fit it
against the first- and second-order dispersion columns of the ``pt=0``,
``chrom=true`` twiss. The fitted coefficients ``c1``/``c2`` in

    x(pt) - x(0) = c1 * pt * dx  +  c2 * pt**2 * ddx

identify the normalisation directly: c1 == 1 means dx is per unit *pt*, c1 == beta
means it is per unit *delta*; c2 == 1 vs 0.5 settles the Taylor factor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pymadng_utils.accelerators import PSB

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.psb_tracking import (  # noqa: E402
    KINETIC_ENERGY_GEV,
    RING,
    SEQ_FILE,
    _apply_bend_errors_to_model,
    _apply_quad_errors_to_model,
)
from tmom_recon.acd.madng_driver import ACDipoleMadDriver  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
DELTAS = (1e-3, 3e-3, 8e-3, 1e-2)
WITH_ERRORS = "--errors" in sys.argv


def main():
    seq = DATA_DIR / "sequences" / SEQ_FILE
    acc = PSB(sequence_file=seq, ring=RING, kinetic_energy=KINETIC_ENERGY_GEV)
    model = ACDipoleMadDriver(accelerator=acc, pt=0.0, observed_elements=f"BR{RING}.DES3L1")
    if WITH_ERRORS:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from co_common import build_line

        _, bend_k0, quad_k1 = build_line(bend_rms=8e-4, bend_seed=7, quad_rms=1e-3, quad_seed=11)
        _apply_bend_errors_to_model(model, bend_k0)
        _apply_quad_errors_to_model(model, quad_k1)
        print("applied the machine's bend + quad errors to the MAD-NG model")

    tw0 = model.run_twiss(observe=0, chrom=True)
    cols = [c for c in ("dx", "dpx", "ddx", "ddpx", "dmu1", "wx", "phix") if c in tw0.columns]
    print("chrom columns present:", cols)
    bpms = [n for n in tw0.index if "BPM" in str(n).upper()]
    print(f"{len(bpms)} BPMs")

    beta = float(acc.beta) if hasattr(acc, "beta") else None
    print(f"beam beta = {beta}")

    x0 = tw0.loc[bpms, "x"].to_numpy(float)
    px0 = tw0.loc[bpms, "px"].to_numpy(float)
    dx = tw0.loc[bpms, "dx"].to_numpy(float)
    dpx = tw0.loc[bpms, "dpx"].to_numpy(float)
    ddx = tw0.loc[bpms, "ddx"].to_numpy(float)
    ddpx = tw0.loc[bpms, "ddpx"].to_numpy(float)

    print(f"\n{'delta':>8} {'pt':>10} | {'c1(x)':>9} {'c2(x)':>9} | {'c1(px)':>9} {'c2(px)':>9}")
    exact = {}
    for d in DELTAS:
        pt = acc.dp2pt(d)
        tw = model.run_twiss(observe=0, pt=pt)
        xe = tw.loc[bpms, "x"].to_numpy(float) - x0
        pxe = tw.loc[bpms, "px"].to_numpy(float) - px0
        exact[d] = (pt, xe + x0, pxe + px0)
        cx = np.linalg.lstsq(np.column_stack([pt * dx, pt**2 * ddx]), xe, rcond=None)[0]
        cp = np.linalg.lstsq(np.column_stack([pt * dpx, pt**2 * ddpx]), pxe, rcond=None)[0]
        print(f"{d:8.1e} {pt:10.3e} | {cx[0]:9.5f} {cx[1]:9.5f} | {cp[0]:9.5f} {cp[1]:9.5f}")

    print("\nresidual of each closed-orbit model vs the exact MAD-NG orbit (max|.| over BPMs)")
    print(f"{'delta':>8} | {'x: 1st':>10} {'x: 2nd':>10} | {'px: 1st':>10} {'px: 2nd':>10}")
    for d in DELTAS:
        pt, xe, pxe = exact[d]
        first_x = x0 + pt * dx
        second_x = x0 + pt * dx + pt**2 * ddx
        first_p = px0 + pt * dpx
        second_p = px0 + pt * dpx + pt**2 * ddpx
        print(
            f"{d:8.1e} | {np.abs(first_x - xe).max():10.3e} "
            f"{np.abs(second_x - xe).max():10.3e} | "
            f"{np.abs(first_p - pxe).max():10.3e} {np.abs(second_p - pxe).max():10.3e}"
        )

    # Chromatic optics: how much do beta/phase move over the pt range? This is the
    # source of the *linear-in-pt gain* error documented in the notes.
    print("\nchromatic optics at pt(delta):")
    for d in DELTAS:
        pt = acc.dp2pt(d)
        tw = model.run_twiss(observe=0, pt=pt, chrom=True)
        for plane, bcol, mcol in (("x", "beta11", "mu1"), ("y", "beta22", "mu2")):
            if bcol not in tw0.columns:
                print("  missing", bcol, list(tw0.columns)[:25])
                break
            b0 = tw0.loc[bpms, bcol].to_numpy(float)
            b1 = tw.loc[bpms, bcol].to_numpy(float)
            mu0 = tw0.loc[bpms, mcol].to_numpy(float)
            mu1 = tw.loc[bpms, mcol].to_numpy(float)
            print(
                f"  delta={d:.1e} {plane}: max|dbet/bet|={np.abs(b1 / b0 - 1).max():.3e} "
                f"max|dmu|={np.abs(mu1 - mu0).max():.3e}"
            )
        if "dmu1" in tw0.columns:
            dmu1 = tw0.loc[bpms, "dmu1"].to_numpy(float)
            mu0 = tw0.loc[bpms, "mu1"].to_numpy(float)
            mu1 = tw.loc[bpms, "mu1"].to_numpy(float)
            pred = pt * dmu1
            print(
                f"    dmu1 prediction: c1={np.linalg.lstsq(pred[:, None], mu1 - mu0, rcond=None)[0][0]:.5f}"
                f"  residual={np.abs(mu0 + pred - mu1).max():.3e} vs raw {np.abs(mu1 - mu0).max():.3e}"
            )


if __name__ == "__main__":
    main()
