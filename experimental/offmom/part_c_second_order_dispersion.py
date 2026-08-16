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
COORDS = (
    ("x", "dx", "ddx"),
    ("px", "dpx", "ddpx"),
    ("y", "dy", "ddy"),
    ("py", "dpy", "ddpy"),
)


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
    # Print what chrom=true actually produced rather than intersecting with a
    # guessed wishlist -- an earlier run of this script filtered against a list
    # containing `phix` and concluded dmu2/ddy/ddpy were absent. They are not;
    # MAD-NG just names the Montague phase `wxp`/`wyp`.
    print("chrom columns present:", list(tw0.columns))
    bpms = [n for n in tw0.index if "BPM" in str(n).upper()]
    print(f"{len(bpms)} BPMs")

    beta = float(acc.beta) if hasattr(acc, "beta") else None
    print(f"beam beta = {beta}")

    # (coordinate, first-order column, second-order column) for both planes. The
    # vertical set exists too; on an uncoupled PSB with no vertical dispersion it
    # is numerically zero, which is itself worth showing rather than assuming.
    ref = {c: tw0.loc[bpms, c].to_numpy(float) for c, _, _ in COORDS}
    d1 = {c: tw0.loc[bpms, f1].to_numpy(float) for c, f1, _ in COORDS}
    d2 = {c: tw0.loc[bpms, f2].to_numpy(float) for c, _, f2 in COORDS}
    for c, f1, f2 in COORDS:
        print(f"  max|{f1}|={np.abs(d1[c]).max():.3e}  max|{f2}|={np.abs(d2[c]).max():.3e}")

    header = " | ".join(f"{'c1(' + c + ')':>9} {'c2(' + c + ')':>9}" for c, _, _ in COORDS)
    print(f"\n{'delta':>8} {'pt':>10} | {header}")
    exact = {}
    for d in DELTAS:
        pt = acc.dp2pt(d)
        tw = model.run_twiss(observe=0, pt=pt)
        exact[d] = (pt, {c: tw.loc[bpms, c].to_numpy(float) for c, _, _ in COORDS})
        cells = []
        for c, _, _ in COORDS:
            dev = exact[d][1][c] - ref[c]
            # A plane with no dispersion at all makes the design matrix singular;
            # lstsq would return a meaningless 0. Report it as such.
            if np.abs(d1[c]).max() == 0.0 and np.abs(d2[c]).max() == 0.0:
                cells.append(f"{'--':>9} {'--':>9}")
                continue
            cc = np.linalg.lstsq(np.column_stack([pt * d1[c], pt**2 * d2[c]]), dev, rcond=None)[0]
            cells.append(f"{cc[0]:9.5f} {cc[1]:9.5f}")
        print(f"{d:8.1e} {pt:10.3e} | " + " | ".join(cells))

    print("\nresidual of each closed-orbit model vs the exact MAD-NG orbit (max|.| over BPMs)")
    header = " | ".join(f"{c + ': 1st':>10} {c + ': 2nd':>10}" for c, _, _ in COORDS)
    print(f"{'delta':>8} | {header}")
    for d in DELTAS:
        pt, ex = exact[d]
        cells = []
        for c, _, _ in COORDS:
            first = ref[c] + pt * d1[c]
            second = first + pt**2 * d2[c]
            cells.append(
                f"{np.abs(first - ex[c]).max():10.3e} {np.abs(second - ex[c]).max():10.3e}"
            )
        print(f"{d:8.1e} | " + " | ".join(cells))

    # Chromatic optics: how much do beta/phase move over the pt range? This is the
    # source of the *linear-in-pt gain* error documented in the notes.
    print("\nchromatic optics at pt(delta):")
    for d in DELTAS:
        pt = acc.dp2pt(d)
        tw = model.run_twiss(observe=0, pt=pt, chrom=True)
        for plane, bcol, mcol, dcol in (
            ("x", "beta11", "mu1", "dmu1"),
            ("y", "beta22", "mu2", "dmu2"),
        ):
            b0 = tw0.loc[bpms, bcol].to_numpy(float)
            b1 = tw.loc[bpms, bcol].to_numpy(float)
            mu_0 = tw0.loc[bpms, mcol].to_numpy(float)
            mu_pt = tw.loc[bpms, mcol].to_numpy(float)
            print(
                f"  delta={d:.1e} {plane}: max|dbet/bet|={np.abs(b1 / b0 - 1).max():.3e} "
                f"max|dmu|={np.abs(mu_pt - mu_0).max():.3e}"
            )
            # mu(pt) = mu(0) + pt * dmu, per unit pt (c1 -> 1, not 1/beta).
            dmu = tw0.loc[bpms, dcol].to_numpy(float)
            pred = pt * dmu
            raw = np.abs(mu_pt - mu_0).max()
            c1 = np.linalg.lstsq(pred[:, None], mu_pt - mu_0, rcond=None)[0][0]
            print(
                f"    {dcol} prediction: c1={c1:.5f}"
                f"  residual={np.abs(mu_0 + pred - mu_pt).max():.3e} vs raw {raw:.3e}"
            )


if __name__ == "__main__":
    main()
