"""Study 1 — non-linearities and momentum offset in a clean FODO.

Demonstrates the best-case reconstruction (perfectly linear, on-momentum,
perfectly known optics → numerical floor) and how it degrades from two intrinsic
sources the *linear* model cannot capture:

* (a) sextupoles — genuine non-linearity; error grows with sextupole strength
  (and with betatron amplitude, the non-linear signature).
* (b) momentum offset — with the bends ON (so dispersion is active), an
  off-momentum particle samples chromatic phase errors the on-momentum model
  ignores; error grows ~linearly with the momentum offset ``pt``.

The model twiss is re-computed for each configuration, so the residual error is
purely what the linear reconstruction cannot capture — not an optics-knowledge
error (that is Study 6). The bend *radius* dependence is Study 2.

Outputs: study/plots/01_fodo_nonlinearities.{pdf,png}, study/results/01_*.csv
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.metrics import reconstruct, score
from study.plotting import plot_planes, save_fig, set_paper_style
from study.xsuite_tracking import bpm_twiss, make_fodo, set_fodo_knobs, track_free

logging.basicConfig(level=logging.WARNING)
NTURN = 256
ACTION = 3e-7  # Courant-Snyder action [m·rad]; larger amplitude exposes non-linearity


def _rmse_at(iface, *, dp, use_dispersion):
    pt = iface.dp2pt(dp)
    tws = bpm_twiss(iface)
    trk = track_free(iface, action=ACTION, nturn=NTURN, dp=dp)
    # pt is known in this controlled study, so override the estimator and isolate
    # the dispersion-*modelling* error from pt-*estimation* error.
    merged = reconstruct(
        trk, tws, use_dispersion=use_dispersion, pt_override=pt if use_dispersion else None
    )
    return score(merged)


def main() -> None:
    set_paper_style()
    iface = make_fodo()

    # Baseline: linear, on-momentum, straight ring -> numerical floor.
    base = _rmse_at(iface, dp=0.0, use_dispersion=False)
    print(
        f"[baseline] px RMSE={base['px_rmse']:.2e}  py RMSE={base['py_rmse']:.2e} "
        f"(signal ~{base['px_scale']:.2e})"
    )

    # Sextupole scan (on-momentum): pure non-linearity.
    sext_vals = np.linspace(0.0, 3.0, 9)
    sext_rows = []
    for k2 in sext_vals:
        set_fodo_knobs(iface, ksf=k2, ksd=-k2)
        s = _rmse_at(iface, dp=0.0, use_dispersion=False)
        sext_rows.append({"k2": k2, **s})
        print(f"[sext] k2={k2:.2f}  px={s['px_rmse']:.2e}  py={s['py_rmse']:.2e}")
    set_fodo_knobs(iface, ksf=0.0, ksd=0.0)

    # Momentum-offset scan (bends ON so dispersion is active): chromatic error.
    # delta in [1e-4, 1e-3] is the realistic operating range; converted to pt
    # inside track_free / dp2pt.
    set_fodo_knobs(iface, bangle=0.05)
    dp_vals = np.linspace(1e-4, 1e-3, 9)
    pt_rows = []
    for dp in dp_vals:
        s = _rmse_at(iface, dp=dp, use_dispersion=True)
        pt_rows.append({"dp": dp, **s})
        print(f"[dp] dp={dp:.2e}  px={s['px_rmse']:.2e}  py={s['py_rmse']:.2e}")
    set_fodo_knobs(iface, bangle=0.0)
    iface.close()

    sext_df = pd.DataFrame(sext_rows)
    pt_df = pd.DataFrame(pt_rows)
    sext_df.to_csv("study/results/01_sextupole_scan.csv", index=False)
    pt_df.to_csv("study/results/01_momentum_scan.csv", index=False)

    # Two figures (absolute + relative), each with the two scan panels.
    for relative, suffix, ylabel in (
        (False, "abs", "reconstruction RMSE [rad]"),
        (True, "rel", r"relative error  RMSE / $\sigma_{p}$"),
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        plot_planes(ax1, sext_df["k2"], sext_df, relative=relative, baseline=base)
        ax1.set_xlabel(r"sextupole strength $k_2$ [m$^{-3}$]")
        ax1.set_ylabel(ylabel)
        ax1.set_title("(a) non-linearity (on-momentum)")
        ax1.legend()

        plot_planes(ax2, pt_df["dp"], pt_df, relative=relative, baseline=base)
        ax2.set_xlabel(r"momentum offset $\delta = \Delta p/p$")
        ax2.set_title("(b) momentum offset (bends on)")
        ax2.legend()

        kind = "relative" if relative else "absolute"
        fig.suptitle(f"FODO: {kind} reconstruction error vs non-linearity and momentum offset")
        save_fig(fig, f"01_fodo_nonlinearities_{suffix}")
    print("wrote study/plots/01_fodo_nonlinearities_{abs,rel}.pdf")


if __name__ == "__main__":
    main()
