"""Study 4 — BPM noise.

BPM resolution feeds straight through the phase-space reconstruction. We track a
few hundred turns and reconstruct with Gaussian noise of increasing RMS injected
into ``x, y``, taking the reconstruction RMSE over all turns/BPMs. The whole
sweep is repeated over many seeds to get a mean and a spread, and run both
**raw** and with **SVD cleaning** to show how much temporal-mode cleaning buys
back. Run on a realistic FODO (bends + sextupoles on) and on the LHC and PSB.

Outputs: study/plots/04_bpm_noise_{lattice}_{abs,rel}.{pdf,png},
         study/results/04_bpm_noise.csv
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.metrics import reconstruct, score
from study.plotting import PX_COLOR, PY_COLOR, save_fig, set_paper_style
from study.xsuite_tracking import bpm_twiss, load_ring, make_fodo, set_fodo_knobs, track_free

logging.basicConfig(level=logging.ERROR)

NTURN = 400
NOISE_STDS = np.logspace(-6, -3, 8)  # BPM resolution sweep [m]
SEEDS = range(16)

LATTICES = {
    "fodo": {"action": 3e-7, "label": "FODO (bends+sextupoles)"},
    "lhc": {"action": 5e-9, "label": "LHC B1"},
    "psb": {"action": 5e-6, "label": "PSB ring 3"},
}


def _build(name):
    if name == "fodo":
        iface = make_fodo()
        set_fodo_knobs(iface, bangle=0.05, ksf=1.0, ksd=-1.0)
        return iface
    return load_ring(name)


def _sweep(trk, tws, *, svd):
    """Return a DataFrame of mean/std RMSE vs noise std (seed-averaged)."""
    rows = []
    for noise in NOISE_STDS:
        px, py, pxs, pys = [], [], [], []
        for seed in SEEDS:
            merged = reconstruct(
                trk,
                tws,
                use_dispersion=False,
                noise_std=float(noise),
                rng=np.random.default_rng(seed),
                svd=svd,
            )
            s = score(merged)
            px.append(s["px_rmse"])
            py.append(s["py_rmse"])
            pxs.append(s["px_scale"])
            pys.append(s["py_scale"])
        rows.append(
            {
                "noise_std": noise,
                "px_rmse": np.mean(px),
                "px_std": np.std(px),
                "px_scale": np.mean(pxs),
                "py_rmse": np.mean(py),
                "py_std": np.std(py),
                "py_scale": np.mean(pys),
            }
        )
    return pd.DataFrame(rows)


def _plot(name, label, raw, clean):
    for relative, suffix, ylabel in (
        (False, "abs", "reconstruction RMSE [rad]"),
        (True, "rel", r"relative error  RMSE / $\sigma_p$"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        for df, style, tag in ((raw, "-", "raw"), (clean, "--", "SVD")):
            sx = df["px_scale"] if relative else 1.0
            sy = df["py_scale"] if relative else 1.0
            ax.errorbar(
                df["noise_std"],
                df["px_rmse"] / sx,
                yerr=df["px_std"] / sx,
                fmt=f"{style}o",
                color=PX_COLOR,
                capsize=3,
                label=rf"$p_x$ ({tag})",
            )
            ax.errorbar(
                df["noise_std"],
                df["py_rmse"] / sy,
                yerr=df["py_std"] / sy,
                fmt=f"{style}s",
                color=PY_COLOR,
                capsize=3,
                label=rf"$p_y$ ({tag})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("BPM noise RMS [m]")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{label}: reconstruction error vs BPM noise")
        ax.legend(ncol=2, fontsize=9)
        save_fig(fig, f"04_bpm_noise_{name}_{suffix}")


def main() -> None:
    set_paper_style()
    all_rows = []
    for name, cfg in LATTICES.items():
        iface = _build(name)
        tws = bpm_twiss(iface)
        trk = track_free(iface, action=cfg["action"], nturn=NTURN, dp=0.0)
        iface.close()
        raw = _sweep(trk, tws, svd=False)
        clean = _sweep(trk, tws, svd=True)
        _plot(name, cfg["label"], raw, clean)
        for df, tag in ((raw, "raw"), (clean, "svd")):
            d = df.copy()
            d.insert(0, "lattice", name)
            d.insert(1, "method", tag)
            all_rows.append(d)
        print(
            f"[{name}] noise {NOISE_STDS[0]:.0e}->{NOISE_STDS[-1]:.0e}: "
            f"px raw {raw['px_rmse'].iloc[0]:.2e}->{raw['px_rmse'].iloc[-1]:.2e}"
        )

    pd.concat(all_rows, ignore_index=True).to_csv("study/results/04_bpm_noise.csv", index=False)
    print("wrote study/results/04_bpm_noise.csv")


if __name__ == "__main__":
    main()
