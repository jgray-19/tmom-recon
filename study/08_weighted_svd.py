"""Study 8 — weighted vs plain SVD under heteroscedastic (per-BPM) noise.

Real BPMs do not share one resolution: on the LHC the per-BPM turn-by-turn noise
clusters near ~0.1 mm but spans roughly a factor of ten, with a tail of noisy
monitors (see ``noise_investigation/bpm_std.txt``). A *plain* truncated SVD treats
every BPM column equally, so a few noisy BPMs leak into the retained modes; a
*weighted* SVD whitens each column by its (known) variance first, down-weighting
the noisy BPMs. With *uniform* noise the two are identical, so the per-BPM spread
is exactly what should make weighted SVD win.

This is the study that most needs the xsuite tracking backend: SVD denoising only
pays off over many turns, and xsuite affords thousands cheaply.

For each lattice (FODO, LHC, PSB) we track once (no noise), then for a swept noise
level draw a per-BPM sigma (log-normal spread + a noisy-BPM tail), inject it, and
reconstruct three ways — **raw** (no clean), **plain SVD**, **weighted SVD** —
scoring the per-plane reconstruction RMSE, averaged over noise seeds. A second
figure fixes the noise level and sweeps the per-BPM *spread* to show the two SVDs
coincide at spread=1 and diverge as the heterogeneity grows.

Outputs: study/plots/08_wsvd_{lattice}_{level,spread}_{abs,rel}.{pdf,png},
         study/results/08_weighted_svd.csv
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study.metrics import (
    draw_per_bpm_sigma,
    inject_per_bpm_noise,
    per_bpm_method_rmse,
    reconstruct_cleaned,
    rmse,
)
from study.plotting import PALETTE, save_fig, set_paper_style
from study.xsuite_tracking import (
    LATTICE_LABEL,
    LATTICES,
    bpm_twiss,
    build_lattice,
    set_fodo_knobs,
    track_free,
)

logging.basicConfig(level=logging.ERROR)

# Many turns are cheap with xsuite and are what SVD denoising needs. The LHC has
# 563 BPMs (~0.1 s/turn) so it gets fewer than the small rings.
NTURN = {"fodo": 2000, "lhc": 400, "psb": 2000}
ACTION = {"fodo": 3e-7, "lhc": 5e-9, "psb": 5e-6}

# Per-BPM noise model (metres): median resolution swept; geometric spread and a
# noisy-BPM tail grounded in the measured LHC distribution (~0.1 mm, ~x7 spread).
NOISE_LEVELS = np.logspace(-5, -3, 7)  # median per-BPM sigma [m]
LEVEL_SPREAD = 3.0  # geometric spread at fixed-level sweep
SPREADS = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 8.0])  # spread sweep (fixed level)
SPREAD_LEVEL = 1e-4  # median sigma for the spread sweep [m]
BAD_FRACTION = 0.1  # 10% noisy BPMs ...
BAD_FACTOR = 5.0  # ... at 5x the resolution
SEEDS = range(12)

# Full-ring per-BPM comparison: one realistic noise setting, error around the ring.
RING_MEDIAN = 1e-4  # median per-BPM resolution [m]
RING_SPREAD = 3.0  # geometric per-BPM spread
RING_SEEDS = range(8)

METHODS = [
    ("none", "raw", PALETTE["black"], ":"),
    ("svd", "plain SVD", PALETTE["orange"], "--"),
    ("wsvd", "weighted SVD", PALETTE["green"], "-"),
]


def _score_methods(trk, tws, *, median, spread, bad_fraction):
    """Mean/std per-plane RMSE over seeds for raw / plain-SVD / weighted-SVD."""
    out = {m: {"px": [], "py": []} for m, *_ in METHODS}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sigma = draw_per_bpm_sigma(
            trk["name"].to_numpy(),
            rng,
            median=median,
            spread=spread,
            bad_fraction=bad_fraction,
            bad_factor=BAD_FACTOR,
        )
        data = inject_per_bpm_noise(trk, rng, sigma)
        for method, *_ in METHODS:
            m = reconstruct_cleaned(trk, tws, data, method=method, use_dispersion=False)
            out[method]["px"].append(rmse(m["px_true"], m["px"]))
            out[method]["py"].append(rmse(m["py_true"], m["py"]))
    row = {"px_scale": float(trk["px"].std()), "py_scale": float(trk["py"].std())}
    for method, *_ in METHODS:
        for p in ("px", "py"):
            row[f"{method}_{p}_rmse"] = float(np.mean(out[method][p]))
            row[f"{method}_{p}_std"] = float(np.std(out[method][p]))
    return row


def _plot(name, label, df, xcol, xlabel, *, xscale, tag, title):
    for relative, suffix, ylabel in (
        (False, "abs", "reconstruction RMSE [rad]"),
        (True, "rel", r"relative error  RMSE / $\sigma_p$"),
    ):
        fig, (axx, axy) = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
        for ax, plane, pl in ((axx, "px", r"$p_x$"), (axy, "py", r"$p_y$")):
            scale = df[f"{plane}_scale"] if relative else 1.0
            for method, mlabel, color, ls in METHODS:
                y = df[f"{method}_{plane}_rmse"] / scale
                yerr = df[f"{method}_{plane}_std"] / scale
                ax.errorbar(
                    df[xcol], y, yerr=yerr, fmt=f"{ls}o", color=color, capsize=3, label=mlabel
                )
            ax.set_xscale(xscale)
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_title(pl)
            ax.legend(fontsize=9)
        axx.set_ylabel(ylabel)
        fig.suptitle(f"{label}: {title}")
        save_fig(fig, f"08_wsvd_{name}_{tag}_{suffix}")


def _plot_ring(name, label, df, n_bpm, nturn):
    """Per-BPM reconstruction error around the full ring: raw vs plain vs weighted SVD."""
    for relative, suffix, ylabel in (
        (False, "abs", "per-BPM RMSE [rad]"),
        (True, "rel", r"per-BPM relative error  RMSE / $\sigma_p$"),
    ):
        fig, (axx, axy) = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True)
        for ax, plane, pl in ((axx, "px", r"$p_x$"), (axy, "py", r"$p_y$")):
            scale = float(df[f"{plane}_scale"].iloc[0]) if relative else 1.0
            for method, mlabel, color, ls in METHODS:
                ax.plot(
                    df["s"],
                    df[f"{method}_{plane}_rmse"] / scale,
                    ls,
                    color=color,
                    lw=1.4,
                    marker="o",
                    ms=2.5,
                    label=mlabel,
                )
            ax.set_yscale("log")
            ax.set_ylabel(f"{pl}  " + ylabel)
            ax.legend(fontsize=9, ncol=3)
        axy.set_xlabel("longitudinal position $s$ [m]")
        axx.set_title(
            f"{label}: full-ring per-BPM error, weighted vs plain SVD "
            f"({n_bpm} BPMs, {nturn} turns, median noise {RING_MEDIAN:.0e} m)"
        )
        save_fig(fig, f"08_wsvd_{name}_ring_{suffix}")


def main() -> None:
    set_paper_style()
    all_rows = []
    for lat in LATTICES:
        if lat == "fodo":
            xl = build_lattice("fodo")
            set_fodo_knobs(xl, bangle=0.05, ksf=1.0, ksd=-1.0)  # realistic, non-linear
        else:
            xl = build_lattice(lat)
        tws = bpm_twiss(xl)
        trk = track_free(xl, action=ACTION[lat], nturn=NTURN[lat], dp=0.0)
        xl.close()
        n_bpm = trk["name"].nunique()

        # (0) full-ring per-BPM comparison at a fixed realistic noise setting.
        ring = per_bpm_method_rmse(
            trk,
            tws,
            [m for m, *_ in METHODS],
            median=RING_MEDIAN,
            spread=RING_SPREAD,
            bad_fraction=BAD_FRACTION,
            bad_factor=BAD_FACTOR,
            seeds=RING_SEEDS,
        )
        ring["px_scale"] = float(trk["px"].std())
        ring["py_scale"] = float(trk["py"].std())
        _plot_ring(lat, LATTICE_LABEL[lat], ring, n_bpm, NTURN[lat])
        ring_csv = ring.reset_index()
        ring_csv.insert(0, "lattice", lat)
        ring_csv.to_csv(f"study/results/08_wsvd_{lat}_ring.csv", index=False)
        print(
            f"[{lat}/ring] svd px {ring['svd_px_rmse'].mean():.2e} "
            f"-> wsvd px {ring['wsvd_px_rmse'].mean():.2e}"
        )

        # (a) sweep the median noise level at a fixed per-BPM spread.
        rows_level = []
        for level in NOISE_LEVELS:
            r = _score_methods(
                trk, tws, median=float(level), spread=LEVEL_SPREAD, bad_fraction=BAD_FRACTION
            )
            rows_level.append({"noise_level": float(level), **r})
            print(
                f"[{lat}/level] s={level:.1e}  raw={r['none_px_rmse']:.2e} "
                f"svd={r['svd_px_rmse']:.2e} wsvd={r['wsvd_px_rmse']:.2e}"
            )
        df_level = pd.DataFrame(rows_level)
        _plot(
            lat,
            LATTICE_LABEL[lat],
            df_level,
            "noise_level",
            "median per-BPM noise [m]",
            xscale="log",
            tag="level",
            title=f"weighted vs plain SVD vs noise ({n_bpm} BPMs, {NTURN[lat]} turns)",
        )

        # (b) sweep the per-BPM spread at a fixed median noise level.
        rows_spread = []
        for spread in SPREADS:
            # No noisy-BPM tail here, so spread=1 is genuinely uniform noise where
            # weighted and plain SVD must coincide; the divergence is purely the spread.
            r = _score_methods(
                trk, tws, median=SPREAD_LEVEL, spread=float(spread), bad_fraction=0.0
            )
            rows_spread.append({"spread": float(spread), **r})
            print(
                f"[{lat}/spread] x{spread:.1f}  svd={r['svd_px_rmse']:.2e} "
                f"wsvd={r['wsvd_px_rmse']:.2e}"
            )
        df_spread = pd.DataFrame(rows_spread)
        _plot(
            lat,
            LATTICE_LABEL[lat],
            df_spread,
            "spread",
            "per-BPM noise spread (geometric)",
            xscale="linear",
            tag="spread",
            title=f"weighted vs plain SVD vs heteroscedasticity (median={SPREAD_LEVEL:.0e} m)",
        )

        for kind, d in (("level", df_level), ("spread", df_spread)):
            dd = d.copy()
            dd.insert(0, "lattice", lat)
            dd.insert(1, "sweep", kind)
            all_rows.append(dd)

    pd.concat(all_rows, ignore_index=True).to_csv("study/results/08_weighted_svd.csv", index=False)
    print("wrote study/results/08_weighted_svd.csv")


if __name__ == "__main__":
    main()
