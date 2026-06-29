"""Study 7 — phase advance between neighbouring BPMs.

The reconstruction infers momenta from *pairs* of neighbouring BPMs, inverting a
2x2 transfer governed by the betatron phase advance ``dmu`` between them. When
``dmu`` approaches 0 or pi the pair is degenerate and the inversion is
ill-conditioned, so BPM noise is amplified. We expose this two ways, with a
fixed BPM noise injected so the conditioning shows up as amplified error:

* **BPM decimation** (FODO, LHC, PSB) — keep every k-th BPM, which widens the
  phase advance between the BPMs actually used;
* **quad-strength scan** (FODO) — retune the quads to change the per-cell phase
  advance directly.

For each configuration the median neighbour phase advance is measured and the
noisy reconstruction is averaged over many seeds (mean + spread).

Outputs: study/plots/07_phase_{lattice}_decimation_{abs,rel}.{pdf,png},
         study/plots/07_phase_fodo_quadscan_{abs,rel}.{pdf,png},
         study/results/07_phase_advance.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from study.metrics import reconstruct, score
from study.plotting import abs_rel_figures, set_paper_style
from study.xsuite_tracking import (
    LATTICE_LABEL,
    LATTICE_TRACK,
    LATTICES,
    bpm_twiss,
    build_lattice,
    make_fodo,
    track_free,
)

logging.basicConfig(level=logging.ERROR)

NOISE_STD = 5e-5  # fixed BPM noise so ill-conditioning shows as amplified error
SEEDS = range(12)
MIN_BPMS = 6  # keep at least this many BPMs in a decimated set


def _median_dmu(tws_sub: pd.DataFrame) -> float:
    """Median horizontal phase advance [rad] between consecutive (s-sorted) BPMs."""
    mu = tws_sub.sort_values("s")["mu1"].to_numpy()
    return float(np.median(np.abs(np.diff(mu))) * 2 * np.pi)


def _noisy_score(trk, tws):
    px, py, pxs, pys = [], [], [], []
    for seed in SEEDS:
        s = score(
            reconstruct(
                trk,
                tws,
                use_dispersion=False,
                noise_std=NOISE_STD,
                rng=np.random.default_rng(seed),
            )
        )
        px.append(s["px_rmse"])
        py.append(s["py_rmse"])
        pxs.append(s["px_scale"])
        pys.append(s["py_scale"])
    return {
        "px_rmse": np.mean(px),
        "px_std": np.std(px),
        "px_scale": np.mean(pxs),
        "py_rmse": np.mean(py),
        "py_std": np.std(py),
        "py_scale": np.mean(pys),
    }


def decimation_scan(trk, tws):
    """Keep every k-th BPM; report RMSE vs median neighbour phase advance."""
    names = tws.sort_values("s").index.tolist()
    n = len(names)
    strides = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24) if n // k >= MIN_BPMS]
    rows = []
    for k in strides:
        sub = names[::k]
        tws_sub = tws.loc[sub]
        dmu = _median_dmu(tws_sub)
        # Beyond ~2*pi consecutive BPMs are more than a full betatron oscillation
        # apart: the neighbour model is aliased and the reconstruction is garbage.
        if dmu > 2 * np.pi + 0.2:
            continue
        sc = _noisy_score(trk[trk["name"].isin(sub)], tws_sub)
        if not np.isfinite(sc["px_rmse"]) or not np.isfinite(sc["py_rmse"]):
            continue
        rows.append({"stride": k, "n_bpm": len(sub), "dmu": dmu, **sc})
        print(
            f"   stride={k:2d}  n_bpm={len(sub):3d}  dmu={dmu:.3f}rad  px={rows[-1]['px_rmse']:.2e}"
        )
    return pd.DataFrame(rows)


def quad_scan():
    """FODO only: retune quads to vary the per-cell phase advance."""
    rows = []
    # Stay clear of the integer (3.0) and half-integer (3.5) resonances.
    for qx in np.linspace(3.06, 3.44, 9):
        qy = qx - 0.07
        iface = make_fodo(qx=qx, qy=qy)
        tws = bpm_twiss(iface)
        cfg = LATTICE_TRACK["fodo"]
        trk = track_free(iface, action=cfg["action"], nturn=cfg["nturn"], dp=0.0)
        iface.close()
        rows.append({"qx": qx, "dmu": _median_dmu(tws), **_noisy_score(trk, tws)})
        print(f"   qx={qx:.3f}  dmu={rows[-1]['dmu']:.3f}rad  px={rows[-1]['px_rmse']:.2e}")
    return pd.DataFrame(rows)


def main() -> None:
    set_paper_style()
    all_rows = []

    for lat in LATTICES:
        print(f"[{lat}] decimation scan")
        cfg = LATTICE_TRACK[lat]
        iface = build_lattice(lat)
        tws = bpm_twiss(iface)
        trk = track_free(iface, action=cfg["action"], nturn=cfg["nturn"], dp=0.0)
        iface.close()
        df = decimation_scan(trk, tws)
        abs_rel_figures(
            f"07_phase_{lat}_decimation",
            df["dmu"],
            df,
            xlabel=r"median neighbour phase advance $\Delta\mu_x$ [rad]",
            title=f"{LATTICE_LABEL[lat]}: reconstruction error vs BPM phase advance (decimation)",
        )
        d = df.copy()
        d.insert(0, "lattice", lat)
        d.insert(1, "method", "decimation")
        all_rows.append(d)

    print("[fodo] quad-strength scan")
    dfq = quad_scan()
    abs_rel_figures(
        "07_phase_fodo_quadscan",
        dfq["dmu"],
        dfq,
        xlabel=r"per-cell phase advance $\Delta\mu_x$ [rad]",
        title="FODO: reconstruction error vs phase advance (quad-strength scan)",
    )
    dq = dfq.copy()
    dq.insert(0, "lattice", "fodo")
    dq.insert(1, "method", "quadscan")
    all_rows.append(dq)

    pd.concat(all_rows, ignore_index=True).to_csv("study/results/07_phase_advance.csv", index=False)
    print("wrote study/results/07_phase_advance.csv")


if __name__ == "__main__":
    main()
