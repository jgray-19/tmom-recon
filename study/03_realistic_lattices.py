"""Study 3 — realistic lattices (LHC and PSB).

Free-betatron reconstruction on the real LHC (beam 1) and PSB (ring 3) lattices.
For each ring we run three cases:

* **on-momentum** (``delta=0``) — the cleanest realistic baseline;
* **off-momentum, pt known** — ``delta`` in the realistic ``1e-4..1e-3`` band,
  ``pt`` handed to the reconstruction (isolates the dispersion *model* error);
* **off-momentum, pt estimated** — the operational case where ``pt`` is inferred
  from the dispersive orbit (adds the estimation error on top).

Reports per-plane RMSE / R^2 and the per-BPM RMSE around the ring (absolute and
relative to the signal size).

Outputs: study/plots/03_realistic_{lhc,psb}_perbpm_{abs,rel}.{pdf,png},
         study/results/03_realistic_summary.csv
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from study.metrics import per_bpm_rmse, reconstruct, score
from study.plotting import PX_COLOR, PY_COLOR, save_fig, set_paper_style
from study.xsuite_tracking import bpm_twiss, load_ring, track_free

logging.basicConfig(level=logging.ERROR)

# Per-ring tracking parameters: action scaled to the typical beta function so the
# betatron amplitude is ~mm-scale, delta inside the realistic 1e-4..1e-3 band.
RINGS = {
    "lhc": {"action": 5e-9, "nturn": 128, "dp": 1e-4, "label": "LHC B1"},
    "psb": {"action": 5e-6, "nturn": 256, "dp": 1e-3, "label": "PSB ring 3"},
}


def main() -> None:
    set_paper_style()
    summary = []
    for ring, cfg in RINGS.items():
        iface = load_ring(ring)
        tws = bpm_twiss(iface)
        pt = iface.dp2pt(cfg["dp"])
        print(
            f"[{ring}] {len(tws)} BPMs, q1={tws.headers.get('q1'):.3f} "
            f"q2={tws.headers.get('q2'):.3f}, delta={cfg['dp']:.1e} -> pt={pt:.2e}"
        )

        cases = [
            ("on-momentum", 0.0, False, None),
            ("off-mom (pt known)", cfg["dp"], True, pt),
            ("off-mom (pt est.)", cfg["dp"], True, None),
        ]
        for tag, dp, disp, override in cases:
            trk = track_free(iface, action=cfg["action"], nturn=cfg["nturn"], dp=dp)
            merged = reconstruct(trk, tws, use_dispersion=disp, pt_override=override)
            s = score(merged)
            print(
                f"   {tag:20s}: px RMSE={s['px_rmse']:.2e} (R2={s['px_r2']:.4f})  "
                f"py RMSE={s['py_rmse']:.2e} (R2={s['py_r2']:.4f})"
            )
            summary.append({"ring": ring, "case": tag, **s})

            if tag == "on-momentum":
                perbpm = per_bpm_rmse(merged, tws)
                scale = {"px": s["px_scale"], "py": s["py_scale"]}
                for relative, suffix, ylabel in (
                    (False, "abs", "per-BPM RMSE [rad]"),
                    (True, "rel", r"per-BPM relative error  RMSE / $\sigma_p$"),
                ):
                    fig, ax = plt.subplots(figsize=(8.4, 4.0))
                    yx = perbpm["px_rmse"] / (scale["px"] if relative else 1.0)
                    yy = perbpm["py_rmse"] / (scale["py"] if relative else 1.0)
                    ax.plot(perbpm["s"], yx, "-o", ms=3, color=PX_COLOR, label=r"$p_x$")
                    ax.plot(perbpm["s"], yy, "-s", ms=3, color=PY_COLOR, label=r"$p_y$")
                    ax.set_yscale("log")
                    ax.set_xlabel("longitudinal position $s$ [m]")
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{cfg['label']}: per-BPM reconstruction error (on-momentum)")
                    ax.legend()
                    save_fig(fig, f"03_realistic_{ring}_perbpm_{suffix}")
        iface.close()

    pd.DataFrame(summary).to_csv("study/results/03_realistic_summary.csv", index=False)
    print("wrote study/results/03_realistic_summary.csv")


if __name__ == "__main__":
    main()
