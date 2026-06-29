"""Study 6 — imperfect optics knowledge (distorted model twiss).

Even a perfectly tracked lattice is reconstructed badly if the *model* optics
handed to the reconstruction are wrong. Each lattice (FODO, LHC, PSB) is tracked
once; the model twiss is then randomly distorted before reconstruction and we
sweep the RMS of two distortion families, averaging over many seeds for a mean
and a spread:

* beta-beating — random relative ``beta`` errors (``beta_rel``);
* phase errors — random per-BPM betatron-phase walk (``phase_abs``).

This isolates the "we never know the optics perfectly" limitation from the
non-linearity (Study 1) and magnet-error (Study 5) contributions.

Outputs: study/plots/06_optics_{lattice}_{beta,phase}_{abs,rel}.{pdf,png},
         study/results/06_model_optics_distortion.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from study.metrics import distort_optics, reconstruct, score
from study.plotting import abs_rel_figures, set_paper_style
from study.xsuite_tracking import (
    LATTICE_LABEL,
    LATTICE_TRACK,
    LATTICES,
    bpm_twiss,
    build_lattice,
    track_free,
)

logging.basicConfig(level=logging.ERROR)
SEEDS = range(20)

FAMILIES = {
    "beta": {
        "kwarg": "beta_rel",
        "levels": np.linspace(0.0, 0.10, 8),
        "label": "beta-beating RMS",
        "title": "beta-beating",
    },
    "phase": {
        "kwarg": "phase_abs",
        "levels": np.linspace(0.0, 0.02, 8),
        "label": "per-BPM phase-error RMS [rad]",
        "title": "phase error",
    },
}


def main() -> None:
    set_paper_style()
    all_rows = []
    for lat in LATTICES:
        cfg = LATTICE_TRACK[lat]
        iface = build_lattice(lat)
        tws = bpm_twiss(iface)
        trk = track_free(iface, action=cfg["action"], nturn=cfg["nturn"], dp=0.0)
        iface.close()

        for tag, fam in FAMILIES.items():
            rows = []
            for level in fam["levels"]:
                px, py, pxs, pys = [], [], [], []
                for seed in SEEDS:
                    model = distort_optics(
                        tws, np.random.default_rng(seed), **{fam["kwarg"]: float(level)}
                    )
                    s = score(reconstruct(trk, model, use_dispersion=False))
                    px.append(s["px_rmse"])
                    py.append(s["py_rmse"])
                    pxs.append(s["px_scale"])
                    pys.append(s["py_scale"])
                rows.append(
                    {
                        "level": level,
                        "px_rmse": np.mean(px),
                        "px_std": np.std(px),
                        "px_scale": np.mean(pxs),
                        "py_rmse": np.mean(py),
                        "py_std": np.std(py),
                        "py_scale": np.mean(pys),
                    }
                )
            df = pd.DataFrame(rows)
            abs_rel_figures(
                f"06_optics_{lat}_{tag}",
                df["level"],
                df,
                xlabel=fam["label"],
                title=f"{LATTICE_LABEL[lat]}: reconstruction error vs model {fam['title']}",
            )
            d = df.copy()
            d.insert(0, "lattice", lat)
            d.insert(1, "family", tag)
            all_rows.append(d)
            print(
                f"[{lat}/{tag}] level {fam['levels'][0]:.3f}->{fam['levels'][-1]:.3f}: "
                f"px {df['px_rmse'].iloc[0]:.2e}->{df['px_rmse'].iloc[-1]:.2e}"
            )

    pd.concat(all_rows, ignore_index=True).to_csv(
        "study/results/06_model_optics_distortion.csv", index=False
    )
    print("wrote study/results/06_model_optics_distortion.csv")


if __name__ == "__main__":
    main()
