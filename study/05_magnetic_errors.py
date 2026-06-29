"""Study 5 — unknown magnetic errors (quadrupole k1, dipole k0).

The reconstruction trusts the *nominal* model optics; real magnets deviate from
it. For each lattice (FODO, LHC, PSB) we perturb the *tracked* machine with
Gaussian relative errors of a given RMS (seeded, ``apply_magnet_perturbations``)
while the reconstruction keeps the nominal twiss captured *before* perturbing,
and sweep the error RMS:

* quadrupole ``k1`` errors — distort the beta/phase the model assumes;
* dipole ``k0`` errors — kick the closed orbit / dispersion the model ignores.

Each (error-RMS, family, lattice) point is averaged over many seeds for a mean
and a spread. The FODO has its bends switched on so the dipole family is
meaningful; a fresh machine is built for every seed so errors never accumulate.

Outputs: study/plots/05_magnetic_{lattice}_{quad,dipole}_{abs,rel}.{pdf,png},
         study/results/05_magnetic_errors.csv
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
    matched_quad_knobs,
    track_free,
)

logging.basicConfig(level=logging.ERROR)

BANGLE = 0.05  # FODO bends on -> dipole-k0 errors meaningful
REL_STDS = np.logspace(-5, -3, 5)  # relative error RMS sweep
SEEDS = range(6)
# A fresh machine is rebuilt for every seed (errors must not accumulate) and the
# full ring is re-tracked each time, which is the cost driver. Each BPM/turn is an
# independent RMSE sample (LHC has 563 BPMs), so a handful of turns already gives a
# well-converged per-seed RMSE; this keeps the LHC sweep to a few minutes.
NTURN = {"fodo": 64, "lhc": 8, "psb": 24}

FAMILIES = {
    "quad": {"family": "q", "label": "quadrupole $k_1$"},
    "dipole": {"family": "d", "label": "dipole $k_0$"},
}


def main() -> None:
    set_paper_style()

    # Matched FODO quads, reused for every cheap FODO rebuild.
    iface0 = make_fodo()
    fodo_knobs = matched_quad_knobs(iface0)
    iface0.close()

    def build(name):
        return build_lattice(name, quad_knobs=fodo_knobs, bangle=BANGLE)

    all_rows = []
    for lat in LATTICES:
        nturn = NTURN[lat]
        nominal_tws = bpm_twiss(build(lat))  # model never sees the errors
        for tag, fam in FAMILIES.items():
            rows = []
            for rel in REL_STDS:
                px, py, pxs, pys = [], [], [], []
                for seed in SEEDS:
                    iface = build(lat)
                    iface.apply_magnet_perturbations(
                        rel_error=float(rel), seed=seed, magnet_type=[fam["family"]]
                    )
                    trk = track_free(
                        iface, action=LATTICE_TRACK[lat]["action"], nturn=nturn, dp=0.0
                    )
                    iface.close()
                    s = score(reconstruct(trk, nominal_tws, use_dispersion=False))
                    px.append(s["px_rmse"])
                    py.append(s["py_rmse"])
                    pxs.append(s["px_scale"])
                    pys.append(s["py_scale"])
                rows.append(
                    {
                        "rel_error": rel,
                        "px_rmse": np.mean(px),
                        "px_std": np.std(px),
                        "px_scale": np.mean(pxs),
                        "py_rmse": np.mean(py),
                        "py_std": np.std(py),
                        "py_scale": np.mean(pys),
                    }
                )
                print(
                    f"[{lat}/{tag}] rel={rel:.1e}  px={np.mean(px):.2e}+-{np.std(px):.1e}  "
                    f"py={np.mean(py):.2e}+-{np.std(py):.1e}"
                )

            df = pd.DataFrame(rows)
            abs_rel_figures(
                f"05_magnetic_{lat}_{tag}",
                df["rel_error"],
                df,
                xlabel=f"{fam['label']} relative error RMS",
                title=f"{LATTICE_LABEL[lat]}: reconstruction error vs {fam['label']} error",
                xscale="log",
            )
            d = df.copy()
            d.insert(0, "lattice", lat)
            d.insert(1, "family", tag)
            all_rows.append(d)

    pd.concat(all_rows, ignore_index=True).to_csv(
        "study/results/05_magnetic_errors.csv", index=False
    )
    print("wrote study/results/05_magnetic_errors.csv")


if __name__ == "__main__":
    main()
