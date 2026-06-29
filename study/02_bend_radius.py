"""Study 2 — dependence on bending radius (dispersion), at physical energy.

How much does the bending strength itself limit the reconstruction? We sweep the
FODO dipole angle, which sets the bending radius ``rho = L_bend / angle`` and
hence the dispersion. A real magnet of fixed field bends a higher-momentum beam
on a larger radius, so the beam energy is made to follow ``rho`` (``p = 0.3 B rho``):
the small-``rho`` end is a PSB-like low-energy point, the large-``rho`` end an
LHC-like high-energy point. We reconstruct a slightly off-momentum particle
(``delta`` in the realistic ``1e-4..1e-3`` band, converted to ``pt``) with the
momentum *estimated* from the noisy orbit. Many noise seeds give a mean and a
spread. The operating-point ``1/rho`` of the LHC and PSB main dipoles are overlaid.

Outputs: study/plots/02_bend_radius_{abs,rel}.{pdf,png}, study/results/02_bend_radius.csv
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from study.fodo import BEND_LEN
from study.metrics import reconstruct, score
from study.plotting import PALETTE, abs_rel_figures, set_paper_style
from study.xsuite_tracking import (
    bpm_twiss,
    energy_from_rho,
    make_fodo,
    matched_quad_knobs,
    set_fodo_knobs,
    track_free,
)

logging.basicConfig(level=logging.ERROR)

NTURN = 256
ACTION = 3e-7  # Courant-Snyder action [m·rad]
DP = 5e-4  # momentum offset (Delta p / p), mid of the 1e-4..1e-3 band
NOISE_STD = 5e-5  # realistic BPM resolution [m]
SEEDS = range(12)
B_FIELD = 1.0  # representative dipole field [T] -> energy = f(rho)

# Main-dipole bending radii of the real machines [m].
LHC_RHO = 2803.95
PSB_RHO = 8.239


def main() -> None:
    set_paper_style()

    # Match the quads once (geometric, energy-independent) and reuse the knobs so
    # every rho/energy point is the same lattice rebuilt cheaply.
    iface0 = make_fodo()
    knobs = matched_quad_knobs(iface0)
    iface0.close()

    angles = np.linspace(0.02, 0.30, 10)
    rows = []
    for ang in angles:
        rho = BEND_LEN / ang
        energy = energy_from_rho(rho, b_field=B_FIELD)
        iface = make_fodo(kinetic_energy=energy, quad_knobs=knobs)
        set_fodo_knobs(iface, bangle=ang)
        tws = bpm_twiss(iface)
        disp_rms = float(np.sqrt(np.mean(tws["dx"] ** 2)))
        trk = track_free(iface, action=ACTION, nturn=NTURN, dp=DP)
        px, py, pxs, pys = [], [], [], []
        for seed in SEEDS:
            merged = reconstruct(
                trk,
                tws,
                use_dispersion=True,
                noise_std=NOISE_STD,
                rng=np.random.default_rng(seed),
            )
            s = score(merged)
            px.append(s["px_rmse"])
            py.append(s["py_rmse"])
            pxs.append(s["px_scale"])
            pys.append(s["py_scale"])
        iface.close()
        rows.append(
            {
                "angle": ang,
                "rho": rho,
                "inv_rho": 1.0 / rho,
                "energy_gev": energy,
                "disp_rms": disp_rms,
                "px_rmse": np.mean(px),
                "px_std": np.std(px),
                "px_scale": np.mean(pxs),
                "py_rmse": np.mean(py),
                "py_std": np.std(py),
                "py_scale": np.mean(pys),
            }
        )
        print(
            f"[rho] rho={rho:6.1f}m E={energy:6.2f}GeV 1/rho={1 / rho:.4f} "
            f"Dx_rms={disp_rms:.2f}m px={np.mean(px):.2e} py={np.mean(py):.2e}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("study/results/02_bend_radius.csv", index=False)

    vlines = [
        (1.0 / LHC_RHO, f"LHC\n1/$\\rho$={1 / LHC_RHO:.1e}", PALETTE["green"]),
        (1.0 / PSB_RHO, f"PSB\n1/$\\rho$={1 / PSB_RHO:.1e}", PALETTE["purple"]),
    ]
    abs_rel_figures(
        "02_bend_radius",
        df["inv_rho"],
        df,
        xlabel=r"inverse bending radius $1/\rho$ [m$^{-1}$]",
        title=r"Off-momentum reconstruction vs bend radius ($\delta=5\times10^{-4}$, estimated)",
        xscale="log",
        vlines=vlines,
    )
    print("wrote study/plots/02_bend_radius_{abs,rel}.pdf")


if __name__ == "__main__":
    main()
