from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xpart as xp
import xtrack as xt
from pymadng import MAD

OUT_DIR = Path("/tmp/mcby_minimal_compare")
SEQ_FILE = OUT_DIR / "mcby_minimal.seq"
MAD_CACHE = OUT_DIR / "mcby_minimal.mad"
PLOT_FILE = OUT_DIR / "mcby_minimal_compare.png"
CSV_FILE = OUT_DIR / "mcby_minimal_compare.csv"
COORDS = ["x", "px", "y", "py"]
KINETIC_ENERGY_GEV = 6800.0
TOTAL_ENERGY_GEV = KINETIC_ENERGY_GEV + xp.PROTON_MASS_EV / 1e9
INITIAL = {
    "x": 4.0e-6,
    "px": 9.0e-9,
    "y": 2.0e-6,
    "py": -1.0e-8,
    "zeta": 0.0,
    "delta": 0.0,
}


def write_sequence() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_FILE.write_text(
        f"""
l.mcbyh = 8.9900000000000000e-01;
l.mcbyv = 8.9900000000000000e-01;

hcorrector: hkicker;
mcbyh: hcorrector, l:=l.mcbyh;

vcorrector: vkicker;
mcbyv: vcorrector, l:=l.mcbyv;

lhcb1: sequence, l=6.0, refer=entry;
    start: marker, at=0.0;
    mcbyh.b5l2.b1: mcbyh, at=1.0, hkick=1.3e-6;
    mid: marker, at=2.5;
    mcbyv.5l2.b1: mcbyv, at=3.0, vkick=-1.7e-6;
    finish: marker, at=6.0;
endsequence;

beam, sequence=lhcb1, particle=proton, energy={TOTAL_ENERGY_GEV:.15e};
use, sequence=lhcb1;
""".strip()
        + "\n"
    )


def xtrack_element_track() -> pd.DataFrame:
    env = xt.load(file=SEQ_FILE, format="madx")
    line = env["lhcb1"]
    line.particle_ref = xt.Particles(
        mass0=xp.PROTON_MASS_EV,
        energy0=TOTAL_ENERGY_GEV * 1e9,
    )
    line.configure_drift_model(model="exact")

    particle = line.build_particles(**INITIAL)
    line.track(particle, turn_by_turn_monitor="ONE_TURN_EBE")
    mon = line.record_last_track
    at_element = np.asarray(mon.at_element[0, :], dtype=int)
    names = [
        line.element_names[i] if 0 <= i < len(line.element_names) else f"__{i}__"
        for i in at_element
    ]
    raw = pd.DataFrame(
        {
            "name": [str(name).upper() for name in names],
            "at_element": at_element,
            "s": np.asarray(mon.s[0, :], dtype=float),
            "x": np.asarray(mon.x[0, :], dtype=float),
            "px": np.asarray(mon.px[0, :], dtype=float),
            "y": np.asarray(mon.y[0, :], dtype=float),
            "py": np.asarray(mon.py[0, :], dtype=float),
        }
    )

    # ONE_TURN_EBE records entrance states. Shift the coordinates onto element exits.
    out = raw.iloc[:-1].copy(deep=True).reset_index(drop=True)
    next_rows = raw.iloc[1:].reset_index(drop=True)
    for col in ["s", *COORDS]:
        out[col] = next_rows[col].to_numpy()
    return out


def madng_element_track() -> pd.DataFrame:
    mad = MAD(stdout=OUT_DIR / "madng.log")
    mad.send("MADX.option.rbarc = false")
    mad.send(f'MADX:load("{SEQ_FILE}", "{MAD_CACHE}", {{rbarc=false}})')
    mad.send("loaded_sequence = MADX.lhcb1")
    mad.send(
        f'loaded_sequence.beam = beam {{ particle = "proton", energy={TOTAL_ENERGY_GEV:.15e} }}'
    )
    mad.send(
        """
x0_particles = py:recv()
tbl, flw = track {
    sequence=loaded_sequence,
    X0=x0_particles,
    save=true,
    nturn=1,
    observe=0,
    method=6,
}
py:send(flw.tpar == flw.npar)
"""
    ).send(
        [
            {
                "x": INITIAL["x"],
                "px": INITIAL["px"],
                "y": INITIAL["y"],
                "py": INITIAL["py"],
                "t": 0.0,
                "pt": 0.0,
            }
        ]
    )
    ok = mad.recv()
    if not ok:
        raise RuntimeError("MAD-NG lost the particle")
    df = mad.tbl.to_df(force_pandas=True).reset_index(drop=True)
    df["name"] = df["name"].astype(str).str.upper()
    return df[["name", "s", *COORDS]]


def compare_tracks(xdf: pd.DataFrame, mdf: pd.DataFrame) -> pd.DataFrame:
    x_unique = xdf.drop_duplicates("name", keep="first").set_index("name")
    m_unique = mdf.drop_duplicates("name", keep="first").set_index("name")
    common = [name for name in m_unique.index if name in x_unique.index]
    rows = []
    for name in common:
        xrow = x_unique.loc[name]
        mrow = m_unique.loc[name]
        diff = mrow[COORDS].to_numpy(dtype=float) - xrow[COORDS].to_numpy(dtype=float)
        rows.append(
            {
                "name": name,
                "s_mad": float(mrow["s"]),
                "s_x": float(xrow["s"]),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "dx": diff[0],
                "dpx": diff[1],
                "dy": diff[2],
                "dpy": diff[3],
                "x_mad": float(mrow["x"]),
                "px_mad": float(mrow["px"]),
                "y_mad": float(mrow["y"]),
                "py_mad": float(mrow["py"]),
                "x_xsuite": float(xrow["x"]),
                "px_xsuite": float(xrow["px"]),
                "y_xsuite": float(xrow["y"]),
                "py_xsuite": float(xrow["py"]),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(cdf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    x = np.arange(len(cdf))
    for coord in COORDS:
        axes[0].plot(x, cdf[f"d{coord}"], marker="o", lw=1.1, label=f"d{coord}")
    axes[0].axhline(0.0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_ylabel("MAD-NG - xtrack")
    axes[0].set_title("Minimal sequence: MCBYH/MCBYV element-exit differences")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(visible=True, alpha=0.25)

    axes[1].semilogy(x, np.maximum(cdf["max_abs_diff"], 1e-30), marker="o", lw=1.1)
    axes[1].set_ylabel("max |diff|")
    axes[1].set_xlabel("element")
    axes[1].grid(visible=True, which="both", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cdf["name"], rotation=25, ha="right")

    fig.savefig(PLOT_FILE, dpi=180)
    plt.close(fig)


def main() -> None:
    write_sequence()
    xdf = xtrack_element_track()
    mdf = madng_element_track()
    cdf = compare_tracks(xdf, mdf)
    cdf.to_csv(CSV_FILE, index=False)
    plot_comparison(cdf)

    print(f"sequence: {SEQ_FILE}")
    print(f"plot:     {PLOT_FILE}")
    print(f"csv:      {CSV_FILE}")
    print("\n[xtrack exit states]")
    print(xdf.to_string(index=False))
    print("\n[MAD-NG states]")
    print(mdf.to_string(index=False))
    print("\n[comparison]")
    print(cdf.to_string(index=False))


if __name__ == "__main__":
    main()
