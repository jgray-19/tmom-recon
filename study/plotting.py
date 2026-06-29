"""Publication-quality plotting helpers (colourblind-safe).

A single :func:`set_paper_style` sets consistent rcParams and exposes the
Okabe-Ito / Wong colourblind-safe palette so every study figure matches and is
legible to readers with colour-vision deficiency. Figures are saved as both PDF
(vector, for the paper) and PNG (preview) into ``study/plots``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Okabe & Ito colourblind-safe qualitative palette.
# See https://davidmathlogic.com/colorblind/#%23D81B60-%231E88E5-%23FFC107-%23004D40
PALETTE = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
CYCLE: list[str] = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
# Stable colours for the two transverse planes across all figures.
PX_COLOR = PALETTE["blue"]
PY_COLOR = PALETTE["vermillion"]


def set_paper_style() -> None:
    """Apply consistent, publication-ready matplotlib settings."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 12,
            "font.family": "serif",
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "axes.prop_cycle": mpl.cycler(color=CYCLE),
        }
    )


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` as PDF + PNG under ``study/plots`` and return the PDF path."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PLOTS_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(PLOTS_DIR / f"{name}.png")
    plt.close(fig)
    return pdf


# Default per-plane drawing spec: (column prefix, legend label, colour, marker).
PLANES = [("px", r"$p_x$", PX_COLOR, "o"), ("py", r"$p_y$", PY_COLOR, "s")]


def plot_planes(ax, x, df, planes=PLANES, *, relative=False, logy=True, baseline=None):
    """Draw per-plane RMSE (absolute or relative) with error bars onto ``ax``.

    ``df`` must hold ``<prefix>_rmse`` (and optionally ``<prefix>_std`` for the
    seed spread and ``<prefix>_scale`` for the signal size). When ``relative`` the
    curve is ``rmse / scale`` (a dimensionless fractional error). ``baseline`` (a
    dict of ``<prefix>_rmse`` / ``<prefix>_scale``) draws the numerical floor.
    """
    import numpy as np

    for prefix, label, color, marker in planes:
        y = np.asarray(df[f"{prefix}_rmse"], dtype=float)
        yerr = np.asarray(df[f"{prefix}_std"], dtype=float) if f"{prefix}_std" in df else None
        if relative:
            scale = np.asarray(df[f"{prefix}_scale"], dtype=float)
            y = y / scale
            yerr = yerr / scale if yerr is not None else None
        if yerr is not None and np.any(yerr > 0):
            ax.errorbar(x, y, yerr=yerr, fmt=f"-{marker}", color=color, capsize=3, label=label)
        else:
            ax.plot(x, y, f"-{marker}", color=color, label=label)
        if baseline is not None:
            b = baseline[f"{prefix}_rmse"]
            if relative:
                b = b / baseline[f"{prefix}_scale"]
            ax.axhline(b, ls=":", lw=1.3, color=color, alpha=0.6)
    if logy:
        ax.set_yscale("log")


def _draw_vlines(ax, vlines):
    for xval, label, color in vlines:
        ax.axvline(xval, ls="--", color=color, lw=1.5)
        ax.text(xval, ax.get_ylim()[1], f" {label}", color=color, va="top", ha="left", fontsize=9)


def abs_rel_figures(
    name, x, df, *, xlabel, title, xscale="linear", vlines=None, baseline=None, planes=PLANES
):
    """Save a matched pair of single-panel figures: ``<name>_abs`` and ``<name>_rel``.

    Both share the same x-axis/series; one shows absolute RMSE, the other the
    relative (fractional) error ``RMSE / sigma_p``. Returns the two PDF paths.
    """
    import matplotlib.pyplot as plt

    paths = []
    for relative, suffix, ylabel, ttl in (
        (False, "abs", "reconstruction RMSE [rad]", title),
        (True, "rel", r"relative error  RMSE / $\sigma_{p}$", f"{title} (relative)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        plot_planes(ax, x, df, planes, relative=relative, baseline=baseline)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(ttl)
        ax.legend()
        if vlines:
            _draw_vlines(ax, vlines)
        paths.append(save_fig(fig, f"{name}_{suffix}"))
    return paths
