import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_dispersion(
    frame: pd.DataFrame, *, value_columns: tuple[str, ...] = ("model", "measured")
) -> pd.DataFrame:
    """Dispersion as the closed orbit's slope against ``pt``, per BPM.

    Args:
        frame: One row per (``plane``, ``bpm``, ``pt``), plus one column per
            name in ``value_columns`` -- the closed-orbit position at that
            point, in whatever units/momentum-units the caller is working in.
        value_columns: Which columns to fit; each becomes its own output
            column of fitted slopes.

    Returns:
        One row per (``plane``, ``bpm``) with a fitted slope column per name
        in ``value_columns``; ``NaN`` where fewer than two points survive.
    """
    rows = []
    frame = frame.dropna(subset=list(value_columns))
    for (plane, bpm), group in frame.groupby(["plane", "bpm"], sort=False):
        row = {"plane": plane, "bpm": bpm}
        if len(group) < 2:
            row.update(dict.fromkeys(value_columns, np.nan))
            rows.append(row)
            continue
        design = np.polyfit(group["pt"].to_numpy(), group[list(value_columns)], 1)
        row.update(dict(zip(value_columns, design[0], strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Straight-line fit with known one-sigma errors on ``y``; return (slope, slope_error)."""
    design = np.column_stack([x, np.ones_like(x)])
    weight = 1.0 / sigma**2
    scaled = design * weight[:, None]
    covariance = np.linalg.inv(design.T @ scaled)
    slope, _intercept = covariance @ (scaled.T @ y)
    return float(slope), float(np.sqrt(covariance[0, 0]))


def _slope_with_pt_error(
    pt: np.ndarray, y: np.ndarray, y_sigma: np.ndarray, pt_sigma: np.ndarray, iterations: int = 5
) -> tuple[float, float]:
    """Slope of ``y`` against ``pt`` with errors on both axes.

    Iterates the effective-variance approximation -- ``hypot(y_sigma, slope *
    pt_sigma)`` -- the same both-axis convention already used for the
    chromaticity fits (``chroma_investigation/analyse.py:uncertain_line_fit``
    in the caller repository).
    """
    slope = float(np.polyfit(pt, y, 1)[0])
    for _ in range(iterations):
        effective_sigma = np.hypot(y_sigma, slope * pt_sigma)
        slope, slope_error = _weighted_line_fit(pt, y, effective_sigma)
    return slope, slope_error


def measure_dispersion(
    orbits_by_pt: dict[float, list[pd.DataFrame]],
    pt_sigma: dict[float, float],
    *,
    tws: pd.DataFrame,
) -> pd.DataFrame:
    """Dispersion from repeated closed-orbit acquisitions, with propagated uncertainty.

    Unlike :func:`fit_dispersion`, which fits a single noiseless value per
    ``pt`` (e.g. a model prediction), this fits repeated *measurements*: the
    scatter across repeat acquisitions at each ``pt`` gives the orbit's own
    uncertainty, and ``pt_sigma`` -- typically the chromaticity scan's
    ``Dp/p`` uncertainty -- gives the momentum's. Both enter the fitted
    slope's error through the standard both-axis (effective-variance) method.

    Args:
        orbits_by_pt: Repeat closed-orbit acquisitions at each measured ``pt``
            setting. Each acquisition is a DataFrame with one row per BPM and
            columns ``name``, ``x``, ``y`` -- the same shape
            :func:`estimate_closed_orbit` produces. Repeats of one ``pt`` are
            averaged and their spread taken as that point's uncertainty; a
            ``pt`` with only one acquisition contributes no scatter estimate.
        pt_sigma: One-sigma uncertainty on each ``pt`` setting. Must have an
            entry for every key of ``orbits_by_pt``.
        tws: Model twiss at ``pt = 0``, indexed by BPM name, computed with
            MAD-NG ``chrom`` so it carries the second-order dispersion columns
            ``ddx``/``ddy``; other columns are ignored. The orbit is
            ``x(pt) = x0 + pt*D + pt**2*D2``; ``pt**2*D2`` is subtracted before
            the straight-line fit, so the slope is ``D`` at ``pt = 0`` for any
            set of settings, not the secant of an asymmetric scan. ``D2`` is
            fixed rather than fitted: three settings would leave a free
            quadratic no degrees of freedom.

    Returns:
        One row per BPM name, with ``dx``, ``dx_err``, ``dy``, ``dy_err``:
        the fitted dispersion and its 1-sigma uncertainty. ``NaN`` where fewer
        than two ``pt`` settings survive for that BPM.

    Raises:
        KeyError: If *tws* lacks ``ddx``/``ddy`` or a measured BPM.
    """
    missing = [pt for pt in orbits_by_pt if pt not in pt_sigma]
    if missing:
        raise ValueError(f"pt_sigma is missing setting(s) {missing}")

    points = []
    for pt, acquisitions in orbits_by_pt.items():
        if not acquisitions:
            raise ValueError(f"pt={pt} has no acquisitions")
        stacked = pd.concat(
            [df.set_index("name")[["x", "y"]] for df in acquisitions],
            axis=1,
            keys=range(len(acquisitions)),
        )
        n = len(acquisitions)
        for coord in ("x", "y"):
            values = stacked.xs(coord, axis=1, level=1)
            mean = values.mean(axis=1)
            sem = (
                values.std(axis=1, ddof=1) / np.sqrt(n)
                if n > 1
                else pd.Series(np.nan, index=mean.index)
            )
            points.append(
                pd.DataFrame(
                    {
                        "name": mean.index,
                        "pt": pt,
                        "coord": coord,
                        "mean": mean.to_numpy(),
                        "sem": sem.to_numpy(),
                    }
                )
            )
    long = pd.concat(points, ignore_index=True)

    missing_cols = [c for c in ("ddx", "ddy") if c not in tws.columns]
    if missing_cols:
        raise KeyError(f"tws is missing column(s) {missing_cols}")
    d2 = tws[["ddx", "ddy"]].copy()
    d2.index = d2.index.astype(str).str.upper()
    names = long["name"].astype(str).str.upper()
    absent = sorted(set(names) - set(d2.index))
    if absent:
        raise KeyError(f"tws is missing BPM(s) {absent}")
    for coord, column in (("x", "ddx"), ("y", "ddy")):
        sel = long["coord"] == coord
        long.loc[sel, "mean"] -= long.loc[sel, "pt"] ** 2 * d2.loc[names[sel], column].to_numpy(
            dtype=float
        )

    rows = []
    for name, group in long.groupby("name", sort=False):
        row = {"name": name}
        for coord, (value_col, error_col) in (("x", ("dx", "dx_err")), ("y", ("dy", "dy_err"))):
            subset = group[group["coord"] == coord].dropna(subset=["mean"])
            if len(subset["pt"].unique()) < 2:
                row[value_col] = np.nan
                row[error_col] = np.nan
                continue
            pt = subset["pt"].to_numpy()
            y = subset["mean"].to_numpy()
            y_sigma = subset["sem"].to_numpy()
            known = np.isfinite(y_sigma) & (y_sigma > 0)
            if not known.any():
                logger.warning(
                    "No repeat acquisitions at any pt for BPM %s (%s); fitting unweighted.",
                    name,
                    coord,
                )
                y_sigma = np.ones_like(y)
            elif not known.all():
                y_sigma = np.where(known, y_sigma, y_sigma[known].mean())
            pt_sig = np.array([pt_sigma[p] for p in subset["pt"]])
            slope, slope_error = _slope_with_pt_error(pt, y, y_sigma, pt_sig)
            row[value_col] = slope
            row[error_col] = slope_error
        rows.append(row)
    return pd.DataFrame(rows).set_index("name")


def estimate_closed_orbit(
    data: pd.DataFrame, tws: pd.DataFrame, pt_est: float = 0.0
) -> pd.DataFrame:
    """Estimate closed orbit from tracking data.

    Args:
        data: Tracking data with BPM readings. Must contain columns: ["name", "x", "y"].
        tws: Twiss parameters DataFrame. Must have columns ["dx", "dy"] and be indexed by BPM name.
        pt_est: Estimated MAD-NG pt.

    Returns:
        DataFrame indexed like tws.index with columns: x, y, var_x, var_y.
    """
    if "name" not in data.columns or "x" not in data.columns or "y" not in data.columns:
        raise ValueError('`data` must contain columns ["name", "x", "y"].')

    # Map dispersion to each row (per BPM), then correct positions turn-by-turn.
    # Force float dtype: a categorical "name" column otherwise propagates a
    # Categorical through .map(), which cannot be scaled by pt_est.
    dx_per_row = data["name"].map(tws["dx"].to_dict()).astype(float)
    dy_per_row = data["name"].map(tws["dy"].to_dict()).astype(float)
    x_corr = data["x"] - pt_est * dx_per_row
    y_corr = data["y"] - pt_est * dy_per_row

    g = pd.DataFrame({"name": data["name"], "x_corr": x_corr, "y_corr": y_corr}).groupby(
        "name", sort=False, observed=False
    )

    co_avg = pd.DataFrame(
        {
            "x": g["x_corr"].mean(),
            "y": g["y_corr"].mean(),
            "var_x": g["x_corr"].var(),
            "var_y": g["y_corr"].var(),
        }
    )

    logger.info("Estimated closed orbit at %d BPMs.", len(co_avg))
    logger.info("Mean closed orbit x: %.3e m, y: %.3e m", co_avg["x"].mean(), co_avg["y"].mean())

    # Align to Twiss order / include missing BPMs as NaN rows
    return co_avg.reindex(tws.index)
