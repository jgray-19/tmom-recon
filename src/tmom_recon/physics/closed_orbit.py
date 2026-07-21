import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Warn when the model and data closed orbits differ by more than this, in metres.
CLOSED_ORBIT_WARN_TOLERANCE = 1e-3


def parse_plane_spec(
    spec: str | tuple[str, ...] | None, *, field: str = "planes"
) -> tuple[str, ...]:
    """Normalise a plane specification such as ``"xy"``/``"yx"``/``"x"`` to ``("x", "y")`` order.

    Args:
        spec: Plane string (case-insensitive, any order), an already-parsed
            tuple, or ``None``/``""`` for no planes.
        field: Name used in error messages.

    Returns:
        Canonically ordered tuple drawn from ``("x", "y")``.

    Raises:
        ValueError: If *spec* contains a character other than ``x``/``y``, or a
            duplicate plane.
    """
    if spec is None:
        return ()
    chars = [c for c in ("".join(spec)).strip().lower() if not c.isspace()]
    unknown = sorted(set(chars) - {"x", "y"})
    if unknown:
        raise ValueError(f"{field}: unknown plane(s) {unknown}; expected only 'x' and 'y'.")
    if len(set(chars)) != len(chars):
        raise ValueError(f"{field}: duplicate plane in {spec!r}.")
    return tuple(plane for plane in ("x", "y") if plane in chars)


def warn_on_closed_orbit_mismatch(
    twiss_co: pd.DataFrame,
    data_co: pd.DataFrame,
    *,
    planes: tuple[str, ...] = ("x", "y"),
) -> None:
    """Warn when the twiss closed orbit is far from the data closed orbit.

    Args:
        twiss_co: Model closed orbit, columns ``x``/``y`` indexed by BPM.
        data_co: Data-mean closed orbit, same index.
        planes: Planes to check.
    """
    for plane in planes:
        if plane not in twiss_co.columns or plane not in data_co.columns:
            continue
        diff = (twiss_co[plane] - data_co[plane]).astype(float).dropna().abs()
        if diff.empty or diff.max() <= CLOSED_ORBIT_WARN_TOLERANCE:
            continue
        logger.warning(
            "Model closed orbit disagrees with the data closed orbit in plane %s by more "
            "than %.1f mm: worst BPM %s at %.2f mm (rms %.2f mm). Consider "
            "data_mean_closed_orbit_planes=%r.",
            plane,
            1e3 * CLOSED_ORBIT_WARN_TOLERANCE,
            diff.idxmax(),
            1e3 * diff.max(),
            1e3 * float(np.sqrt(np.mean(diff**2))),
            plane,
        )


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
