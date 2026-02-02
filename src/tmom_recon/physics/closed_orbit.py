import logging

import pandas as pd

logger = logging.getLogger(__name__)


def estimate_closed_orbit(
    data: pd.DataFrame, tws: pd.DataFrame, dpp_est: float = 0.0
) -> pd.DataFrame:
    """Estimate closed orbit from tracking data.

    Args:
        data: Tracking data with BPM readings. Must contain columns: ["name", "x", "y"].
        tws: Twiss parameters DataFrame. Must have columns ["dx", "dy"] and be indexed by BPM name.
        dpp_est: Estimated relative momentum deviation.

    Returns:
        DataFrame indexed like tws.index with columns: x, y, var_x, var_y.
    """
    if "name" not in data.columns or "x" not in data.columns or "y" not in data.columns:
        raise ValueError('`data` must contain columns ["name", "x", "y"].')

    # Map dispersion to each row (per BPM), then correct positions turn-by-turn
    x_corr = data["x"] - dpp_est * data["name"].map(tws["dx"].to_dict())
    y_corr = data["y"] - dpp_est * data["name"].map(tws["dy"].to_dict())

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
