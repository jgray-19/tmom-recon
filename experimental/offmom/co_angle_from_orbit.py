"""Can the closed-orbit angle be measured instead of modelled?

`restore_closed_orbit_and_reference_momenta` currently takes px/py of the closed
orbit from the model's `closed_orbit_tws`. With unknown dipole errors that orbit
is wrong, and §B of the investigation notes traced the px R^2 ~ 0.82 ceiling to
exactly that: a 1.044e-04 rad angle error the orbit fit could not remove.

The alternative tested here: throw the model's closed *orbit* away and keep only
its *optics*, then reconstruct the closed-orbit angle from the measured orbit
itself by neighbour-pair transport -- the same machinery the turn-by-turn
reconstruction already uses.

The physical case for it: a dipole error distorts the orbit at first order but
the optics only at second, so the model's optics is the robust half and its
orbit the fragile half. The case against: neighbour-pair transport assumes no
kick *between* the two BPMs, which a dipole error violates by construction.
This script measures which effect wins.

Run with `python -m experimental.offmom.co_angle_from_orbit` from the repo root.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tests.psb_tracking import build_psb_tracking_setup
from tmom_recon.optics import resolve_optics
from tmom_recon.physics.transverse import reconstruct_momenta

logging.basicConfig(level=logging.ERROR)

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
BEND_ERROR_RMS = 1e-4
TURNS_FOR_TRANSPORT = 3


def measured_closed_orbit(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """Per-BPM turn mean: betatron motion averages away, the closed orbit does not."""
    grouped = tracking_df.groupby("name", sort=False, observed=True)
    return pd.DataFrame({col: grouped[col].mean() for col in ("x", "y", "px", "py")})


def angle_from_orbit(orbit: pd.DataFrame, optics_tws: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct closed-orbit px/py from its positions and the model optics.

    The orbit is fed in as a few identical "turns" so the neighbour-pair
    machinery has something to chew on; with dispersion off and a zero closed
    orbit to subtract, the result is the angle implied by free transport between
    each BPM and its neighbours.
    """
    bpms = orbit.index.tolist()
    data = pd.DataFrame(
        {
            "name": np.tile(bpms, TURNS_FOR_TRANSPORT),
            "turn": np.repeat(np.arange(TURNS_FOR_TRANSPORT), len(bpms)),
            "x": np.tile(orbit["x"].to_numpy(), TURNS_FOR_TRANSPORT),
            "y": np.tile(orbit["y"].to_numpy(), TURNS_FOR_TRANSPORT),
        }
    )
    data["var_x"] = 0.0
    data["var_y"] = 0.0
    zero = pd.DataFrame({"x": 0.0, "y": 0.0}, index=pd.Index(bpms, name="name"))
    optics = resolve_optics(
        optics_tws=optics_tws,
        closed_orbit_tws=zero,
        reference=zero,
        use_dispersion=False,
        bpm_names=bpms,
    )
    out = reconstruct_momenta(data, optics, info=False)
    first_turn = out[out["turn"] == 1].set_index("name")
    return first_turn[["px", "py"]]


def report(label: str, truth: pd.Series, estimate: pd.Series) -> dict:
    residual = (estimate - truth).dropna()
    return {
        "case": label,
        "n": len(residual),
        "truth_rms": float(np.sqrt(np.mean(truth.dropna() ** 2))),
        "err_rms": float(np.sqrt(np.mean(residual**2))),
        "err_max": float(np.abs(residual).max()),
    }


def main() -> None:
    rows = []
    for label, apply_to_model in (("model knows errors", True), ("model blind", False)):
        setup = build_psb_tracking_setup(
            DATA_DIR,
            0.0,
            state_markers=False,
            bend_error_rms=BEND_ERROR_RMS,
            apply_bend_errors_to_model=apply_to_model,
        )
        tws = setup["tws"]
        orbit = measured_closed_orbit(setup["tracking_df"])
        orbit = orbit.loc[orbit.index.intersection(tws.index)]

        measured = angle_from_orbit(orbit[["x", "y"]], tws)
        common = measured.index.intersection(orbit.index)

        # What the pipeline does today: take the angle straight from the model.
        modelled = tws.loc[common, ["px", "py"]]

        for plane in ("px", "py"):
            rows.append(
                report(
                    f"{label} | {plane} | from orbit",
                    orbit.loc[common, plane],
                    measured.loc[common, plane],
                )
            )
            rows.append(
                report(f"{label} | {plane} | from model", orbit.loc[common, plane], modelled[plane])
            )

    table = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
