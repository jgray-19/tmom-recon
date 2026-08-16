"""How does the closed-orbit angle estimate depend on the bend and quad error RMS?

`co_angle_realistic.py` answered "which estimator wins" at one error level. This
sweeps both knobs to check the answer is not an artefact of that choice, and to
separate fitting bends alone from fitting bends *and* quads -- bends alone was
found to go actively wrong when the quad errors dominate.

The two knobs do different things, which is the point of separating them:

  bend RMS  kicks the orbit directly, and is the thing being fitted. More error
            means more signal for the fit but also more of it in the null space
            of a 16-observation / 32-unknown problem.
  quad RMS  never kicks the orbit on its own. It perturbs beta, phase and
            dispersion, so it corrupts the *response matrix* the fit inverts and
            the *optics* the transport route relies on. It is never fitted here,
            so it is pure unmodelled error -- the honest stand-in for "we do not
            know the machine".

Reported at dp/p = 0 (where the closed orbit is measured) and dp/p = 8e-3.

Run with `python -m experimental.offmom.co_angle_error_sweep` from the repo root.
"""

from __future__ import annotations

import logging

import pandas as pd

from experimental.offmom.co_angle_realistic import Baseline, evaluate

logging.basicConfig(level=logging.ERROR)

BEND_RMS_VALUES = (1e-4, 4e-4, 8e-4, 2e-3)
QUAD_RMS_VALUES = (0.0, 1e-3, 5e-3)
NOISE = 3e-6
REPORT_DPS = (0.0, 8e-3)


def main() -> None:
    base = Baseline()
    print(f"{len(base.bpms)} BPMs, {len(base.bends)} bend unknowns, orbit noise {NOISE:.0e} m")

    rows = []
    for bend_rms in BEND_RMS_VALUES:
        for quad_rms in QUAD_RMS_VALUES:
            rows += evaluate(
                base,
                bend_rms=bend_rms,
                quad_rms=quad_rms,
                noise=NOISE,
                report_dps=REPORT_DPS,
                # Vertical is identically zero in this lattice, so it carries no
                # information about the estimators -- only about noise gain.
                planes=("px",),
            )
            print(f"  done bend={bend_rms:.0e} quad={quad_rms:.0e}", flush=True)

    table = pd.DataFrame(rows).drop(columns=["plane", "noise"])
    # How much the quad fit buys over bends alone, and over doing nothing.
    table["gain_b"] = table["nominal"] / table["fit_bend"]
    table["gain_bq"] = table["nominal"] / table["fit_bend_quad"]
    table["bq_over_b"] = table["fit_bend"] / table["fit_bend_quad"]

    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        for dp in REPORT_DPS:
            print(f"\n=== px closed-orbit angle RMS error [rad], dp/p = {dp:g}")
            print(table[table["dp"] == dp].drop(columns="dp").to_string(index=False))


if __name__ == "__main__":
    main()
