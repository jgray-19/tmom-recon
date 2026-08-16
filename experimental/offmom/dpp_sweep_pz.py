"""Does `calculate_pz` recover px/py at every dp/p, without the AC dipole?

Sweeps dp/p over the PSB ring-3 tracking setup and, for each point, runs the
plain (non-ACD) reconstruction twice: once with the model twiss as generated
(second-order dispersion columns present) and once with `ddx`/`ddpx`/`ddy`/`ddpy`
dropped, which is exactly the first-order behaviour the code had before.

Reported per dp/p:
  pt_est vs the true model pt -- how well the momentum itself is recovered;
  px/py RMSE vs tracking truth -- whether that pt lands correctly in the momenta.

Run with `python -m experimental.offmom.dpp_sweep_pz` from the repo root.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from tests.momentum.momentum_test_utils import rmse, zero_momentum_reference
from tests.psb_tracking import build_psb_tracking_setup
from tmom_recon import ModelDetails, calculate_pz, reconstruction

logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
SECOND_ORDER = ("ddx", "ddpx", "ddy", "ddpy")
DELTA_PS = (0.0, 1e-4, 1e-3, 3e-3, 5e-3, 8e-3, 1e-2)


def _run(tracking_df: pd.DataFrame, model_details: ModelDetails, *, second_order: bool):
    """Reconstruct once; `second_order=False` strips the chrom columns first."""
    # `calculate_pz` generates its own twiss, so the only lever for forcing the
    # old first-order behaviour is to strip the chrom columns off the generated
    # model twiss -- everything downstream treats them as optional and falls
    # back to first order when they are absent.
    original = reconstruction.resolve_model_details

    def without_second_order(*args, **kwargs):
        resolved = original(*args, **kwargs)
        return replace(
            resolved,
            optics_tws=resolved.optics_tws.drop(columns=list(SECOND_ORDER), errors="ignore"),
            closed_orbit_tws=resolved.closed_orbit_tws.drop(
                columns=list(SECOND_ORDER), errors="ignore"
            ),
        )

    if not second_order:
        reconstruction.resolve_model_details = without_second_order
    try:
        return calculate_pz(
            tracking_df.copy(deep=True),
            model_details,
            reference=zero_momentum_reference(tracking_df),
            use_dispersion=True,
            info=False,
        )
    finally:
        reconstruction.resolve_model_details = original


def main() -> None:
    rows = []
    for delta_p in DELTA_PS:
        setup = build_psb_tracking_setup(DATA_DIR, delta_p, state_markers=False)
        tracking_df = setup["tracking_df"]
        truth = setup["truth"]
        model = setup["model"]
        pt_true = model.pt
        # The model is built on momentum: pt is what the reconstruction must
        # recover from the orbit, not something it is told.
        details = ModelDetails(accelerator=model.accelerator, pt=0.0)

        for label, second_order in (("2nd order", True), ("1st order", False)):
            result = _run(tracking_df, details, second_order=second_order)
            merged = truth.merge(result[["name", "turn", "px", "py"]], on=["name", "turn"])
            rows.append(
                {
                    "dp/p": delta_p,
                    "order": label,
                    "pt_true": pt_true,
                    "pt_est": result.attrs["PT_EST"],
                    "pt_rel_err": (
                        abs(result.attrs["PT_EST"] - pt_true) / abs(pt_true)
                        if pt_true
                        else abs(result.attrs["PT_EST"])
                    ),
                    "px_rmse": rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy()),
                    "py_rmse": rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy()),
                }
            )
            print(rows[-1], flush=True)

    table = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda v: f"{v:.4e}"):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
