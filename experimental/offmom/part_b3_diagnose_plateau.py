"""Decompose the Part B plateau: is the residual an offset, a gain or a phase error?

For one delta_p, reconstructs with the nominal model, the best fitted model and the
true-error model, and fits reconstructed = a * true + b per BPM side.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from co_common import build_line  # noqa: E402
from part_b_crossover import (  # noqa: E402
    ACD_DRIVEN_TUNES,
    BEND_RMS,
    BEND_SEED,
    DATA_DIR,
    QUAD_RMS,
    QUAD_SEED,
    strengths_from,
)

from tests.acd.acd_test_helpers import _truth_at, acd_state_marker_names, r_squared  # noqa: E402
from tests.psb_tracking import ACD_ELEMENT, build_psb_tracking_setup  # noqa: E402
from tmom_recon import ACDipoleConfig, ModelDetails, calculate_pz  # noqa: E402

DP = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3


def main():
    warnings.filterwarnings("ignore")
    nominal, _, _ = build_line()
    _, bend_true, quad_true = build_line(
        bend_rms=BEND_RMS, bend_seed=BEND_SEED, quad_rms=QUAD_RMS, quad_seed=QUAD_SEED
    )
    d = np.load(Path(__file__).with_name("fitted_errors_q16.npz"), allow_pickle=True)
    bends, quads = [str(x) for x in d["bends"]], [str(x) for x in d["quads"]]
    fitted = strengths_from(d["dk0"], d["dq"], bends, quads, nominal)
    bend_only = strengths_from(d["dk0"], np.zeros(len(quads)), bends, quads, nominal)
    truth_s = {f"{n.upper()}.k0": v for n, v in bend_true.items()}
    truth_s.update({f"{n.upper()}.k1": v for n, v in quad_true.items()})
    # bends true, quads from the fit: isolates which of the two fits limits things
    mixed = dict(bend_only)
    mixed.update({f"{n.upper()}.k0": v for n, v in bend_true.items()})

    setup = build_psb_tracking_setup(
        DATA_DIR,
        delta_p=DP,
        driven_tunes=ACD_DRIVEN_TUNES,
        bend_error_rms=BEND_RMS,
        bend_error_seed=BEND_SEED,
        apply_bend_errors_to_model=False,
        quad_error_rms=QUAD_RMS,
        quad_error_seed=QUAD_SEED,
        apply_quad_errors_to_model=False,
    )
    df = setup["tracking_df"]
    model = setup["model"]
    before, after = acd_state_marker_names(model)
    bpm_df = df.loc[~df["name"].isin([before, after])].copy()

    for label, strengths in (
        ("nominal", {}),
        ("bend-only fit", bend_only),
        ("bend+quad q16 fit", fitted),
        ("true bends + fitted quads", mixed),
        ("TRUE errors", truth_s),
    ):
        result = calculate_pz(
            bpm_df,
            model_details=ModelDetails(
                accelerator=model.accelerator, pt=model.pt, magnet_strengths=strengths
            ),
            acd=ACDipoleConfig(
                ac_dipole_marker=ACD_ELEMENT,
                driven_tunes=ACD_DRIVEN_TUNES,
                dispersive_closed_orbit=False,
            ),
            acd_only=True,
        )
        s = result.attrs["summary"]
        print(f"\n== {label}")
        for side in ("upstream", "downstream"):
            m = s.merge(_truth_at(df, result.attrs[f"bpm_{side}"]), on="turn", how="inner")
            t = m["px_true"].to_numpy()
            p = m[f"px_bpm_{side}"].to_numpy()
            a, b = np.polyfit(t, p, 1)
            print(
                f"   px {side:<10} R2={r_squared(t, p):9.6f}  gain={a:8.5f} "
                f"offset={b:+10.3e}  R2_after_affine={r_squared(t, (p - b) / a):9.6f}  "
                f"amp_true={t.std():.3e}"
            )


if __name__ == "__main__":
    main()
