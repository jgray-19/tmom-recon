"""Closed-orbit angle in the scenario that will actually happen.

The machine has bend *and* quad errors and we never learn them. What we get is a
plain closed-orbit measurement at a few momenta, taken with the AC dipole off.
From that we can estimate the bend errors (Part A) and build a corrected model.

So which of these gives the best closed-orbit px/py at the BPMs?

  nominal      the model as built, knowing nothing -- what the pipeline uses today
  fit_bend       a model carrying only the bend errors estimated from the orbits
  fit_bend_quad  bends *and* quad gradients estimated, by alternating solves
  transport    no model orbit at all: neighbour-pair transport of the *measured*
               orbit positions, using only the model's optics

Truth is the machine line's own twiss px/py at the BPMs. Everything runs on
xsuite closed orbits (no tracking), so the whole comparison costs seconds.

Run with `python -m experimental.offmom.co_angle_realistic` from the repo root.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tfs

sys.path.insert(0, str(Path(__file__).resolve().parent))

from co_common import bend_names, bpm_names, build_line, closed_orbit, nominal_k0  # noqa: E402
from part_a3_quad_fit import quad_names, quad_response  # noqa: E402
from part_a_bend_fit import BEND_RMS, QUAD_RMS, response_matrix, svd_solve  # noqa: E402

from tmom_recon.optics import resolve_optics  # noqa: E402
from tmom_recon.physics.transverse import reconstruct_momenta  # noqa: E402

logging.basicConfig(level=logging.ERROR)

# Momenta at which the closed orbit is measured, and the one we report at.
FIT_DPS = (0.0, 1e-3, -1e-3, 3e-3, -3e-3, 8e-3, -8e-3)
REPORT_DPS = (0.0, 1e-3, 3e-3, 8e-3)
N_SV = 16  # 16 BPMs -> 16 usable modes; see NOTES section A.3
N_SV_QUAD = 16
# The quad response is built from the orbit at large |dp|, where the dispersion
# the quads control dominates; the bend step uses dp=0, where it does not.
QUAD_FIT_DPS = (8e-3, -8e-3)
FIT_ITERATIONS = 4

# Orbit noise on the *measured* closed orbit. A per-turn BPM resolution of 1e-4 m
# averaged over ~1000 turns lands around 3e-6 m; 1e-5 m is the pessimistic end.
NOISE_LEVELS = (0.0, 3e-6, 1e-5)
NOISE_SEED = 20260811


def twiss_frame(line, bpms: list[str]) -> tfs.TfsDataFrame:
    """Model optics at the BPMs, in the column names tmom_recon expects."""
    tw = line.twiss(method="4d")
    row = tw.rows[bpms]
    frame = tfs.TfsDataFrame(
        {
            "beta11": np.asarray(row.betx, float),
            "beta22": np.asarray(row.bety, float),
            "alfa11": np.asarray(row.alfx, float),
            "alfa22": np.asarray(row.alfy, float),
            # xsuite mux/muy are already in tune units, like MAD-NG's mu1/mu2.
            "mu1": np.asarray(row.mux, float),
            "mu2": np.asarray(row.muy, float),
            "s": np.asarray(row.s, float),
            "x": np.asarray(row.x, float),
            "y": np.asarray(row.y, float),
            "px": np.asarray(row.px, float),
            "py": np.asarray(row.py, float),
        },
        index=pd.Index(list(bpms), name="name"),
    )
    frame.headers = {"q1": float(tw.qx), "q2": float(tw.qy)}
    return frame


def true_orbit(line, dp: float, bpms: list[str]) -> pd.DataFrame:
    """The machine's own closed orbit and, crucially, its angles."""
    row = line.twiss(method="4d", delta0=float(dp)).rows[bpms]
    return pd.DataFrame(
        {
            "x": np.asarray(row.x, float),
            "y": np.asarray(row.y, float),
            "px": np.asarray(row.px, float),
            "py": np.asarray(row.py, float),
        },
        index=pd.Index(list(bpms), name="name"),
    )


def angle_by_transport(orbit_xy: pd.DataFrame, optics: tfs.TfsDataFrame) -> pd.DataFrame:
    """Closed-orbit angle from measured positions plus model optics only.

    The orbit is presented as three identical turns so the neighbour-pair
    machinery has a middle turn with both neighbours defined. Dispersion is off
    and the subtracted closed orbit is zero, so what comes back is the angle
    implied by free transport between each BPM and its neighbours.
    """
    bpms = orbit_xy.index.tolist()
    turns = 3
    data = pd.DataFrame(
        {
            "name": np.tile(bpms, turns),
            "turn": np.repeat(np.arange(turns), len(bpms)),
            "x": np.tile(orbit_xy["x"].to_numpy(), turns),
            "y": np.tile(orbit_xy["y"].to_numpy(), turns),
            "var_x": 0.0,
            "var_y": 0.0,
        }
    )
    zero = pd.DataFrame({"x": 0.0, "y": 0.0}, index=pd.Index(bpms, name="name"))
    resolved = resolve_optics(
        optics_tws=optics,
        closed_orbit_tws=zero,
        reference=zero,
        use_dispersion=False,
        bpm_names=bpms,
    )
    out = reconstruct_momenta(data, resolved, info=False)
    return out[out["turn"] == 1].set_index("name")[["px", "py"]]


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if values.size else float("nan")


class Baseline:
    """Everything that depends only on the nominal model, so it is built once.

    The response matrix is a derivative of the *model*, not of the machine, so it
    is identical for every error level -- which is what makes sweeping bend and
    quad RMS cheap.
    """

    def __init__(self) -> None:
        self.model, _, _ = build_line()
        self.bpms = bpm_names(self.model)
        self.bends = bend_names(self.model)
        self.quads = quad_names(self.model)
        self.response = response_matrix(self.model, self.bends, self.bpms)
        self.quad_response = quad_response(self.model, self.quads, self.bpms, QUAD_FIT_DPS)
        self.co_model = {dp: closed_orbit(self.model, dp, self.bpms) for dp in FIT_DPS}
        self.optics_nominal = twiss_frame(self.model, self.bpms)

    def corrected(self, dk0, dq=None):
        """A line carrying the fitted bend (and optionally quad) errors."""
        line, _, _ = build_line()
        for name, delta in zip(self.bends, dk0, strict=True):
            line[name].k0 = nominal_k0(line, name) + float(delta)
        if dq is not None:
            for name, delta in zip(self.quads, dq, strict=True):
                line[name].k1 = float(line[name].k1) * (1.0 + float(delta))
        return line


def fit_bends_only(base: Baseline, measured: dict) -> np.ndarray:
    """One linear solve: all the orbit residual is blamed on the bends."""
    stacked = np.vstack([base.response] * len(FIT_DPS))
    residual = np.concatenate([measured[dp] - base.co_model[dp] for dp in FIT_DPS])
    return svd_solve(stacked, residual, N_SV)


def fit_bends_and_quads(base: Baseline, measured: dict) -> tuple[np.ndarray, np.ndarray]:
    """Alternate the two solves, so dispersion error is not blamed on the bends.

    Bends are fitted against the dp=0 orbit, where dispersion contributes
    nothing; quads against the residual at large |dp|, which is where a wrong
    dispersion shows up. Iterating decouples the two.
    """
    dk0 = np.zeros(len(base.bends))
    dq = np.zeros(len(base.quads))
    for _ in range(FIT_ITERATIONS):
        current = base.corrected(dk0, dq)
        dk0 = dk0 + svd_solve(
            base.response, measured[0.0] - closed_orbit(current, 0.0, base.bpms), N_SV
        )
        current = base.corrected(dk0, dq)
        residual = np.concatenate(
            [measured[dp] - closed_orbit(current, dp, base.bpms) for dp in QUAD_FIT_DPS]
        )
        dq = dq + svd_solve(base.quad_response, residual, N_SV_QUAD)
    return dk0, dq


def evaluate(
    base: Baseline,
    *,
    bend_rms: float,
    quad_rms: float,
    noise: float,
    report_dps: tuple[float, ...] = REPORT_DPS,
    planes: tuple[str, ...] = ("px", "py"),
) -> list[dict]:
    """Compare nominal / fitted / transport closed-orbit angles at one error level."""
    machine, _, _ = build_line(bend_rms=bend_rms, quad_rms=quad_rms)
    bpms = base.bpms
    co_machine = {dp: closed_orbit(machine, dp, bpms) for dp in FIT_DPS}

    rng = np.random.default_rng(NOISE_SEED)
    measured = {
        dp: (co_machine[dp] + rng.normal(0.0, noise, co_machine[dp].shape))
        if noise
        else co_machine[dp]
        for dp in FIT_DPS
    }

    # Both fits see exactly what the control room sees: measured minus model.
    fitted_b = base.corrected(fit_bends_only(base, measured))
    dk0_bq, dq_bq = fit_bends_and_quads(base, measured)
    fitted_bq = base.corrected(dk0_bq, dq_bq)
    optics_fitted = twiss_frame(fitted_bq, bpms)

    rows = []
    for dp in report_dps:
        truth = true_orbit(machine, dp, bpms)
        nominal_tw = true_orbit(base.model, dp, bpms)
        fit_b_tw = true_orbit(fitted_b, dp, bpms)
        fit_bq_tw = true_orbit(fitted_bq, dp, bpms)

        noisy = measured[dp]
        half = len(bpms)
        orbit_xy = pd.DataFrame(
            {"x": noisy[:half], "y": noisy[half:]}, index=pd.Index(bpms, name="name")
        )
        transported = angle_by_transport(orbit_xy, optics_fitted)

        for plane in planes:
            truth_plane = truth[plane]
            rows.append(
                {
                    "bend_rms": bend_rms,
                    "quad_rms": quad_rms,
                    "noise": noise,
                    "dp": dp,
                    "plane": plane,
                    "truth_rms": rms(truth_plane.to_numpy()),
                    "nominal": rms((nominal_tw[plane] - truth_plane).to_numpy()),
                    "fit_bend": rms((fit_b_tw[plane] - truth_plane).to_numpy()),
                    "fit_bend_quad": rms((fit_bq_tw[plane] - truth_plane).to_numpy()),
                    "transport": rms(
                        (transported[plane] - truth_plane.loc[transported.index]).to_numpy()
                    ),
                }
            )
    return rows


def main() -> None:
    base = Baseline()
    print(f"{len(base.bpms)} BPMs, {len(base.bends)} bend unknowns, fitting with n_sv={N_SV}")

    rows = []
    for noise in NOISE_LEVELS:
        rows += evaluate(base, bend_rms=BEND_RMS, quad_rms=QUAD_RMS, noise=noise)

    table = pd.DataFrame(rows).drop(columns=["bend_rms", "quad_rms"])
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        for plane in ("px", "py"):
            print(f"\n=== {plane} closed-orbit angle, RMS error over {len(base.bpms)} BPMs [rad]")
            print(table[table["plane"] == plane].drop(columns="plane").to_string(index=False))


if __name__ == "__main__":
    main()
