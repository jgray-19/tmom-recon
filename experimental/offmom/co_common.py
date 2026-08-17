"""Shared helpers for the off-momentum closed-orbit investigation (Part A).

Everything here works on *xsuite* lines: the closed orbit at a given ``delta_p``
is taken from ``line.twiss(method="4d", delta0=dp)`` rather than from the mean of
tracked AC-dipole data. Justification: the two agree to ~1e-8 (established in
NOTES_offmomentum_closed_orbit.md §2, where the exact MAD-NG orbit matched the
tracked turn-mean to that level), and the twiss route costs milliseconds instead
of the ~1-2 min an AC-dipole tracking run takes, which is what makes a 32-column
response matrix affordable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.errors import apply_relative_bend_field_errors

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.psb_tracking import (  # noqa: E402
    KINETIC_ENERGY_GEV,
    MAIN_BEND_PREFIX,
    SEQ_FILE,
    SEQ_NAME,
    _apply_relative_quad_gradient_errors,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
BPM_PREFIX = "br3.bpm"


def build_line(
    *, bend_rms: float = 0.0, bend_seed: int = 7, quad_rms: float = 0.0, quad_seed: int = 11
):
    seq = DATA_DIR / "sequences" / SEQ_FILE
    env = create_xsuite_environment(
        sequence_file=seq,
        kinetic_energy=KINETIC_ENERGY_GEV,
        seq_name=SEQ_NAME,
        json_file=DATA_DIR / "sequences" / f"{seq.stem}.json",
    )
    line = env[SEQ_NAME].copy()
    bend_k0, quad_k1 = {}, {}
    if bend_rms > 0:
        bend_k0 = apply_relative_bend_field_errors(
            line, rms=bend_rms, seed=bend_seed, name_prefix=MAIN_BEND_PREFIX
        )
    if quad_rms > 0:
        quad_k1 = _apply_relative_quad_gradient_errors(line, rms=quad_rms, seed=quad_seed)
    return line, bend_k0, quad_k1


def bpm_names(line) -> list[str]:
    return [n for n in line.element_names if str(n).lower().startswith(BPM_PREFIX)]


def bend_names(line) -> list[str]:
    """Powered main-bend elements (aperture markers sharing the prefix are skipped)."""
    out = []
    for n in line.element_names:
        if not str(n).lower().startswith(MAIN_BEND_PREFIX):
            continue
        el = line[n]
        if getattr(el, "k0", None) is None or float(getattr(el, "angle", 0.0)) == 0.0:
            continue
        out.append(n)
    return out


def nominal_k0(line, name) -> float:
    """Reference bend field ``h = angle / length`` (the line stores ``k0='from_h'``)."""
    el = line[name]
    return float(el.angle) / float(el.length)


def closed_orbit(line, dp: float, names: list[str]) -> np.ndarray:
    """Return the concatenated (x, y) closed orbit at *names* for momentum *dp*."""
    tw = line.twiss(method="4d", delta0=float(dp))
    row = tw.rows[names]
    return np.concatenate([np.asarray(row.x, float), np.asarray(row.y, float)])
