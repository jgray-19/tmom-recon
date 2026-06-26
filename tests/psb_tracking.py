"""Shared PSB AC-dipole tracking setup used by several integration tests.

Both the standalone AC-dipole reconstruction test ([tests/acd/test_psb_acd_momentum.py])
and the dispersive-measurement test ([tests/momentum/test_dispersive_measurement.py])
need one PSB ring-3 AC-dipole excitation tracked on the (off-)momentum closed orbit
together with a matching MAD-NG model. This module centralises that setup so the two
tests share a single, cacheable implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymadng_utils.accelerators import PSB
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles
from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.monitors import process_tracking_data

from tests.momentum.momentum_test_utils import get_truth
from tmom_recon.acd.madng_driver import ACDipoleMadDriver

if TYPE_CHECKING:
    from pathlib import Path

RING = 3
SEQ_FILE = "psb3_saved.seq"
SEQ_NAME = f"psb{RING}"
KINETIC_ENERGY_GEV = 0.160
ACD_ELEMENT = f"BR{RING}.DES3L1"
BPM_PATTERN = rf"(?i)br{RING}\.bpm.*"
DRIVEN_TUNES = (0.16, 0.24)
RAMP_TURNS = 1000
FLATTOP_TURNS = 1000


def build_psb_tracking_setup(
    data_dir: Path,
    delta_p: float,
    *,
    ramp_turns: int = RAMP_TURNS,
    flattop_turns: int = FLATTOP_TURNS,
    state_markers: bool = True,
) -> dict[str, Any]:
    """Track one PSB AC-dipole excitation seeded on the ``delta_p`` closed orbit.

    Returns a dict with the tracked BPM data (``tracking_df``), the MAD-NG model
    twiss (``tws``), the per-turn truth momenta (``truth``), the MAD-NG ``model``
    and the requested ``delta_p``.
    """
    delta_p = float(delta_p)
    seq = data_dir / "sequences" / SEQ_FILE
    json_path = data_dir / "sequences" / f"{seq.stem}.json"

    env = create_xsuite_environment(
        sequence_file=seq,
        kinetic_energy=KINETIC_ENERGY_GEV,
        seq_name=SEQ_NAME,
        json_file=json_path,
    )
    line = env[SEQ_NAME].copy()

    # Seed the tracked particle on the (off-)momentum closed orbit so the
    # excitation starts from the correct dispersive coordinates.
    co = line.twiss(method="4d", delta0=delta_p)
    particle_coords = {
        "x": [float(co.x[0])],
        "px": [float(co.px[0])],
        "y": [float(co.y[0])],
        "py": [float(co.py[0])],
        "delta": [delta_p],
    }
    monitored_line = run_ac_dipole_tracking_with_particles(
        line=line,
        acd_marker=ACD_ELEMENT,
        sequence_name=SEQ_NAME,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=list(DRIVEN_TUNES),
        bpm_pattern=BPM_PATTERN,
        particle_coords=particle_coords,
        state_markers=state_markers,
    )
    tracking_df = process_tracking_data(
        monitored_line, ramp_turns=ramp_turns, flattop_turns=flattop_turns
    )

    accelerator = PSB(sequence_file=seq, ring=RING, kinetic_energy=KINETIC_ENERGY_GEV)
    model = ACDipoleMadDriver(
        accelerator=accelerator, pt=accelerator.dp2pt(delta_p), observed_elements=ACD_ELEMENT
    )
    # coupling=true so the twiss carries the MAD-X betx/alfx columns the dispersive
    # measurement pipeline needs when converting to MAD-X format.
    tws = model.run_twiss(observe=1, coupling=True)
    truth = get_truth(tracking_df, tws)
    return {
        "tracking_df": tracking_df,
        "tws": tws,
        "truth": truth,
        "model": model,
        "delta_p": delta_p,
    }
