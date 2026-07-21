"""Shared PSB AC-dipole tracking setup used by several integration tests.

Both the standalone AC-dipole reconstruction test ([tests/acd/test_psb_acd_momentum.py])
and the dispersive-measurement test ([tests/momentum/test_dispersive_measurement.py])
need one PSB ring-3 AC-dipole excitation tracked on the (off-)momentum closed orbit
together with a matching MAD-NG model. This module centralises that setup so the two
tests share a single, cacheable implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymadng_utils.accelerators import PSB
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles
from xtrack_tools.env import create_xsuite_environment
from xtrack_tools.errors import (
    apply_relative_bend_field_errors,
    apply_vertical_quad_misalignment,
)
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

# The 16 powered PSB main bending dipoles are each modelled as two half-bends
# named ``br.bhzN1`` / ``br.bhzN2`` in xsuite (upper-cased in MAD-NG). These are
# the only powered (non-zero-angle) bends; the ``bi3.bsw*`` injection bumpers have
# zero nominal field.
MAIN_BEND_PREFIX = "br.bhz"
# PSB ring quadrupoles are ``br.qde*`` / ``br.qfo*``.
QUAD_PREFIX = "br.q"


def _apply_bend_errors_to_model(model: ACDipoleMadDriver, new_k0: dict[str, float]) -> None:
    """Write the same absolute bend ``k0`` values onto the MAD-NG model.

    The MAD-NG main bends share xsuite's geometry (identical ``angle``/``length``)
    and nominal ``k0 == h``, so writing the exact ``k0`` computed in
    :func:`_apply_bend_errors_to_line` gives the two codes the same distorted
    closed orbit (agreeing to ~1e-8 at the BPMs).
    """
    for name, k0 in new_k0.items():
        model.mad.send(f"loaded_sequence['{name.upper()}'].k0 = {k0!r}")


def build_psb_tracking_setup(
    data_dir: Path,
    delta_p: float,
    *,
    driven_tunes: tuple[float, float] = DRIVEN_TUNES,
    ramp_turns: int = RAMP_TURNS,
    flattop_turns: int = FLATTOP_TURNS,
    state_markers: bool = True,
    bend_error_rms: float = 0.0,
    bend_error_seed: int = 0,
    apply_bend_errors_to_model: bool = True,
    quad_misalign_y_rms: float = 0.0,
    quad_misalign_seed: int = 0,
) -> dict[str, Any]:
    """Track one PSB AC-dipole excitation seeded on the ``delta_p`` closed orbit.

    Returns a dict with the tracked BPM data (``tracking_df``), the MAD-NG model
    twiss (``tws``), the per-turn truth momenta (``truth``), the MAD-NG ``model``
    and the requested ``delta_p``.

    When ``bend_error_rms > 0`` a seeded relative dipole error of that RMS is added
    to the xsuite tracking line's powered main bends. If
    ``apply_bend_errors_to_model`` (the default), the *same* absolute ``k0`` values
    are also written onto the MAD-NG model so both share the same distorted
    (dipole-error) closed orbit and the returned ``tws`` matches the model. Set it
    ``False`` to distort only the tracked data while the model stays nominal — a
    "twiss on zero" scenario where the model twiss does not represent the machine
    closed orbit.

    When ``quad_misalign_y_rms > 0`` a seeded vertical misalignment of that RMS
    (metres) is applied to the tracking line's quadrupoles only, distorting the
    vertical closed orbit of the data while the MAD-NG model stays nominal.
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

    bend_k0: dict[str, float] = {}
    if bend_error_rms > 0.0:
        bend_k0 = apply_relative_bend_field_errors(
            line, rms=bend_error_rms, seed=bend_error_seed, name_prefix=MAIN_BEND_PREFIX
        )
    if quad_misalign_y_rms > 0.0:
        apply_vertical_quad_misalignment(
            line, rms=quad_misalign_y_rms, seed=quad_misalign_seed, name_prefix=QUAD_PREFIX
        )

    # Use explicit on- and off-momentum natural tunes: reconstruction optics are
    # matched off momentum, while the closed-orbit reference is on momentum only
    # (never including the dispersive orbit).
    off_momentum_tws = line.twiss(method="4d", delta0=delta_p)
    particle_coords = {
        "x": [float(off_momentum_tws.x[0])],
        "px": [float(off_momentum_tws.px[0])],
        "y": [float(off_momentum_tws.y[0])],
        "py": [float(off_momentum_tws.py[0])],
        "delta": [delta_p],
    }
    monitored_line = run_ac_dipole_tracking_with_particles(
        line=line,
        acd_marker=ACD_ELEMENT,
        sequence_name=SEQ_NAME,
        tws=off_momentum_tws,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=list(driven_tunes),
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
    if bend_k0 and apply_bend_errors_to_model:
        # Give the MAD-NG model the same distorted closed orbit as the tracked line,
        # then refresh the cached element twiss the closed-orbit check reads.
        _apply_bend_errors_to_model(model, bend_k0)
        model.twiss_elements = model.run_twiss(observe=0)
    # coupling=true so the twiss carries the MAD-X betx/alfx columns the dispersive
    # measurement pipeline needs when converting to MAD-X format.
    tws = model.run_twiss(observe=1, coupling=True)

    # The observed twiss handed to the reconstruction must report the *same* closed
    # orbit as the model's full-element twiss; they are one MAD-NG solution sampled at
    # different points. Assert it so a twiss/observe regression cannot silently feed a
    # mismatched orbit into the reconstruction.
    common = tws.index.intersection(model.twiss_elements.index)
    for coord in ("x", "px", "y", "py"):
        max_diff = float(
            np.abs(
                tws.loc[common, coord].to_numpy(float)
                - model.twiss_elements.loc[common, coord].to_numpy(float)
            ).max()
        )
        assert max_diff < 1e-12, (
            f"observed twiss {coord} disagrees with model.twiss_elements at "
            f"{len(common)} common rows (max|diff|={max_diff:.3e})"
        )

    truth = get_truth(tracking_df, tws)
    return {
        "tracking_df": tracking_df,
        "tws": tws,
        "truth": truth,
        "model": model,
        "delta_p": delta_p,
        "bend_k0": bend_k0,
    }
