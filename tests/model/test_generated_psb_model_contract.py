"""Contract checks for the generated PSB ring-3 model."""

from __future__ import annotations

import numpy as np
import pytest
from xtrack_tools.env import create_xsuite_environment

from tests.psb_tracking import (
    ACD_ELEMENT,
    KINETIC_ENERGY_GEV,
    NATURAL_TUNES,
    QUAD_PREFIX,
    RING,
    SEQ_FILE,
    SEQ_NAME,
)


@pytest.mark.slow
def test_generated_psb_model_contract(psb_model_dir) -> None:
    """Pin the lattice features required by the reconstruction integration tests."""
    sequence = psb_model_dir / SEQ_FILE
    assert sequence.is_file()

    environment = create_xsuite_environment(
        sequence_file=sequence,
        kinetic_energy=KINETIC_ENERGY_GEV,
        seq_name=SEQ_NAME,
        json_file=psb_model_dir / f"{sequence.stem}.json",
    )
    line = environment[SEQ_NAME]
    names = [str(name).upper() for name in line.element_names]
    assert any(name == ACD_ELEMENT for name in names)

    bpm_indices = [index for index, name in enumerate(names) if f"BR{RING}.BPM" in name]
    assert len(bpm_indices) > 4
    acd_index = names.index(ACD_ELEMENT)
    assert min(bpm_indices) < acd_index < max(bpm_indices)

    twiss = line.twiss(method="4d")
    assert twiss.qx == pytest.approx(NATURAL_TUNES[0], abs=2e-2)
    assert twiss.qy == pytest.approx(NATURAL_TUNES[1], abs=2e-2)

    bpm_names = [line.element_names[index] for index in bpm_indices]
    assert np.max(np.abs(twiss.rows[bpm_names].x)) < 1e-10
    assert np.max(np.abs(twiss.rows[bpm_names].dx)) > 1e-2

    powered_bends = [
        name
        for name in line.element_names
        if str(name).lower().startswith("br.bhz") and abs(float(line[name].h)) > 0.0
    ]
    powered_quads = [
        name
        for name in line.element_names
        if str(name).lower().startswith(QUAD_PREFIX)
        and abs(float(getattr(line[name], "k1", 0.0))) > 0.0
    ]
    assert len(powered_bends) >= 16
    assert powered_quads
