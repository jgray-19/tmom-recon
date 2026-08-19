"""Generated PSB model infrastructure contracts."""

from __future__ import annotations

import pytest
from pymadng_utils.accelerators import PSB
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
from tmom_recon.acd.madng_driver import ACDipoleMadDriver

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.slow]


def _machine(psb_model_dir):
    sequence = psb_model_dir / SEQ_FILE
    environment = create_xsuite_environment(
        sequence_file=sequence,
        kinetic_energy=KINETIC_ENERGY_GEV,
        seq_name=SEQ_NAME,
        json_file=psb_model_dir / f"{sequence.stem}.json",
    )
    line = environment[SEQ_NAME]
    accelerator = PSB(sequence_file=sequence, ring=RING, kinetic_energy=KINETIC_ENERGY_GEV)
    model = ACDipoleMadDriver(
        accelerator=accelerator,
        pt=0.0,
        observed_elements=ACD_ELEMENT,
    )
    return line, model


def test_generated_model_has_expected_bpm_set_and_acd_anchor(psb_model_dir) -> None:
    line, model = _machine(psb_model_dir)
    model_twiss = model.run_twiss(observe=1, coupling=True)
    tracked_names = {str(name).upper() for name in line.element_names if "BPM" in str(name).upper()}
    bpm_names = [
        str(name)
        for name in model_twiss.index
        if "BPM" in str(name).upper() and str(name).upper() in tracked_names
    ]

    assert len(bpm_names) >= 8
    assert ACD_ELEMENT.upper() in {str(name).upper() for name in model.twiss_elements.index}

    line_names = [str(name).upper() for name in line.element_names]
    assert any(name == ACD_ELEMENT for name in line_names)
    bpm_indices = [index for index, name in enumerate(line_names) if f"BR{RING}.BPM" in name]
    assert len(bpm_indices) > 4
    acd_index = line_names.index(ACD_ELEMENT)
    assert min(bpm_indices) < acd_index < max(bpm_indices)

    twiss = line.twiss(method="4d")
    assert twiss.qx == pytest.approx(NATURAL_TUNES[0], abs=2e-2)
    assert twiss.qy == pytest.approx(NATURAL_TUNES[1], abs=2e-2)
    bpm_names = [line.element_names[index] for index in bpm_indices]
    assert float(abs(twiss.rows[bpm_names].x).max()) < 1e-10
    assert float(abs(twiss.rows[bpm_names].dx).max()) > 1e-2

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


def test_generated_model_nominal_orbit_is_closed(psb_model_dir) -> None:
    _, model = _machine(psb_model_dir)
    twiss = model.run_twiss(observe=1, coupling=True)
    assert float(twiss["x"].abs().max()) < 1e-5
    assert float(twiss["y"].abs().max()) < 1e-5
