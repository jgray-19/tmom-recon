from __future__ import annotations

from math import isclose
from pathlib import Path

import pytest
from pymadng_utils.accelerators import LHC, PSB


def test_lhc_exposes_expected_sequence_fields() -> None:
    accelerator = LHC(
        beam=2,
        sequence_file="machine.seq",
        pc=7000,
        bpm_pattern="^BPM.*$",
    )

    assert accelerator.beam == 2
    assert accelerator.sequence_file == Path("machine.seq")
    assert accelerator.seq_name == "lhcb2"
    assert accelerator.pc == 7000.0
    assert accelerator.bpm_pattern == "^BPM.*$"


def test_psb_derives_pc_and_sequence_name() -> None:
    accelerator = PSB(
        ring=3,
        sequence_file="psb.seq",
        pc=0.2,
    )

    assert accelerator.ring == 3
    assert accelerator.sequence_file == Path("psb.seq")
    assert accelerator.seq_name == "psb3"
    assert isclose(accelerator.pc, 0.2)
    assert accelerator.bpm_pattern == "^BR3%.BPM"


def test_lhc_rejects_invalid_beam() -> None:
    with pytest.raises(ValueError, match="LHC beam"):
        LHC(beam=3, sequence_file="machine.seq")


def test_psb_rejects_invalid_ring() -> None:
    with pytest.raises(ValueError, match="PSB ring"):
        PSB(ring=5, sequence_file="psb.seq")
