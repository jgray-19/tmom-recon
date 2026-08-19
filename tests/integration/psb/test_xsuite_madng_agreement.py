"""Explicit Xsuite-to-MAD-NG compatibility contracts for PSB."""

import pytest

from tests.momentum.test_dispersive_measurement import test_psb_xsuite_madng_optics_agreement

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.crosscode]

__all__ = ["test_psb_xsuite_madng_optics_agreement"]
