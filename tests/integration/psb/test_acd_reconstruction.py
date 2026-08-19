"""PSB AC-dipole reconstruction integration contracts."""

import pytest

from tests.acd.test_psb_acd_momentum import *  # noqa: F401,F403
from tests.acd.test_psb_closed_orbit_acd import *  # noqa: F401,F403

pytestmark = [pytest.mark.psb, pytest.mark.integration]
