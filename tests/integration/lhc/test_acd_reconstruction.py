"""LHC AC-dipole reconstruction integration contracts."""

import pytest

from tests.acd.test_ac_dipole_momentum import *  # noqa: F401,F403
from tests.acd.test_acd_generator import *  # noqa: F401,F403

pytestmark = [pytest.mark.lhc, pytest.mark.integration]
