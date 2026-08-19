"""LHC dispersive reconstruction integration contracts."""

import pytest

from tests.momentum.test_dispersive_momentum import *  # noqa: F401,F403

pytestmark = [pytest.mark.lhc, pytest.mark.integration]
