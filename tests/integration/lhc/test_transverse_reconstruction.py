"""LHC transverse reconstruction integration contracts."""

import pytest

from tests.momentum.test_transverse_momentum import *  # noqa: F401,F403

pytestmark = [pytest.mark.lhc, pytest.mark.integration]
