"""Regression for matched distorted PSB closed-orbit reconstruction."""

import pytest

from tests.acd.test_psb_closed_orbit_acd import *  # noqa: F401,F403

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.regression]
