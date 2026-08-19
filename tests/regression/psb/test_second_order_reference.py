"""Regression for the PSB second-order dispersion reference path."""

import pytest

from tests.momentum.test_second_order_dispersion_pipeline import *  # noqa: F401,F403

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.regression]
