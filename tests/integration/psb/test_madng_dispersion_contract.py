"""MAD-NG second-order dispersion convention contract."""

import pytest

from tests.acd.test_madng_second_order_dispersion import *  # noqa: F401,F403

pytestmark = [pytest.mark.psb, pytest.mark.integration, pytest.mark.crosscode]
