"""Generated PSB model infrastructure contracts."""

import pytest

from tests.regression.psb.test_model_contract import *  # noqa: F401,F403

pytestmark = [pytest.mark.psb, pytest.mark.integration]
