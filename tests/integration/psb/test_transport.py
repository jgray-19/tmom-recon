"""PSB MAD-NG transport integration contracts."""

import pytest

from tests.acd.test_psb_transport_backtracking import *  # noqa: F401,F403

pytestmark = [pytest.mark.psb, pytest.mark.integration]
