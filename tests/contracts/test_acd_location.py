"""Contract that the tracking and reconstruction models agree on the ACD anchor."""

from __future__ import annotations

import numpy as np
import pytest

from tests.contracts.conftest import scenario_params
from tests.psb_tracking import ACD_ELEMENT as PSB_ACD_MARKER
from tests.support.lhc import AC_DIPOLE_MARKER

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]


@pytest.mark.parametrize("contract_scenario", scenario_params(0.0), indirect=True)
def test_xtrack_and_madng_agree_on_ac_dipole_position(contract_scenario) -> None:
    """Failure means the barrier would protect a different location than the kick."""
    marker = PSB_ACD_MARKER if contract_scenario.machine == "psb" else AC_DIPOLE_MARKER
    # This is an installation contract, not an optics contract. `Line.get_table`
    # records the actual placement requested from Xtrack; `TwissTable.s` is
    # reconstructed by longitudinal accumulation during Twiss and is not used as
    # an optics authority in this project.
    tracking_table = contract_scenario.tracking_line.get_table()
    if contract_scenario.machine == "psb":
        installed_names = [marker]
    else:
        # The Xtrack installer replaces the LHC thick element with two thin
        # kickers. Their names are derived from the source element; neither is
        # called exactly ``MKQA.6L4.B1`` after slicing.
        installed_names = [f"{marker}_x", f"{marker}_y"]

    installed_s = []
    for installed_name in installed_names:
        tracking_name = next(
            (
                name
                for name in contract_scenario.tracking_line.element_names
                if str(name).upper() == installed_name.upper()
            ),
            None,
        )
        candidates = [
            name
            for name in contract_scenario.tracking_line.element_names
            if marker.lower() in str(name).lower()
        ]
        assert tracking_name is not None, (
            f"Xtrack line has no installed ACD component {installed_name}; "
            f"matching elements: {candidates}"
        )
        position = np.asarray(tracking_table.rows[tracking_name].s_center, dtype=float).ravel()
        assert position.size == 1
        installed_s.append(float(position[0]))

    assert np.allclose(installed_s, installed_s[0], atol=1e-12, rtol=0.0)
    assert installed_s[0] == pytest.approx(contract_scenario.barrier_s, abs=1e-12)
