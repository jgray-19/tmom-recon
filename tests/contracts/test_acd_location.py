"""Contract that the tracking and reconstruction models agree on the ACD anchor."""

from __future__ import annotations

import numpy as np
import pytest

from tests.psb_tracking import ACD_ELEMENT as PSB_ACD_MARKER
from tests.psb_tracking import build_psb_tracking_setup
from tests.support.acd_barrier import acd_barrier_s
from tests.support.lhc import AC_DIPOLE_MARKER, lhc_acd_barrier_s, lhc_model_details

pytestmark = [pytest.mark.diagnostic, pytest.mark.integration, pytest.mark.slow]


def _assert_installed_position(
    line, marker: str, expected_s: float, installed_names: list[str]
) -> None:
    """Failure means the barrier would protect a different location than the kick."""
    # This is an installation contract, not an optics contract. `Line.get_table`
    # records the actual placement requested from Xtrack; `TwissTable.s` is
    # reconstructed by longitudinal accumulation during Twiss and is not used as
    # an optics authority in this project.
    tracking_table = line.get_table()

    installed_s = []
    for installed_name in installed_names:
        tracking_name = next(
            (name for name in line.element_names if str(name).upper() == installed_name.upper()),
            None,
        )
        candidates = [name for name in line.element_names if marker.lower() in str(name).lower()]
        assert tracking_name is not None, (
            f"Xtrack line has no installed ACD component {installed_name}; "
            f"matching elements: {candidates}"
        )
        position = np.asarray(tracking_table.rows[tracking_name].s_center, dtype=float).ravel()
        assert position.size == 1
        installed_s.append(float(position[0]))

    assert np.allclose(installed_s, installed_s[0], atol=1e-12, rtol=0.0)
    assert installed_s[0] == pytest.approx(expected_s, abs=1e-12)


def test_psb_xtrack_and_madng_agree_on_ac_dipole_position(psb_model_dir) -> None:
    """The short-lived tracking line never enters a reconstruction contract."""
    setup = build_psb_tracking_setup(psb_model_dir, 0.0)
    # Recreate only for this installation check; the scenario itself deliberately
    # does not retain the hundreds-of-MiB line.
    from xtrack_tools.env import create_xsuite_environment

    env = create_xsuite_environment(
        sequence_file=setup.machine.accelerator.sequence_file,
        kinetic_energy=0.160,
        seq_name="psb3",
        json_file=psb_model_dir / "psb3_saved.json",
    )
    line = env["psb3"]
    expected = acd_barrier_s(setup.machine.madng_model, PSB_ACD_MARKER)
    _assert_installed_position(line, PSB_ACD_MARKER, expected, [PSB_ACD_MARKER])


@pytest.mark.parametrize("machine", ["lhcb1", "b1_120cm_crossing"])
def test_lhc_xtrack_and_madng_agree_on_ac_dipole_position(
    machine, data_dir, acd_tracking_setup
) -> None:
    sequence = data_dir / "sequences" / f"{machine}.seq"
    setup = acd_tracking_setup(sequence, include_line=True)
    details = lhc_model_details(sequence)
    expected = lhc_acd_barrier_s(details.accelerator, details.pt)
    _assert_installed_position(
        setup.baseline_line,
        AC_DIPOLE_MARKER,
        expected,
        [f"{AC_DIPOLE_MARKER}_x", f"{AC_DIPOLE_MARKER}_y"],
    )
