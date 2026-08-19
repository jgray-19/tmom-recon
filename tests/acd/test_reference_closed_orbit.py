"""Tests for the mandatory generated ACD closed-orbit API."""

import pytest

from tmom_recon import ACDipoleConfig

__test__ = False


def test_acd_config_rejects_dispersive_closed_orbit_mode() -> None:
    with pytest.raises(TypeError, match="dispersive_closed_orbit"):
        ACDipoleConfig(
            ac_dipole_marker="acd",
            driven_tunes=(0.2, 0.3),
            dispersive_closed_orbit=True,
        )


def test_acd_config_rejects_reference_closed_orbit_mode() -> None:
    with pytest.raises(TypeError, match="use_reference_closed_orbit"):
        ACDipoleConfig(
            ac_dipole_marker="acd",
            driven_tunes=(0.2, 0.3),
            use_reference_closed_orbit=True,
        )
