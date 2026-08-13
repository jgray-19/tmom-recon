"""Focused tests for explicit ACD closed-orbit reference selection."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tmom_recon import ACDipoleConfig
from tmom_recon.reconstruction import _acd_closed_orbit_reference


def _resolved(config: ACDipoleConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        tracking_tws=pd.DataFrame({"source": ["tracking"]}),
        closed_orbit_tws=pd.DataFrame({"source": ["model"]}),
    )


def test_acd_can_use_complete_explicit_reference_closed_orbit() -> None:
    config = ACDipoleConfig(
        ac_dipole_marker="acd",
        driven_tunes=(0.2, 0.3),
        use_reference_closed_orbit=True,
    )
    reference = pd.DataFrame({name: [0.0] for name in ("x", "px", "y", "py")})

    assert _acd_closed_orbit_reference(_resolved(config), reference) is reference


def test_acd_explicit_reference_requires_full_transverse_state() -> None:
    config = ACDipoleConfig(
        ac_dipole_marker="acd",
        driven_tunes=(0.2, 0.3),
        use_reference_closed_orbit=True,
    )

    with pytest.raises(ValueError, match="missing.*px"):
        _acd_closed_orbit_reference(_resolved(config), pd.DataFrame({"x": [0.0], "y": [0.0]}))


def test_reference_and_dispersive_closed_orbits_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        ACDipoleConfig(
            ac_dipole_marker="acd",
            driven_tunes=(0.2, 0.3),
            use_reference_closed_orbit=True,
            dispersive_closed_orbit=True,
        )
