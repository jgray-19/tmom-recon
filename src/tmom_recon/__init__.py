"""Momentum reconstruction utilities.

This package centralises the transverse momentum reconstruction
implementations and consolidates shared helpers.  The public API mirrors the
historical modules under ``tmom_recon.physics`` while exposing the
consolidated functions directly.
"""

from __future__ import annotations

from .acd.integration import ACDipoleConfig
from .acd.reconstruction import calculate_ac_dipole_momentum
from .measurements.dispersive_measurement import calculate_pz_measurement
from .measurements.twiss_from_measurement import build_twiss_from_measurements
from .nbpm import calculate_transverse_pz_nbpm
from .physics.dispersive import calculate_pz as calculate_dispersive_pz
from .physics.transverse import calculate_pz as calculate_transverse_pz
from .physics.transverse import inject_noise_xy_inplace

__all__ = [
    "ACDipoleConfig",
    "build_twiss_from_measurements",
    "calculate_ac_dipole_momentum",
    "calculate_dispersive_pz",
    "calculate_transverse_pz_nbpm",
    "calculate_transverse_pz",
    "calculate_pz_measurement",
    "inject_noise_xy_inplace",
]
