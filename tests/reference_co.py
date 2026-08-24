"""Nominal-RF reference closed orbits for tests.

``calculate_pz`` requires a *measured* closed orbit at nominal RF as the momentum
reference. On a real machine it cannot be replaced by a model orbit: the bend
response matrix spans the entire horizontal BPM space, so an unknown
dipole-error closed orbit is exactly degenerate with the dispersive orbit and
leaks straight into pt.

Simulated tests are the one case where a substitute is legitimate, because the
machine's errors are known by construction. Use the narrowest helper that is
true of the setup under test rather than defaulting to zero everywhere.
"""

from __future__ import annotations

import pandas as pd

from tmom_recon import ReconstructionFrame


def measured_zero_reference_for_simulation(
    data: pd.DataFrame, *, pt: float = 0.0
) -> ReconstructionFrame:
    """Measured zero orbit for a simulated machine without orbit errors."""
    names = pd.Index(pd.unique(data["name"]), name="name")
    origin = pd.DataFrame({"x": 0.0, "y": 0.0}, index=names)
    momenta = pd.DataFrame({"px": 0.0, "py": 0.0}, index=names)
    return ReconstructionFrame(origin, fitted_momenta=momenta)


def position_only_reference_from_twiss(
    tws: pd.DataFrame, *, pt: float = 0.0
) -> ReconstructionFrame:
    """Position-only reference from a Twiss table matching the simulated machine.

    Only valid when the model provably carries the machine's errors -- i.e. the
    test applied them to both sides.
    """
    return ReconstructionFrame(
        pd.DataFrame({"x": tws["x"].astype(float), "y": tws["y"].astype(float)}),
        fitted_momenta=tws[["px", "py"]],
    )


def full_state_reference_from_twiss(tws: pd.DataFrame, *, pt: float = 0.0) -> ReconstructionFrame:
    """Full transverse-state reference from a matching Twiss table."""
    return ReconstructionFrame(
        tws[["x", "y"]].astype(float).copy(),
        fitted_momenta=tws[["px", "py"]].astype(float).copy(),
    )
