"""Generate MAD-NG model optics from an accelerator description.

The user never supplies a twiss. They describe the *accelerator*, its **tunes**,
its **momentum** (``pt``) and any **additional magnet strengths** missing from the
base sequence; :func:`resolve_model_details` builds the MAD-NG model, matches the
tunes and returns the off-momentum optics twiss. The AC-dipole layer
(:mod:`tmom_recon.acd.integration`) builds on this to add the undriven closed-orbit
reference and the driven optics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tmom_recon.acd.madng_driver import ACDipoleMadDriver

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import pandas as pd
    from pymadng_utils.accelerators import Accelerator

# Matches ``ModelCreatorMadInterface.TUNE_MATCH_TOLERANCE`` in pymadng-utils.
TUNE_MATCH_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ModelDetails:
    """Accelerator description from which the model optics are generated.

    Attributes:
        accelerator: Owns sequence loading, beam parameters and BPM patterns.
        tunes: Target fractional tunes ``(qx, qy)`` to match.
        onmom_tunes: Optional target fractional tunes for the on-momentum
            settings. When omitted, we assume the on and off momentum
            tunes are the same, so ``tunes`` is used for both.
        pt: MAD-NG longitudinal energy coordinate for the tracked beam.
        magnet_strengths: Additional magnet strengths missing from the base
            sequence (see :meth:`ACDipoleMadDriver.apply_strengths`).
        tune_knobs_file: Optional knob file applied for tune corrections.
        corrector_knobs_file: Optional knob file applied for corrector settings.
    """

    accelerator: Accelerator
    pt: float = 0.0
    magnet_strengths: Mapping[str, float] | None = None
    tune_knobs_file: Path | None = None
    corrector_knobs_file: Path | None = None


@dataclass(frozen=True)
class ResolvedModel:
    """A generated, tune-matched model and its separate twiss frames."""

    model: ACDipoleMadDriver
    optics_tws: pd.DataFrame
    closed_orbit_tws: pd.DataFrame


def resolve_model_details(
    details: ModelDetails,
    *,
    observed_elements: str | list[str] | None = None,
    install_ac_dipole_markers: bool = False,
) -> ResolvedModel:
    """Generate a tune-matched model and its off-momentum optics twiss.

    The AC-dipole before/after markers are only inserted when
    *install_ac_dipole_markers* is set, so a plain reconstruction is not coupled
    to the AC-dipole machinery.
    """
    model = ACDipoleMadDriver(
        accelerator=details.accelerator,
        pt=details.pt,
        observed_elements=observed_elements,
        magnet_strengths=details.magnet_strengths,
        install_ac_dipole_markers=install_ac_dipole_markers,
        tune_knobs_file=details.tune_knobs_file,
        corrector_knobs_file=details.corrector_knobs_file,
    )
    closed_orbit_tws = model.run_twiss(observe=1, coupling=True, deltap=0.0)
    optics_tws = model.run_twiss(observe=1, coupling=True, pt=model.pt)
    return ResolvedModel(model=model, optics_tws=optics_tws, closed_orbit_tws=closed_orbit_tws)


__all__ = [
    "TUNE_MATCH_TOLERANCE",
    "ModelDetails",
    "ResolvedModel",
    "resolve_model_details",
]
