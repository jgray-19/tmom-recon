"""Generate MAD-NG model optics from an accelerator description.

The user never supplies a twiss. They describe the *accelerator*, its **momentum**
(``pt``) and any **additional magnet strengths** missing from the base sequence;
:func:`resolve_model_details` builds the MAD-NG model and returns the off-momentum
optics twiss. Tunes are never matched here: the lattice the caller describes is
taken to already sit on the machine's tunes at ``pt``. The AC-dipole layer
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


@dataclass(frozen=True)
class ModelDetails:
    """Accelerator description from which the model optics are generated.

    Attributes:
        accelerator: Owns sequence loading, beam parameters and BPM patterns.
        pt: MAD-NG longitudinal energy coordinate for the tracked beam.
        magnet_strengths: Additional magnet strengths missing from the base
            sequence (see :meth:`ACDipoleMadDriver.apply_strengths`). Nothing
            here rematches tunes: the supplied lattice is taken to already sit on
            the machine's tunes at *pt*.

            This is also the seam for a *fitted* lattice. The closed-orbit angle
            ``px``/``py`` cannot be measured, so it comes from
            ``closed_orbit_tws``; with a nominal model that angle is simply
            wrong, and on PSB ring 3 its error equals the entire true angle.
            Passing magnet strengths fitted to a measured closed orbit and phase
            (``aba_optimiser.momentum_reference`` in the sgd-magnet-tuner
            project does this) cut that error ~20x in simulation. The coupling
            is deliberately plain data -- a mapping of strengths -- so this
            package never imports the fitting code.
        tune_knobs: Optional tune-correction knobs, as name/value pairs or as a
            knobs file to read.
        corrector_knobs: Optional orbit-corrector settings, as name/value pairs,
            a knobs file, or a TFS corrector table.
    """

    accelerator: Accelerator
    pt: float = 0.0
    magnet_strengths: Mapping[str, float] | None = None
    tune_knobs: Mapping[str, float] | Path | None = None
    corrector_knobs: Mapping[str, float] | Path | None = None


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
        tune_knobs=details.tune_knobs,
        corrector_knobs=details.corrector_knobs,
    )
    # `chrom=True` adds the second-order dispersion columns ddx/ddpx/ddy/ddpy,
    # which the pt estimate and the dispersive momentum term both use. They are
    # optional downstream, so a twiss without them still works -- just to first
    # order in pt.
    closed_orbit_tws = model.run_twiss(observe=1, coupling=True, chrom=True, deltap=0.0)
    optics_tws = model.run_twiss(observe=1, coupling=True, chrom=True, pt=model.pt)
    return ResolvedModel(model=model, optics_tws=optics_tws, closed_orbit_tws=closed_orbit_tws)


__all__ = [
    "ModelDetails",
    "ResolvedModel",
    "resolve_model_details",
]
