"""Generate the model twiss used by momentum reconstruction.

The user never supplies a twiss. They describe the *accelerator*, its **momentum**
(``pt``) and any **additional magnet strengths** missing from the base sequence;
:func:`resolve_model_details` builds the MAD-NG model and returns both the active
chromatic twiss and its generated zero-momentum reference. Callers supply only
measured zero-orbit positions. Tunes are never matched here.
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

            This is also the seam for a *fitted* lattice. BPMs cannot measure
            the reference angle ``px``/``py``; an external fitting workflow can
            use these strengths to derive fitted momenta for a reconstruction frame.
            The coupling is deliberately plain data -- a mapping of strengths
            -- so this package never imports or runs fitting code.
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
    """A generated model with active- and zero-momentum Twiss tables."""

    model: ACDipoleMadDriver
    tws: pd.DataFrame
    zero_tws: pd.DataFrame


def resolve_model_details(
    details: ModelDetails,
    *,
    observed_elements: str | list[str] | None = None,
    install_ac_dipole_markers: bool = False,
) -> ResolvedModel:
    """Generate a model and its active and zero-momentum reconstruction twisses.

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
    zero_tws = model.run_twiss(observe=1, coupling=True, chrom=True, deltap=0.0)
    tws = (
        zero_tws
        if model.pt == 0.0
        else model.run_twiss(observe=1, coupling=True, chrom=True, pt=model.pt)
    )
    return ResolvedModel(model=model, tws=tws, zero_tws=zero_tws)


__all__ = [
    "ModelDetails",
    "ResolvedModel",
    "resolve_model_details",
]
