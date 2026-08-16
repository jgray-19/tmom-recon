"""The momentum origin a reconstruction is expressed against.

Why this type exists
--------------------
Every reconstruction in this package works in *deviations from a reference
closed orbit*: the orbit is subtracted from the data, the dispersive orbit is
removed with ``pt * D + pt**2 * D''``, and the reference angles are added back at
the end. The ``pt`` in those expressions is therefore the momentum **offset from
the reference orbit**, never the absolute MAD-NG ``pt`` of the measurement.

Nothing in a bare ``(DataFrame, float)`` signature says which of the two a caller
means, and the two agree in the two commonest cases -- a reference at nominal RF
(``pt = 0``) and a linear lattice, where the first-order terms cancel the
difference exactly. The whole penalty lands on the second-order dispersion term,
so the mistake is invisible until it is expensive. The off-momentum study
measured it on a reference sitting 3e-3 off the origin: passing the absolute
``pt`` degraded the reconstructed ``px`` from 4.741e-4 to 7.702e-2, while passing
the offset degraded it to 1.162e-3 -- a factor 66
(``tmom-recon-study/results/10_offmom_refmom.csv``).

:class:`MomentumReference` removes the choice. The caller states the absolute
momentum of both the reference orbit and the measurement, and this package does
the subtraction. There is no way to pass an offset where an absolute value is
wanted, because no entry point accepts an offset any more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    import pandas as pd

__all__ = ["MomentumReference"]


@dataclass(frozen=True)
class MomentumReference:
    """A reference closed orbit together with the momentum it sits at.

    Attributes:
        closed_orbit: **Measured** closed orbit indexed by BPM name, with an
            ``x`` column (``y`` where the vertical plane is used, and ``px``/
            ``py`` when the reference comes from a fit that knows the machine's
            real angles). A model closed orbit is not a substitute: the bend
            response spans the whole horizontal BPM space, so an unknown
            dipole-error orbit is exactly degenerate with the dispersive orbit
            and a mismatched model biases ``pt`` by tens of percent.
        pt: Absolute MAD-NG ``pt`` this orbit was taken at. ``0.0`` means
            nominal RF, which is what a plain measured orbit is. Set it when the
            reference was fitted or measured off-momentum -- e.g.
            ``aba_optimiser.momentum_reference.MomentumReference.reference_pt``
            in the sibling repository.
    """

    closed_orbit: pd.DataFrame
    pt: float = 0.0

    def __post_init__(self) -> None:
        if "x" not in getattr(self.closed_orbit, "columns", ()):
            raise ValueError('MomentumReference.closed_orbit needs an "x" column.')
        object.__setattr__(self, "pt", float(self.pt))

    def offset_from(self, measurement_pt: float) -> float:
        """The momentum offset of a measurement at absolute *measurement_pt*.

        This is the quantity every dispersion term in the reconstruction is
        expanded in; see the module docstring for what passing the absolute
        value instead costs.
        """
        return float(measurement_pt) - self.pt
