"""Single-kick reconstruction from the first beam pass after a localised kicker.

A single kicker fires between two turns.  Before it fires the beam sits on the
closed orbit; immediately after it the state at the kicker is, in
closed-orbit-subtracted coordinates,

.. math::

   (x, p_x, y, p_y)_\\text{kicker} = (0, \\Delta p_x, 0, \\Delta p_y).

Every BPM the beam reaches during the following turn therefore reads

.. math::

   \\begin{pmatrix} x_i \\\\ y_i \\end{pmatrix}
   = \\mathbf{R}_{k \\to i}
     \\begin{pmatrix} \\Delta p_x \\\\ \\Delta p_y \\end{pmatrix},

with :math:`\\mathbf{R}_{k \\to i}` the coupled kick response of
:func:`~tmom_recon.lattice.transport.kick_response_from_twiss`.  Only the first
BPM after the kicker is used: its two positions are exactly the two equations
for ``(delta_px, delta_py)``, and the short transport to it is the least exposed
to lattice errors between the model and the machine.

The optics must be those *at the kick point*.  MAD-NG puts a Twiss row at each
element's exit, while a thick element deflects the beam at its centre, so a
thick kicker has to be represented in the model by a marker at its centre --
which is what :meth:`Accelerator.kicker_marker_name` is the seam for.  Getting
this wrong is a pure phase error at the source: a 1.23 m PSB quadrupole shifts
it by 0.017 turns and leaves a 6% residual in
:attr:`Kick.rms_residual`, which is the number to look at when a fit disagrees
with expectation.

Unlike the all-BPM and AC-dipole workflows, this one is deliberately
**model-optics only**.  The transport above needs the phase advance *from the
kicker*, and an omc3 measurement only ever provides BPM-to-BPM phases with an
arbitrary origin -- there is no measured kicker phase to chain onto.  A better
lattice is supplied the same way as everywhere else, through
:attr:`ModelDetails.magnet_strengths`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tmom_recon.data.checks import validate_input
from tmom_recon.data.config import FILE_COLUMNS
from tmom_recon.lattice.transport import kick_response_from_twiss
from tmom_recon.model import ModelDetails, ResolvedModel, resolve_model_details

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Mapping

    from tmom_recon.acd.madng_driver import ACDipoleMadDriver
    from tmom_recon.orbit_reference import OrbitReference

#: A turn is flagged as the kick turn once the orbit deviation exceeds both the
#: pre-kick spread by this factor and this fraction of the largest post-kick
#: excursion.  A real kick is orders of magnitude above both.
_KICK_SIGMA = 5.0
_KICK_PEAK_FRACTION = 0.01


@dataclass(frozen=True)
class KickerConfig:
    """Known kicker identity and the length of the undisturbed pre-kick window.

    Attributes:
        kicker: Element name of the kicker in the accelerator sequence.
        n_turns_free: Number of leading turns known to be free of the kick.
            They define the undisturbed reference the kick is detected against;
            the kick is searched for from this turn onwards.
    """

    kicker: str
    n_turns_free: int = 1000


def find_kick(data: pd.DataFrame, n_turns_free: int = 1000) -> tuple[str, int]:
    """Find the first turn carrying a clear kick, and the BPM that shows it most.

    The deviation of every reading from its own pre-kick mean is compared
    against two scales that both come from the data itself: the pre-kick spread
    (noise) and the largest post-kick excursion (the kick).  No absolute
    position floor is needed, so the detection works equally for noiseless
    tracking output and for noisy measurements.

    Args:
        data: Turn-by-turn BPM data with ``name``, ``turn``, ``x`` and ``y``.
        n_turns_free: Number of leading turns known to be kick-free.

    Returns:
        ``(kick_bpm, kick_turn)``.

    Raises:
        ValueError: If no turn stands out from the pre-kick window.
    """
    names = data["name"].to_numpy(str)
    turns = data["turn"].to_numpy(int)
    free = turns < n_turns_free
    if not free.any():
        raise ValueError(
            f"No turns below n_turns_free={n_turns_free} to reference the kick against"
        )
    if free.all():
        raise ValueError(f"No turns at or beyond n_turns_free={n_turns_free} to search for a kick")

    positions = data[["x", "y"]].to_numpy(float)
    centre = pd.DataFrame(positions[free], index=names[free]).groupby(level=0).mean()
    deviation = np.abs(positions - centre.reindex(names).to_numpy()).max(axis=1)

    noise = float(deviation[free].max())
    peak = float(deviation[~free].max())
    threshold = max(_KICK_SIGMA * noise, _KICK_PEAK_FRACTION * peak)
    if peak <= threshold:
        raise ValueError(
            f"No kick found: the largest excursion after turn {n_turns_free} is {peak:.3e}, "
            f"which does not stand out from the pre-kick spread {noise:.3e}"
        )

    kicked = ~free & (deviation > threshold)
    kick_turn = int(turns[kicked].min())
    on_kick_turn = kicked & (turns == kick_turn)
    kick_bpm = str(names[on_kick_turn][np.argmax(deviation[on_kick_turn])])
    return kick_bpm, kick_turn


def first_pass(
    data: pd.DataFrame, twiss: pd.DataFrame, kicker: str, kick_turn: int
) -> pd.DataFrame:
    """Select the BPM readings of the first beam pass after the kick.

    The kicker fires part-way round the ring, so the first pass is split across
    two recorded turns: the BPMs downstream of the kicker read the kicked beam
    already on *kick_turn*, while those upstream of it are only reached on the
    following turn.  Together they form exactly one turn of beam, each with a
    phase advance from the kicker of less than one full tune.

    Args:
        data: Turn-by-turn BPM data with ``name`` and ``turn`` columns.
        twiss: Twiss table indexed by element name, containing ``s`` and the
            kicker row.
        kicker: Index label of the kicker in *twiss*.
        kick_turn: Turn on which the kick fired.

    Returns:
        The subset of *data* forming the first pass.
    """
    kicker_s = float(twiss.at[kicker, "s"])
    downstream = data["name"].map(twiss["s"]).to_numpy(float) > kicker_s
    turns = data["turn"].to_numpy(int)
    rows = data[((turns == kick_turn) & downstream) | ((turns == kick_turn + 1) & ~downstream)]
    if rows.empty:
        raise ValueError(f"No BPM readings of the first pass after the kick at turn {kick_turn}")
    return rows


def first_reading(rows: pd.DataFrame, twiss: pd.DataFrame, kicker: str) -> pd.DataFrame:
    """The one reading of the BPM the beam reaches first after the kicker.

    Distance is measured downstream along the ring, wrapping at the
    circumference, so a kicker after the last BPM picks the first BPM of the
    next turn.
    """
    s = rows["name"].map(twiss["s"]).to_numpy(float)
    kicker_s = float(twiss.at[kicker, "s"])
    length = float(twiss["s"].max())
    distance = np.where(s > kicker_s, s - kicker_s, s - kicker_s + length)
    return rows.iloc[[int(np.argmin(distance))]]


@dataclass(frozen=True)
class Kick:
    """A reconstructed instantaneous kick, in closed-orbit-subtracted coordinates.

    Attributes:
        delta_px: Reconstructed horizontal kick [rad].
        delta_py: Reconstructed vertical kick [rad].
        var_px: Variance of *delta_px* [rad^2].
        var_py: Variance of *delta_py* [rad^2].
        bpm: The BPM the kick was solved from.
    """

    delta_px: float
    delta_py: float
    var_px: float
    var_py: float
    bpm: str


def solve_kick(reading: pd.DataFrame, twiss: pd.DataFrame, kicker: str) -> Kick:
    """Solve one kick exactly from a single closed-orbit-subtracted BPM reading.

    .. math::

       \\begin{pmatrix} x \\\\ y \\end{pmatrix} = R \\begin{pmatrix} \\Delta p_x \\\\ \\Delta p_y \\end{pmatrix}

    with :math:`R` the coupled kick response to that BPM.  The covariance is
    :math:`R^{-1} \\operatorname{diag}(\\sigma_x^2, \\sigma_y^2) R^{-T}`.

    Args:
        reading: One row with ``name``, ``x``, ``y`` and, optionally,
            ``var_x``/``var_y`` (see :func:`first_reading`).
        twiss: Coupled Twiss table indexed by element name with the kicker row
            and the tune headers ``q1``/``q2``.
        kicker: Index label of the kicker in *twiss*.
    """
    if len(reading) != 1:
        raise ValueError(f"The kick is solved from one BPM reading, got {len(reading)}")
    row = reading.iloc[0]
    tunes = (float(twiss.headers["q1"]), float(twiss.headers["q2"]))
    response = kick_response_from_twiss(twiss, source=kicker, target=row["name"], tunes=tunes)
    inverse = np.linalg.inv(response)
    solution = inverse @ np.array([row["x"], row["y"]], dtype=float)
    variance = np.array([_variance(reading, "var_x")[0], _variance(reading, "var_y")[0]])
    covariance = inverse @ np.diag(variance) @ inverse.T
    return Kick(
        delta_px=float(solution[0]),
        delta_py=float(solution[1]),
        var_px=float(covariance[0, 0]),
        var_py=float(covariance[1, 1]),
        bpm=str(row["name"]),
    )


def _variance(rows: pd.DataFrame, column: str) -> np.ndarray:
    """Measurement variances, defaulting to equal weights when absent."""
    if column not in rows.columns:
        return np.ones(len(rows))
    values = rows[column].to_numpy(float)
    return np.where(np.isfinite(values) & (values > 0.0), values, 1.0)


#: The four phase-space columns a closed orbit is described by.
_STATE = ("x", "px", "y", "py")


def model_closed_orbit(model: ACDipoleMadDriver, tws: pd.DataFrame) -> pd.DataFrame:
    """The closed-orbit shift the beam's momentum makes, taken exactly.

    The frame removes a measured **setting-zero** orbit, which is recorded on
    momentum.  What is left in the data before the kick is the machine's closed
    orbit at the beam's momentum minus its on-momentum orbit -- in angle as well
    as position -- so it has to come off the BPM readings before the kick is
    solved for, and back onto the kicker state afterwards.

    That difference is taken from two MAD-NG twisses rather than from a Taylor
    expansion ``pt*D + pt^2*D2`` in the dispersion columns.  The reconstruction
    twiss is generated *at* ``pt``, so its dispersion columns are derivatives
    **about** ``pt``; re-expanding from them double-counts the second order.
    Measured against PSB tracking at ``dp/p = 1e-3``, the exact difference
    reproduces the tracked orbit to 1.6e-10 m where the expansion leaves
    6.1e-7 m -- which is 6% of a 1e-5 rad kick once it is fitted.

    Args:
        model: The driver that produced *tws*, used to twiss again at zero.
        tws: That driver's twiss at the beam's momentum.

    Returns:
        A frame indexed by upper-cased element name with columns ``x``, ``px``,
        ``y``, ``py``.  Exactly zero on momentum, where there is nothing for the
        setting-zero orbit to miss.
    """
    tws = tws.copy()
    tws.index = tws.index.astype(str).str.upper()
    orbit = tws[list(_STATE)].astype(float)
    if model.pt == 0.0:
        return orbit * 0.0
    at_zero = model.run_twiss(observe=1, coupling=True, chrom=True, deltap=0.0)
    at_zero.index = at_zero.index.astype(str).str.upper()
    return orbit - at_zero.loc[tws.index, list(_STATE)].astype(float).to_numpy()


def reconstruct_kick(
    data: pd.DataFrame,
    *,
    reference: OrbitReference,
    twiss: pd.DataFrame,
    kicker: str,
    closed_orbit: pd.DataFrame | None = None,
    n_turns_free: int = 1000,
) -> pd.DataFrame:
    """Reconstruct the kicker state from turn-by-turn BPM data.

    Args:
        data: Turn-by-turn BPM data with ``name``, ``turn``, ``x``, ``y`` and
            optionally ``var_x``/``var_y``.
        reference: Coordinate frame whose measured orbit zero is removed before the
            reconstruction.
        twiss: Model twiss indexed by element name, containing the kicker row,
            the optics columns and the tune headers ``q1``/``q2``.
        kicker: Index label of the kicker in *twiss*.
        closed_orbit: The orbit the beam rides before the kick, in frame
            coordinates, indexed by element name with ``x``/``px``/``y``/``py``
            (see :func:`model_closed_orbit`).  ``None`` means the beam is
            already at the frame's origin, which is the on-momentum case.
        n_turns_free: Number of leading kick-free turns (see :func:`find_kick`).

    Returns:
        A one-row frame with the standard reconstruction columns holding the
        kicker state on the kick turn, in **frame** coordinates: the closed
        orbit at the kicker -- the model's, restored by the frame -- plus the
        reconstructed kick.  Nothing measures the beam at the
        kicker, so ``var_x``/``var_y`` are infinite.  ``attrs["kick"]`` carries
        the underlying :class:`Kick`.
    """
    validate_input(data)
    twiss = twiss.copy()
    twiss.index = twiss.index.astype(str).str.upper()
    kicker = kicker.upper()
    if kicker not in twiss.index:
        raise ValueError(
            f"The twiss has no row for the kicker {kicker!r}; the model must observe it"
        )

    # Tracking tables routinely carry markers and monitors that are not model
    # BPMs; only the elements the model knows can contribute to the transport.
    data = data.copy()
    data["name"] = data["name"].astype(str).str.upper()
    known = data["name"].isin(twiss.index)
    if not known.any():
        raise ValueError("None of the measured elements appear in the model twiss")
    if not known.all():
        LOGGER.info(
            "Ignoring elements absent from the model twiss: %s",
            sorted(set(data.loc[~known, "name"])),
        )
    data = reference.subtract(data[known]).reset_index(drop=True)

    if closed_orbit is None:
        closed_orbit = pd.DataFrame(0.0, index=twiss.index, columns=list(_STATE))
    data[["x", "y"]] -= closed_orbit.loc[data["name"], ["x", "y"]].to_numpy()

    _kick_bpm, kick_turn = find_kick(data, n_turns_free)
    # The beam's own pre-kick orbit removes every closed-orbit error, so the
    # kick does not depend on the model orbit.
    pre_kick = data[data["turn"] < n_turns_free].groupby("name")[["x", "y"]].mean()
    data[["x", "y"]] -= pre_kick.reindex(data["name"]).to_numpy()
    reading = first_reading(first_pass(data, twiss, kicker, kick_turn), twiss, kicker)
    kick = solve_kick(reading, twiss, kicker)
    at_kicker = closed_orbit.loc[kicker] + reference.restored.loc[kicker]

    result = pd.DataFrame(
        [
            {
                "name": kicker,
                "turn": kick_turn,
                "x": float(at_kicker["x"]),
                "px": float(at_kicker["px"]) + kick.delta_px,
                "y": float(at_kicker["y"]),
                "py": float(at_kicker["py"]) + kick.delta_py,
                "var_x": np.inf,
                "var_y": np.inf,
                "var_px": kick.var_px,
                "var_py": kick.var_py,
            }
        ],
        columns=list(FILE_COLUMNS),
    )
    result.attrs["kick"] = kick
    return result


def _kicker_name(model_details: ModelDetails, config: KickerConfig) -> str:
    """The kicker's element name, upper-cased to match the MAD-NG sequence."""
    return model_details.accelerator.kicker_marker_name(config.kicker).upper()


def calculate_kicker_pz(
    data: pd.DataFrame,
    model_details: ModelDetails,
    config: KickerConfig,
    *,
    closed_orbit_at_zero: pd.DataFrame,
    orbit_mode: str,
) -> pd.DataFrame:
    """Reconstruct one localised kick from generated model optics.

    See :func:`reconstruct_kick` for the returned frame.
    """
    kicker = _kicker_name(model_details, config)
    model = resolve_model_details(model_details, observed_elements=kicker)
    from tmom_recon.orbit_reference import build_orbit_reference

    reference = build_orbit_reference(
        closed_orbit_at_zero,
        orbit_mode,
        model.zero_tws,
        [str(name) for name in data["name"].unique()],
    )
    return reconstruct_kick(
        data,
        reference=reference,
        twiss=model.tws,
        kicker=kicker,
        closed_orbit=model_closed_orbit(model.model, model.tws),
        n_turns_free=config.n_turns_free,
    )


class KickerPzGenerator:
    """Repeated kick reconstruction for fixed BPM data and frame.

    The MAD-NG model is generated once and kept, so :meth:`update` only pays
    for a new twiss and the small least-squares solve.
    """

    def __init__(
        self,
        *,
        data: pd.DataFrame,
        resolved_model: ResolvedModel,
        kicker: str,
        closed_orbit_at_zero: pd.DataFrame,
        orbit_mode: str,
        n_turns_free: int,
    ) -> None:
        self._data = data.copy(deep=True)
        self._resolved_model = resolved_model
        self._kicker = kicker
        self._closed_orbit_at_zero = closed_orbit_at_zero.copy(deep=True)
        self._orbit_mode = orbit_mode
        self._zero_tws = resolved_model.zero_tws
        self._n_turns_free = n_turns_free
        self._tws = resolved_model.tws
        self._closed_orbit = model_closed_orbit(resolved_model.model, self._tws)
        self.latest: pd.DataFrame | None = None

    @classmethod
    def build(
        cls,
        *,
        data: pd.DataFrame,
        model_details: ModelDetails,
        config: KickerConfig,
        closed_orbit_at_zero: pd.DataFrame,
        orbit_mode: str,
    ) -> KickerPzGenerator:
        validate_input(data)
        kicker = _kicker_name(model_details, config)
        return cls(
            data=data,
            resolved_model=resolve_model_details(model_details, observed_elements=kicker),
            kicker=kicker,
            closed_orbit_at_zero=closed_orbit_at_zero,
            orbit_mode=orbit_mode,
            n_turns_free=config.n_turns_free,
        )

    @property
    def model(self):
        """The MAD-NG driver used to generate the optics (mutate magnets here)."""
        return self._resolved_model.model

    def update(
        self,
        *,
        magnet_strengths: Mapping[str, float] | None = None,
        pt: float | None = None,
    ) -> pd.DataFrame:
        """Recompute the kick, optionally from a mutated model."""
        if pt is not None:
            self.model.pt = float(pt)
        if magnet_strengths is not None:
            self.model.apply_strengths(magnet_strengths)
        if magnet_strengths is not None:
            self._zero_tws = self.model.run_twiss(observe=1, coupling=True, chrom=True, deltap=0.0)
        if magnet_strengths is not None or pt is not None:
            self._tws = self.model.run_twiss(observe=1, coupling=True, chrom=True, pt=self.model.pt)
            self._closed_orbit = model_closed_orbit(self.model, self._tws)
        from tmom_recon.orbit_reference import build_orbit_reference

        reference = build_orbit_reference(
            self._closed_orbit_at_zero,
            self._orbit_mode,
            self._zero_tws,
            [str(name) for name in self._data["name"].unique()],
        )
        self.latest = reconstruct_kick(
            self._data,
            reference=reference,
            twiss=self._tws,
            kicker=self._kicker,
            closed_orbit=self._closed_orbit,
            n_turns_free=self._n_turns_free,
        )
        return self.latest


__all__ = [
    "Kick",
    "KickerConfig",
    "KickerPzGenerator",
    "calculate_kicker_pz",
    "find_kick",
    "first_pass",
    "first_reading",
    "model_closed_orbit",
    "reconstruct_kick",
    "solve_kick",
]
