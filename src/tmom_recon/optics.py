"""Per-category optics-source resolution for momentum reconstruction.

Builds a single resolved twiss DataFrame from a model twiss and/or an omc3
optics measurement directory. Each optics category — ``phase`` (mu1/mu2 and
tunes), ``beta``, ``alpha`` and ``dispersion`` — is sourced independently,
defaulting to the measurement when one is available. Beta and alpha are
separate categories so a measured beta can be paired with a model alpha, which
is what beta from amplitude needs (see :data:`CATEGORIES`).

Dispersion, by contrast, is deliberately a *single* category covering ``dx``,
``dy``, ``dpx`` and ``dpy``, because the reconstruction removes the dispersive
displacement from position using ``D`` and restores the dispersive angle using
``D'``. Those are two halves of one cancellation, so taking them from different
sources breaks it, and the off-momentum study measured that mixed pair to be
worse than a consistent modelled one.

A BPM measures position, so a momentum scan yields ``Dx``/``Dy`` directly and
the angular half has to be solved from the measured ``D`` at each BPM and its
successor by :func:`tmom_recon.momentum_dispersion.derive_momentum_dispersion`.
That derivation is the *caller's* job and its result must be present as
``dpx``/``dpy`` in the measurement; requesting measured dispersion without them
is an error. ``resolve_optics`` cannot do it itself because it does not know
which lattice the position dispersion was measured in — deriving against the
model it happens to be given silently attributes to the measurement an optics it
was never taken in (on PSB Ring 3, deriving against the AC-dipole-driven twiss
instead of the free on-momentum one biased ``Dx'`` by 1.5%).

All dispersion is an expansion about the ``pt = 0`` closed orbit, the point a
momentum scan measures it at: ``x(pt) - x(0) = pt D + pt^2 D^(2)``. Model
dispersion (first and second order) therefore comes from the undriven
``dp/p = 0`` twiss (``zero_tws``), never from the optics twiss at ``pt``, and
measured ``D``/``D'`` are used as measured. Only the betatron optics (phase,
beta, alpha) are taken at ``pt``.

Where a lattice has been fitted to the measured closed orbit, prefer that fitted
model's dispersion (``model_optics=("dispersion", ...)`` against a twiss of the
fitted machine) over the measured dispersion: a fit to the closed orbit alone
already recovers ``Dx`` to ~1.4% of the nominal error and ``Dx'`` equally well
(0.0141 vs 0.0140), and adding ``dx``/``dy`` to the fit target reaches ~0.0058 on
both — one model supplying both halves of the pair consistently.

Every resolved twiss carries a full set of uncertainty columns: measured
errors where the measurement is used, and rough configurable uncertainties
(:class:`ModelOpticsErrors`) where the model is used, so error propagation
always produces meaningful ``var_px``/``var_py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import tfs

from tmom_recon.data.columns import (
    DISPERSION_RENAME_MAPPING,
    ERROR_DISPERSION_RENAME_MAPPING,
    ERROR_RENAME_MAPPING,
    MEASUREMENT_RENAME_MAPPING,
)
from tmom_recon.measurements.twiss_from_measurement import build_twiss_from_measurements

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from collections.abc import Collection

LOGGER = logging.getLogger(__name__)

OpticsCategory = Literal["phase", "beta", "alpha", "dispersion"]
OpticsSource = Literal["model", "measurement"]

CATEGORIES: tuple[OpticsCategory, ...] = ("phase", "beta", "alpha", "dispersion")

PHASE_COLUMNS = ("mu1", "mu2")
BETA_COLUMNS = ("betx", "bety")
ALPHA_COLUMNS = ("alfx", "alfy")
DISPERSION_COLUMNS = ("dx", "dy", "dpx", "dpy")
SECOND_ORDER_DISPERSION_COLUMNS = ("ddx", "ddpx", "ddy", "ddpy")

CATEGORY_COLUMNS: dict[OpticsCategory, tuple[str, ...]] = {
    "phase": PHASE_COLUMNS,
    "beta": BETA_COLUMNS,
    "alpha": ALPHA_COLUMNS,
    "dispersion": DISPERSION_COLUMNS,
}


@dataclass(frozen=True)
class OpticsInput:
    """Explicit optics provenance for a reconstruction.

    ``sources`` is deliberately a mapping rather than a fallback policy.  A
    category requested from a measurement is required to be present there;
    silently substituting model data changes the physics question a caller is
    asking.  Omitted categories default to the generated model.
    """

    measurement_dir: str | Path | None = None
    sources: dict[OpticsCategory, OpticsSource] = field(default_factory=dict)
    reverse_measurement_order: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.sources) - set(CATEGORIES)
        invalid = {
            name: source
            for name, source in self.sources.items()
            if source not in {"model", "measurement"}
        }
        if unknown or invalid:
            details = []
            if unknown:
                details.append(f"unknown categories {sorted(unknown)}")
            if invalid:
                details.append(f"invalid sources {invalid}")
            raise ValueError("Invalid OpticsInput: " + "; ".join(details))
        if (
            any(source == "measurement" for source in self.sources.values())
            and self.measurement_dir is None
        ):
            raise ValueError(
                "Measured optics were requested but OpticsInput.measurement_dir is missing"
            )

    def source_for(self, category: OpticsCategory) -> OpticsSource:
        return self.sources.get(category, "model")


@dataclass(frozen=True)
class ModelOpticsErrors:
    """Rough uncertainties assigned to model-sourced optics.

    Attributes:
        beta_rel: Relative beta uncertainty (e.g. expected beta-beating level).
        alpha_rel: Relative alpha uncertainty.  When > 0, ``alpha_abs`` is used
            only as an absolute floor so locations where alpha ≈ 0 are still
            assigned a non-zero uncertainty.
        alpha_abs: Absolute alpha uncertainty (floor when ``alpha_rel`` > 0).
        phase_abs: Phase-advance uncertainty per BPM-to-BPM step [turns];
            accumulated linearly along the ring like a measured phase chain.
        dispersion_rel: Relative dispersion uncertainty.
        dispersion_abs: Absolute dispersion uncertainty floor [m] (and [rad]
            for the momentum dispersion), used where the dispersion is ~0.
    """

    beta_rel: float = 0.02
    alpha_rel: float = 0.0
    alpha_abs: float = 0.01
    phase_abs: float = 2e-3
    dispersion_rel: float = 0.05
    dispersion_abs: float = 1e-3


@dataclass(frozen=True)
class ResolvedOptics:
    """Resolved optics inputs for the reconstruction pipeline.

    Attributes:
        tws: Twiss DataFrame (tfs, lowercase columns/headers) with all optics
            and uncertainty columns, indexed by BPM name in ring order.
        sources: Resolved source per optics category.
    """

    tws: tfs.TfsDataFrame
    sources: dict[OpticsCategory, OpticsSource]


def _get_tune(tws: pd.DataFrame, key: str) -> float:
    """Read a tune from Twiss headers."""
    for k, value in tws.headers.items():
        if str(k).lower() == key:
            return float(value)
    raise KeyError(f"Twiss table is missing tune {key!r}")


@dataclass(frozen=True)
class LoadedMeasurement:
    """A measurement directory loaded into a lowercase-renamed twiss.

    Caching this lets :func:`resolve_optics` be re-run for many model twisses
    without re-reading the (unchanging) omc3 measurement from disk each time.

    Attributes:
        tws: The lowercase-renamed measurement twiss.
        dispersion_found: Whether measured dispersion columns were present.
    """

    tws: pd.DataFrame
    dispersion_found: bool


def load_measurement(
    measurement_dir: str | Path,
    *,
    reverse_meas_tws: bool = False,
    bpm_names: Collection[str] | None = None,
) -> LoadedMeasurement:
    """Load an omc3 measurement directory into a cacheable :class:`LoadedMeasurement`."""
    tws, dispersion_found = load_measurement_twiss(
        measurement_dir, reverse_meas_tws=reverse_meas_tws, bpm_names=bpm_names
    )
    return LoadedMeasurement(tws=tws, dispersion_found=dispersion_found)


def load_measurement_twiss(
    measurement_dir: str | Path,
    *,
    reverse_meas_tws: bool = False,
    bpm_names: Collection[str] | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Load an omc3 measurement directory as a lowercase-renamed twiss.

    Returns:
        Tuple of (twiss, dispersion_found).
    """
    tws, dispersion_found = build_twiss_from_measurements(
        Path(measurement_dir),
        include_errors=True,
        reverse_bpm_order=reverse_meas_tws,
    )
    if bpm_names is not None:
        tws = tws[tws.index.isin(set(bpm_names))]

    rename_mapping = {**MEASUREMENT_RENAME_MAPPING, **ERROR_RENAME_MAPPING}
    if dispersion_found:
        rename_mapping.update(DISPERSION_RENAME_MAPPING)
        rename_mapping.update(ERROR_DISPERSION_RENAME_MAPPING)
    tws = tws.rename(columns=rename_mapping)

    tws.index.name = (tws.index.name or "name").lower()
    tws.columns = [str(col).lower() for col in tws.columns]
    tws.headers = {str(key).lower(): value for key, value in tws.headers.items()}
    return tws, dispersion_found


def _resolve_sources(
    *, model_tws: pd.DataFrame | None, optics: OpticsInput
) -> dict[OpticsCategory, OpticsSource]:
    """Validate the caller's explicit source choice for every category."""
    sources = {category: optics.source_for(category) for category in CATEGORIES}
    if model_tws is None and any(source == "model" for source in sources.values()):
        raise ValueError("Model optics were requested but no model twiss was given")
    return sources


def _require_measured_category(
    tws: pd.DataFrame, category: OpticsCategory, *, dispersion_found: bool
) -> None:
    if category == "dispersion" and not dispersion_found:
        raise KeyError(
            "Measured dispersion was requested but the measurement contains no dispersion"
        )
    missing = [column for column in CATEGORY_COLUMNS[category] if column not in tws.columns]
    if missing:
        hint = ""
        if category == "dispersion":
            hint = (
                " The angular half is not measured: derive it with "
                "tmom_recon.momentum_dispersion.derive_momentum_dispersion against the "
                "free, on-momentum lattice the position dispersion was measured in, and "
                "write it into the measurement."
            )
        raise KeyError(
            f"Measured optics category {category!r} was requested but is missing columns "
            f"{missing}.{hint}"
        )


def _synthesise_beta_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    sqrt_betax = np.sqrt(tws["betx"].to_numpy(dtype=float))
    sqrt_betay = np.sqrt(tws["bety"].to_numpy(dtype=float))
    tws["sqrt_betax_err"] = errors.beta_rel * sqrt_betax / 2.0
    tws["sqrt_betay_err"] = errors.beta_rel * sqrt_betay / 2.0


def _synthesise_alpha_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    if errors.alpha_rel > 0.0:
        alfa_x = np.abs(tws["alfx"].to_numpy(dtype=float))
        alfa_y = np.abs(tws["alfy"].to_numpy(dtype=float))
        tws["alfax_err"] = np.maximum(errors.alpha_rel * alfa_x, errors.alpha_abs)
        tws["alfay_err"] = np.maximum(errors.alpha_rel * alfa_y, errors.alpha_abs)
    else:
        tws["alfax_err"] = errors.alpha_abs
        tws["alfay_err"] = errors.alpha_abs


def _convert_measured_beta_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    """Convert raw measured beta errors to sqrt(beta) errors, with model fallback."""
    for err_col, beta_col in (("sqrt_betax_err", "betx"), ("sqrt_betay_err", "bety")):
        sqrt_beta = np.sqrt(tws[beta_col].to_numpy(dtype=float))
        if err_col in tws.columns:
            tws[err_col] = tws[err_col].to_numpy(dtype=float) / (2.0 * sqrt_beta)
        else:
            LOGGER.warning("Measured %s missing; using model uncertainty", err_col)
            tws[err_col] = errors.beta_rel * sqrt_beta / 2.0


def _check_measured_alpha_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    """Keep the measured alpha errors, filling in the model value where absent."""
    for alfa_col in ("alfax_err", "alfay_err"):
        if alfa_col not in tws.columns:
            LOGGER.warning("Measured %s missing; using model uncertainty", alfa_col)
            tws[alfa_col] = errors.alpha_abs


def _synthesise_phase_variances(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    """Accumulate a linear phase-variance ramp, mimicking a measured phase chain."""
    n = len(tws)
    ramp = np.arange(n, dtype=float) * errors.phase_abs**2
    tws["mu1_var"] = ramp
    tws["mu2_var"] = ramp
    tws.headers["mu1_total_var"] = float(n) * errors.phase_abs**2
    tws.headers["mu2_total_var"] = float(n) * errors.phase_abs**2


def _synthesise_dispersion_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    for col in DISPERSION_COLUMNS:
        err_col = f"{col}_err"
        if err_col not in tws.columns:
            values = np.abs(tws[col].to_numpy(dtype=float))
            tws[err_col] = np.maximum(errors.dispersion_rel * values, errors.dispersion_abs)


def _canonical_model_betas(optics_tws: tfs.TfsDataFrame, dpp: float) -> tfs.TfsDataFrame:
    """Model twiss with ``betx``/``bety`` divided by ``1 + dp/p``.

    Off momentum MAD-NG reports ``betx = beta11_decoupled * (1 + dp/p)`` (on an
    uncoupled lattice exactly ``beta11 * (1 + dp/p)``), with ``alfx`` unscaled.
    Fed to the two-BPM formula, which returns the canonical ``px``, that factor
    gives a momentum error of order dp/p (0.13 % / 0.3 % in x / y at
    dp/p = 1.2e-3 on the PSB). ``beta11`` is not used instead: under coupling the
    decoupled ``betx`` suits the uncoupled formula better.
    """
    if not dpp:
        return optics_tws
    out = optics_tws.copy(deep=True)
    for column in BETA_COLUMNS:
        if column in out.columns:
            out[column] = out[column].to_numpy(dtype=float) / (1.0 + dpp)
    return out


def resolve_optics(
    *,
    optics_tws: tfs.TfsDataFrame,
    zero_tws: pd.DataFrame,
    optics: OpticsInput = OpticsInput(),  # noqa: B008 - frozen dataclass
    model_errors: ModelOpticsErrors | None = None,
    bpm_names: Collection[str] | None = None,
    measured: LoadedMeasurement | None = None,
    dpp: float = 0.0,
) -> ResolvedOptics:
    """Build the resolved twiss used by the momentum reconstruction pipeline.

    Args:
        optics_tws: Model twiss indexed by element name (lowercase optics
            columns ``betx/alfx/mu1/...`` and tune headers ``q1``/``q2``).
        zero_tws: Undriven ``dp/p = 0`` model twiss (``chrom=True``), the
            source of all model dispersion and of the second-order dispersion.
        optics: Explicit source selection for model and measured optics.
        model_errors: Rough uncertainties for model-sourced categories.
        bpm_names: Optional BPM subset to restrict the twiss to.
        measured: Pre-loaded measurement (see :func:`load_measurement`). When
            given, it is used instead of reading *measurement_dir* from disk,
            so repeated resolves for different model twisses avoid the reload.
        dpp: The dp/p of *pt*. MAD-NG's ``betx``/``bety`` carry a factor
            ``1 + dp/p`` (they are the betas of ``x' = px/(1+dp/p)``), while the
            reconstruction works with the canonical ``px``; the model betas are
            divided by it (see :func:`_canonical_model_betas`).

    Returns:
        A :class:`ResolvedOptics` bundle.

    Raises:
        ValueError: On invalid categories or unsatisfiable source requests.
        KeyError: If a required optics column is missing from its source.
    """
    errors = model_errors if model_errors is not None else ModelOpticsErrors()
    optics_tws = _canonical_model_betas(optics_tws, dpp)
    measured_tws = None
    measurement_dispersion_found = False
    if measured is not None:
        measured_tws = measured.tws
        measurement_dispersion_found = measured.dispersion_found
    elif optics.measurement_dir is not None:
        measured_tws, measurement_dispersion_found = load_measurement_twiss(
            optics.measurement_dir,
            reverse_meas_tws=optics.reverse_measurement_order,
            bpm_names=bpm_names,
        )

    sources = _resolve_sources(model_tws=optics_tws, optics=optics)
    if any(source == "measurement" for source in sources.values()) and measured_tws is None:
        raise ValueError("Measured optics were requested but could not be loaded")
    if measured_tws is not None:
        for category, source in sources.items():
            if source == "measurement":
                _require_measured_category(
                    measured_tws, category, dispersion_found=measurement_dispersion_found
                )
    LOGGER.info("Resolved optics sources: %s", sources)

    if measured_tws is not None:
        tws = measured_tws
        shared = tws.index.intersection(optics_tws.index)
        tws = tws.loc[shared].copy(deep=True)
    else:
        tws = optics_tws.copy(deep=True)
        if bpm_names is not None:
            tws: tfs.TfsDataFrame = tws[tws.index.isin(set(bpm_names))]  # ty:ignore[invalid-assignment]
        tws.headers = {"q1": _get_tune(optics_tws, "q1"), "q2": _get_tune(optics_tws, "q2")}

    model_view = optics_tws.loc[tws.index]
    zero = pd.DataFrame(zero_tws).copy()
    zero.index = zero.index.astype(str).str.upper()
    zero_view = zero.reindex(tws.index.astype(str).str.upper())
    for category, columns in CATEGORY_COLUMNS.items():
        # Dispersion is always taken about pt = 0 below.
        if sources[category] != "model" or measured_tws is None or category == "dispersion":
            continue
        for column in columns:
            if column not in model_view.columns:
                raise KeyError(f"Model twiss is missing required column {column!r}")
            tws[column] = model_view[column].to_numpy(dtype=float)

    if sources["phase"] == "model":
        tws.headers["q1"] = _get_tune(optics_tws, "q1")
        tws.headers["q2"] = _get_tune(optics_tws, "q2")

    if sources["beta"] == "measurement":
        _convert_measured_beta_errors(tws, errors)
    else:
        _synthesise_beta_errors(tws, errors)

    if sources["alpha"] == "measurement":
        _check_measured_alpha_errors(tws, errors)
    else:
        _synthesise_alpha_errors(tws, errors)

    if sources["phase"] == "model" or "mu1_var" not in tws.columns:
        _synthesise_phase_variances(tws, errors)

    if sources["dispersion"] == "model":
        missing = [col for col in DISPERSION_COLUMNS if col not in zero_view.columns]
        if missing:
            raise KeyError(f"Zero-momentum model twiss is missing dispersion columns {missing}")
        absent = zero_view.index[zero_view[list(DISPERSION_COLUMNS)].isna().any(axis=1)].tolist()
        if absent:
            raise KeyError(f"Zero-momentum model twiss is missing BPM(s) {absent}")
        for column in DISPERSION_COLUMNS:
            tws[column] = zero_view[column].to_numpy(dtype=float)
    missing = [col for col in DISPERSION_COLUMNS if col not in tws.columns]
    if missing:
        raise KeyError(f"Resolved optics is missing required dispersion columns {missing}")
    _synthesise_dispersion_errors(tws, errors)
    for column in SECOND_ORDER_DISPERSION_COLUMNS:
        if column in zero_view.columns:
            tws[column] = zero_view[column].to_numpy(dtype=float)

    return ResolvedOptics(
        tws=tws,
        sources=sources,
    )
