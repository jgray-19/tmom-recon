"""Per-category optics-source resolution for momentum reconstruction.

Builds a single resolved twiss DataFrame from a model twiss and/or an omc3
optics measurement directory. Each optics category — ``phase`` (mu1/mu2 and
tunes), ``amplitude`` (beta and alpha) and ``dispersion`` — is sourced
independently, defaulting to the measurement when one is available.

Every resolved twiss carries a full set of uncertainty columns: measured
errors where the measurement is used, and rough configurable uncertainties
(:class:`ModelOpticsErrors`) where the model is used, so error propagation
always produces meaningful ``var_px``/``var_py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
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

    import pandas as pd

LOGGER = logging.getLogger(__name__)

OpticsCategory = Literal["phase", "amplitude", "dispersion"]
OpticsSource = Literal["model", "measurement"]

CATEGORIES: tuple[OpticsCategory, ...] = ("phase", "amplitude", "dispersion")

PHASE_COLUMNS = ("mu1", "mu2")
AMPLITUDE_COLUMNS = ("beta11", "beta22", "alfa11", "alfa22")
DISPERSION_COLUMNS = ("dx", "dy", "dpx", "dpy")

CATEGORY_COLUMNS: dict[OpticsCategory, tuple[str, ...]] = {
    "phase": PHASE_COLUMNS,
    "amplitude": AMPLITUDE_COLUMNS,
    "dispersion": DISPERSION_COLUMNS,
}


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
        co: Twiss used for closed-orbit removal/restoration and pt
            estimation (the model twiss when available, else ``tws``).
        sources: Resolved source per optics category.
        use_dispersion: Whether dispersion is available and enabled.
    """

    tws: tfs.TfsDataFrame
    co: pd.DataFrame
    sources: dict[OpticsCategory, OpticsSource]
    use_dispersion: bool


def _get_tune(tws: pd.DataFrame, key: str) -> float:
    """Read a tune from twiss headers (case-insensitive) or attributes."""
    headers = dict(getattr(tws, "headers", {}) or {})
    for k, value in headers.items():
        if str(k).lower() == key:
            return float(value)
    value = getattr(tws, key, None)
    if value is not None and not callable(value):
        return float(value)
    raise KeyError(f"Twiss table is missing tune {key!r} in headers or attributes")


@dataclass(frozen=True)
class LoadedMeasurement:
    """A measurement directory loaded into a lowercase-renamed twiss.

    Caching this lets :func:`resolve_optics` be re-run for many model twisses
    without re-reading the (unchanging) omc3 measurement from disk each time.

    Attributes:
        tws: The lowercase-renamed measurement twiss.
        dispersion_found: Whether measured dispersion columns were present.
    """

    tws: tfs.TfsDataFrame
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
) -> tuple[tfs.TfsDataFrame, bool]:
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


def _validate_categories(model_optics: Collection[str]) -> set[OpticsCategory]:
    invalid = set(model_optics) - set(CATEGORIES)
    if invalid:
        raise ValueError(
            f"Unknown optics categories in model_optics: {sorted(invalid)}; "
            f"valid categories are {list(CATEGORIES)}"
        )
    return set(model_optics)  # type: ignore[arg-type]


def _resolve_sources(
    *,
    model_tws: pd.DataFrame | None,
    has_measurement: bool,
    model_optics: set[OpticsCategory],
    use_dispersion: bool,
    measurement_dispersion_found: bool,
) -> tuple[dict[OpticsCategory, OpticsSource], bool]:
    """Decide the source for each category and whether dispersion stays on."""
    sources: dict[OpticsCategory, OpticsSource] = {}
    for category in ("phase", "amplitude"):
        if category in model_optics:
            sources[category] = "model"
        else:
            sources[category] = "measurement" if has_measurement else "model"
        if sources[category] == "model" and model_tws is None:
            raise ValueError(
                f"Optics category {category!r} resolved to the model but no model twiss was given"
            )

    dispersion_on = use_dispersion
    if not use_dispersion:
        sources["dispersion"] = "measurement" if has_measurement else "model"
    elif "dispersion" in model_optics:
        if model_tws is None:
            raise ValueError(
                "Optics category 'dispersion' resolved to the model but no model twiss was given"
            )
        sources["dispersion"] = "model"
    elif has_measurement and measurement_dispersion_found:
        sources["dispersion"] = "measurement"
    elif model_tws is not None:
        if has_measurement:
            LOGGER.warning("No measured dispersion found; falling back to model dispersion")
        sources["dispersion"] = "model"
    else:
        LOGGER.warning("No dispersion available from the measurement; disabling dispersion")
        sources["dispersion"] = "measurement"
        dispersion_on = False

    return sources, dispersion_on


def _synthesise_amplitude_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    sqrt_betax = np.sqrt(tws["beta11"].to_numpy(dtype=float))
    sqrt_betay = np.sqrt(tws["beta22"].to_numpy(dtype=float))
    tws["sqrt_betax_err"] = errors.beta_rel * sqrt_betax / 2.0
    tws["sqrt_betay_err"] = errors.beta_rel * sqrt_betay / 2.0
    if errors.alpha_rel > 0.0:
        alfa_x = np.abs(tws["alfa11"].to_numpy(dtype=float))
        alfa_y = np.abs(tws["alfa22"].to_numpy(dtype=float))
        tws["alfax_err"] = np.maximum(errors.alpha_rel * alfa_x, errors.alpha_abs)
        tws["alfay_err"] = np.maximum(errors.alpha_rel * alfa_y, errors.alpha_abs)
    else:
        tws["alfax_err"] = errors.alpha_abs
        tws["alfay_err"] = errors.alpha_abs


def _convert_measured_beta_errors(tws: pd.DataFrame, errors: ModelOpticsErrors) -> None:
    """Convert raw measured beta errors to sqrt(beta) errors, with model fallback."""
    for err_col, beta_col, alfa_col in (
        ("sqrt_betax_err", "beta11", "alfax_err"),
        ("sqrt_betay_err", "beta22", "alfay_err"),
    ):
        sqrt_beta = np.sqrt(tws[beta_col].to_numpy(dtype=float))
        if err_col in tws.columns:
            tws[err_col] = tws[err_col].to_numpy(dtype=float) / (2.0 * sqrt_beta)
        else:
            LOGGER.warning("Measured %s missing; using model uncertainty", err_col)
            tws[err_col] = errors.beta_rel * sqrt_beta / 2.0
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


def resolve_optics(
    *,
    model_tws: tfs.TfsDataFrame | None = None,
    measurement_dir: str | Path | None = None,
    model_optics: Collection[OpticsCategory] = (),
    use_dispersion: bool = True,
    model_errors: ModelOpticsErrors | None = None,
    reverse_meas_tws: bool = False,
    bpm_names: Collection[str] | None = None,
    measured: LoadedMeasurement | None = None,
) -> ResolvedOptics:
    """Build the resolved twiss used by the momentum reconstruction pipeline.

    Args:
        model_tws: Model twiss indexed by element name (lowercase optics
            columns ``beta11/alfa11/mu1/...`` and tune headers ``q1``/``q2``).
        measurement_dir: omc3 optics measurement directory.
        model_optics: Categories forced to come from the model. Categories not
            listed come from the measurement when available, model otherwise.
        use_dispersion: If False, dispersion is excluded from the reconstruction.
        model_errors: Rough uncertainties for model-sourced categories.
        reverse_meas_tws: Reverse BPM ordering when reading measurement phases.
        bpm_names: Optional BPM subset to restrict the twiss to.
        measured: Pre-loaded measurement (see :func:`load_measurement`). When
            given, it is used instead of reading *measurement_dir* from disk,
            so repeated resolves for different model twisses avoid the reload.

    Returns:
        A :class:`ResolvedOptics` bundle.

    Raises:
        ValueError: On invalid categories or unsatisfiable source requests.
        KeyError: If a required optics column is missing from its source.
    """
    has_measurement = measurement_dir is not None or measured is not None
    if model_tws is None and not has_measurement:
        raise ValueError("At least one of model_tws or measurement_dir must be provided")
    model_categories = _validate_categories(model_optics)
    errors = model_errors if model_errors is not None else ModelOpticsErrors()

    measured_tws = None
    measurement_dispersion_found = False
    if measured is not None:
        measured_tws = measured.tws
        measurement_dispersion_found = measured.dispersion_found
    elif measurement_dir is not None:
        measured_tws, measurement_dispersion_found = load_measurement_twiss(
            measurement_dir, reverse_meas_tws=reverse_meas_tws, bpm_names=bpm_names
        )

    sources, dispersion_on = _resolve_sources(
        model_tws=model_tws,
        has_measurement=has_measurement,
        model_optics=model_categories,
        use_dispersion=use_dispersion,
        measurement_dispersion_found=measurement_dispersion_found,
    )
    LOGGER.info("Resolved optics sources: %s (dispersion %s)", sources, dispersion_on)

    if measured_tws is not None:
        tws = measured_tws
        if model_tws is not None:
            shared = tws.index.intersection(model_tws.index)
            tws = tws.loc[shared].copy(deep=True)
    else:
        tws = tfs.TfsDataFrame(model_tws.copy(deep=True))
        if bpm_names is not None:
            tws = tws[tws.index.isin(set(bpm_names))]
        tws.headers = {"q1": _get_tune(model_tws, "q1"), "q2": _get_tune(model_tws, "q2")}

    # Overwrite model-sourced categories from the model twiss
    model_view = model_tws.loc[tws.index] if model_tws is not None else None
    for category, columns in CATEGORY_COLUMNS.items():
        if category == "dispersion" and not dispersion_on:
            continue
        if sources[category] != "model" or measured_tws is None:
            continue
        for column in columns:
            if column not in model_view.columns:
                raise KeyError(f"Model twiss is missing required column {column!r}")
            tws[column] = model_view[column].to_numpy(dtype=float)

    # Tunes follow the phase source
    if sources["phase"] == "model" and model_tws is not None:
        tws.headers["q1"] = _get_tune(model_tws, "q1")
        tws.headers["q2"] = _get_tune(model_tws, "q2")

    # Uncertainty columns: measured where measured, rough model errors elsewhere
    if sources["amplitude"] == "measurement":
        _convert_measured_beta_errors(tws, errors)
    else:
        _synthesise_amplitude_errors(tws, errors)

    if sources["phase"] == "model" or "mu1_var" not in tws.columns:
        _synthesise_phase_variances(tws, errors)

    if dispersion_on and any(col not in tws.columns for col in DISPERSION_COLUMNS):
        missing = [col for col in DISPERSION_COLUMNS if col not in tws.columns]
        LOGGER.warning("Dispersion columns %s unavailable; disabling dispersion", missing)
        dispersion_on = False
    if dispersion_on:
        _synthesise_dispersion_errors(tws, errors)

    return ResolvedOptics(
        tws=tws,
        co=model_tws if model_tws is not None else tws,
        sources=sources,
        use_dispersion=dispersion_on,
    )
