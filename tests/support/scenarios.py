"""Named accelerator scenarios used by integration and regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.psb_tracking import DRIVEN_TUNES, build_psb_tracking_setup
from tests.support.psb import PSBScenario


@dataclass(frozen=True)
class MagnetErrors:
    """Seeded magnet perturbations and whether the model carries them."""

    bend_rms: float = 0.0
    bend_seed: int = 0
    quad_rms: float = 0.0
    quad_seed: int = 0
    quad_misalign_y_rms: float = 0.0
    quad_misalign_seed: int = 0
    model_matches_machine: bool = True


NO_ERRORS = MagnetErrors()
MATCHED_BEND_ERRORS = MagnetErrors(bend_rms=8e-4, bend_seed=7)
UNMODELLED_BEND_ERRORS = MagnetErrors(
    bend_rms=8e-4,
    bend_seed=7,
    model_matches_machine=False,
)
MATCHED_BEND_AND_QUAD_ERRORS = MagnetErrors(
    bend_rms=8e-4,
    bend_seed=7,
    quad_rms=1e-3,
    quad_seed=11,
)


def scenario(
    model_dir: Path,
    *,
    delta_p: float = 0.0,
    errors: MagnetErrors = NO_ERRORS,
    driven_tunes: tuple[float, float] = DRIVEN_TUNES,
    sextupole_k2l: float = 0.0,
) -> PSBScenario:
    """Build one explicit PSB scenario from a machine/error specification."""
    return build_psb_tracking_setup(
        model_dir,
        delta_p=delta_p,
        driven_tunes=driven_tunes,
        bend_error_rms=errors.bend_rms,
        bend_error_seed=errors.bend_seed,
        apply_bend_errors_to_model=errors.model_matches_machine,
        quad_error_rms=errors.quad_rms,
        quad_error_seed=errors.quad_seed,
        apply_quad_errors_to_model=errors.model_matches_machine,
        quad_misalign_y_rms=errors.quad_misalign_y_rms,
        quad_misalign_seed=errors.quad_misalign_seed,
        sextupole_k2l=sextupole_k2l,
    )


def psb_clean(model_dir: Path, *, delta_p: float = 0.0) -> PSBScenario:
    return scenario(model_dir, delta_p=delta_p, errors=NO_ERRORS)


def psb_offmomentum(model_dir: Path, delta_p: float) -> PSBScenario:
    return scenario(model_dir, delta_p=delta_p, errors=NO_ERRORS)


def psb_matched_bend_errors(model_dir: Path, *, delta_p: float = 0.0) -> PSBScenario:
    return scenario(model_dir, delta_p=delta_p, errors=MATCHED_BEND_ERRORS)


def psb_unmodelled_bend_errors(model_dir: Path, *, delta_p: float = 0.0) -> PSBScenario:
    return scenario(model_dir, delta_p=delta_p, errors=UNMODELLED_BEND_ERRORS)


def psb_matched_bend_and_quad_errors(model_dir: Path, *, delta_p: float = 0.0) -> PSBScenario:
    return scenario(model_dir, delta_p=delta_p, errors=MATCHED_BEND_AND_QUAD_ERRORS)


def psb_sextupole_feeddown(
    model_dir: Path, *, k2l: float, bend_errors: bool = False
) -> PSBScenario:
    errors = MATCHED_BEND_ERRORS if bend_errors else NO_ERRORS
    return scenario(model_dir, errors=errors, sextupole_k2l=k2l)
