"""Named scenario factories exposed as the test-support API."""

from tests.support.scenarios import (
    MATCHED_BEND_AND_QUAD_ERRORS,
    MATCHED_BEND_ERRORS,
    NO_ERRORS,
    UNMODELLED_BEND_ERRORS,
    MagnetErrors,
    psb_clean,
    psb_matched_bend_and_quad_errors,
    psb_matched_bend_errors,
    psb_offmomentum,
    psb_sextupole_feeddown,
    psb_unmodelled_bend_errors,
    scenario,
)

__all__ = [
    "MATCHED_BEND_AND_QUAD_ERRORS",
    "MATCHED_BEND_ERRORS",
    "MagnetErrors",
    "NO_ERRORS",
    "UNMODELLED_BEND_ERRORS",
    "psb_clean",
    "psb_matched_bend_and_quad_errors",
    "psb_matched_bend_errors",
    "psb_offmomentum",
    "psb_sextupole_feeddown",
    "psb_unmodelled_bend_errors",
    "scenario",
]
