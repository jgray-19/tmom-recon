"""Explicit AC-dipole barrier positions for all-BPM reconstruction tests."""

from __future__ import annotations


def acd_barrier_s(model, ac_dipole_marker: str) -> float:
    """Return the longitudinal position of an ACD marker from a MAD-NG model.

    This deliberately lives in test support rather than reconstruction: callers
    must state whether their data contain a localised kick, and must supply its
    position to ``calculate_pz`` rather than relying on an implicit model lookup.
    """
    twiss_elements = model.twiss_elements
    return float(twiss_elements.loc[ac_dipole_marker.upper(), "s"])


__all__ = ["acd_barrier_s"]
