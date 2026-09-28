"""Contracts for the explicit closed-orbit reconstruction inputs."""

from __future__ import annotations

import inspect

from tmom_recon import calculate_pz


def test_frame_api_is_absent() -> None:
    parameters = inspect.signature(calculate_pz).parameters
    assert "frame" not in parameters
    assert "closed_orbit_at_zero" in parameters
    assert "orbit_mode" in parameters
    assert "reference" not in parameters and "measurement_pt" not in parameters


def test_barrier_decision_remains_explicit() -> None:
    assert (
        inspect.signature(calculate_pz).parameters["barrier_s"].default is inspect.Parameter.empty
    )
