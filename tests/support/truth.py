"""Truth and optics-conversion helpers for integration tests."""

from __future__ import annotations

import pandas as pd

from tmom_recon import ModelDetails, MomentumReference
from tmom_recon.model import resolve_model_details


def model_details_for(accelerator, *, pt: float) -> ModelDetails:
    """Build model details for a generated accelerator at absolute ``pt``."""
    return ModelDetails(accelerator=accelerator, pt=float(pt))


def simulated_mixed_reference_from_model(
    model_details: ModelDetails, df: pd.DataFrame, *, include_angles: bool = True
) -> MomentumReference:
    """Build simulated measured positions with optional model-derived angles.

    This is a test-only compatibility helper. ``x``/``y`` represent the
    nominal reference measurement; ``px``/``py`` represent the angle returned
    by an external fitted-strength model. It never claims that a model orbit is
    a production measurement.
    """
    if model_details.pt != 0.0:
        raise ValueError("simulated_mixed_reference_from_model requires pt=0.0")
    names = pd.Index(pd.unique(df["name"]), name="name")
    wants_markers = any(str(name).endswith(("_BEFORE", "_AFTER")) for name in names)
    closed_orbit = resolve_model_details(
        model_details,
        observed_elements=[str(name) for name in names],
        install_ac_dipole_markers=wants_markers,
    ).tws
    by_upper = closed_orbit.rename(index=lambda name: str(name).upper())
    wanted = pd.Index([str(name).upper() for name in names], name="name")
    missing = wanted.difference(by_upper.index)
    if len(missing):
        raise ValueError(f"Model twiss is missing names present in data: {list(missing)[:10]}")
    orbit = by_upper.loc[wanted].copy()
    orbit.index = names
    if not include_angles:
        orbit = orbit[["x", "y"]]
    elif not {"px", "py"}.issubset(orbit.columns):
        raise ValueError("Model twiss is missing closed-orbit angle columns px/py")
    return MomentumReference(
        orbit[[col for col in ("x", "y", "px", "py") if col in orbit]], pt=model_details.pt
    )


def simulated_reference_from_tracking_positions_and_model_angles(
    nominal_tracking_data: pd.DataFrame,
    model_details: ModelDetails,
    reconstruction_data: pd.DataFrame,
) -> MomentumReference:
    """Build the simulated optimiser hand-off from independent inputs.

    The nominal orbit positions are the synthetic *measurement*, obtained from
    tracked nominal-reference data.  The fitted-strength model supplies only
    the unmeasurable ``px``/``py`` reference angles.  This deliberately avoids
    constructing both halves of the reference from one model.
    """
    if model_details.pt != 0.0:
        raise ValueError("reference-angle model must be at nominal pt=0.0")
    names = pd.Index(pd.unique(reconstruction_data["name"]), name="name")
    tracked = nominal_tracking_data.loc[
        nominal_tracking_data["name"].isin(names), ["name", "x", "y"]
    ]
    positions = tracked.groupby("name", sort=False)[["x", "y"]].mean().reindex(names)
    if positions.isna().any().any():
        missing = positions.index[positions.isna().any(axis=1)].tolist()
        raise ValueError(f"Nominal tracking reference is missing BPMs: {missing[:10]}")

    model_reference = simulated_mixed_reference_from_model(
        model_details, reconstruction_data, include_angles=True
    ).closed_orbit
    positions["px"] = model_reference.loc[names, "px"].to_numpy(dtype=float)
    positions["py"] = model_reference.loc[names, "py"].to_numpy(dtype=float)
    return MomentumReference(positions, pt=0.0)
