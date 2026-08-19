"""Reconstruction entry points used by integration tests."""

from __future__ import annotations

from tmom_recon import calculate_pz


def transverse_calc(df, model_details, reference, *, ac_dipole_config=None, use_dispersion: bool = True, **kwargs):
    """Model-only transverse reconstruction entry point."""
    result = calculate_pz(
        df,
        model_details,
        reference=reference,
        use_dispersion=use_dispersion,
        acd=ac_dipole_config,
        **kwargs,
    )
    assert hasattr(result, "columns")
    return result


def dispersive_calc(df, model_details, reference, *, ac_dipole_config=None, **kwargs):
    """Model-only dispersive reconstruction entry point."""
    result = calculate_pz(
        df,
        model_details,
        reference=reference,
        use_dispersion=True,
        acd=ac_dipole_config,
        **kwargs,
    )
    assert hasattr(result, "columns")
    return result


__all__ = ["dispersive_calc", "transverse_calc"]
