"""Model-detail factories for integration tests."""

from pymadng_utils.accelerators import LHC

from tmom_recon import ModelDetails


def model_details_for(accelerator, *, pt: float) -> ModelDetails:
    """Build model details for a generated accelerator at absolute ``pt``."""
    return ModelDetails(accelerator=accelerator, pt=float(pt))


def lhc_model_details(seq_file: str, data_dir, *, delta_p: float = 0.0) -> ModelDetails:
    """Build LHC model details at the tracked absolute momentum."""
    accelerator = LHC(
        beam=1,
        sequence_file=data_dir / "sequences" / seq_file,
        kinetic_energy=6800,
    )
    return model_details_for(accelerator, pt=accelerator.dp2pt(delta_p))


__all__ = ["lhc_model_details", "model_details_for"]
