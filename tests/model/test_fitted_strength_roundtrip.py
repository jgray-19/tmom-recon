"""Plain-data fitted strengths reproduce the same closed orbit in both repos."""

from __future__ import annotations

import numpy as np
from aba_optimiser.accelerators import PSB as OptimiserPSB  # noqa: N811
from aba_optimiser.mad import GradientDescentMadInterface
from aba_optimiser.momentum_reference import closed_orbit_at
from pymadng_utils.accelerators import PSB

from tmom_recon import ModelDetails
from tmom_recon.model import resolve_model_details


def test_dk0l_dk1l_strengths_round_trip_through_model_details(seq_psb3) -> None:
    fit_accelerator = OptimiserPSB(
        ring=3,
        sequence_file=seq_psb3,
        optimise_bends=True,
        optimise_quadrupoles=True,
    )
    interface = GradientDescentMadInterface(fit_accelerator)
    try:
        bend = next(name for name in interface.knob_names if name.endswith(".dk0l"))
        quad = next(name for name in interface.knob_names if name.endswith(".dk1l"))
    finally:
        interface.close()
    strengths = {bend: 1.7e-5, quad: -2.3e-4}

    fitted_orbit = closed_orbit_at(fit_accelerator, strengths, pt=0.0)
    reconstruction = resolve_model_details(
        ModelDetails(
            accelerator=PSB(
                ring=3,
                sequence_file=seq_psb3,
                kinetic_energy=0.160,
            ),
            magnet_strengths=strengths,
        )
    ).closed_orbit_tws

    common = fitted_orbit.index.intersection(reconstruction.index)
    # The optimiser observes the transfer monitor BR3.BPMT3L1 as a seventeenth
    # diagnostic; tmom-recon's production PSB BPM pattern intentionally carries
    # the sixteen position pickups used by reconstruction.
    assert len(common) == 16
    np.testing.assert_allclose(
        reconstruction.loc[common, ["x", "y", "px", "py"]],
        fitted_orbit.loc[common, ["x", "y", "px", "py"]],
        rtol=1e-10,
        atol=1e-12,
    )
