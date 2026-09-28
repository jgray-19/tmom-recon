"""Integration tests for :class:`tmom_recon.reconstruction.ACDipolePzGenerator`.

The generator must give *exactly* the same result as a one-shot
``calculate_acd_pz`` call for the same optics (it shares the same
``reconstruct_from_prepared`` code path), must freeze the input data so repeated
updates are deterministic, and must track changes to the model optics.

These run the real MAD-NG driver, mirroring the other ACD integration tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.reference_co import measured_zero_reference_for_simulation
from tmom_recon import ACDipoleConfig, ACDipolePzGenerator, ModelDetails, calculate_acd_pz
from tmom_recon.acd.madng_driver import ACDipoleMadDriver

from .acd_test_helpers import AC_DIPOLE_ELEMENT, _ac_dipole_segment_around_element, _get_driver

SEQ_FILE = "lhcb1.seq"
DRIVEN_TUNES = (0.27, 0.322)

pytestmark = [pytest.mark.lhc, pytest.mark.integration, pytest.mark.slow]


def _config(*, bpm_upstream: str, bpm_downstream: str) -> ACDipoleConfig:
    return ACDipoleConfig(
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        driven_tunes=DRIVEN_TUNES,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
    )


def _setup(data_dir, acd_tracking_setup) -> tuple[pd.DataFrame, ACDipoleMadDriver, str, str]:
    sequence_file = data_dir / "sequences" / SEQ_FILE
    setup = acd_tracking_setup(sequence_file, delta_p=0.0, flattop_turns=100)
    tracking_df = setup.data
    driver = _get_driver(sequence_file, debug=False)
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        driver.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )
    return tracking_df, driver, bpm_upstream, bpm_downstream


@pytest.mark.slow
def test_generator_update_matches_acd_only(data_dir, acd_tracking_setup) -> None:
    """The ACD generator is identical to the one-shot ACD workflow."""
    tracking_df, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    model_details = ModelDetails(accelerator=driver.accelerator, pt=driver.pt)
    config = _config(bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    generator = ACDipolePzGenerator.build(
        data=tracking_df,
        closed_orbit_at_zero=measured_zero_reference_for_simulation(tracking_df),
        orbit_mode="dynamic",
        model_details=model_details,
        config=config,
    )
    assert isinstance(generator, ACDipolePzGenerator)
    assert generator.model.accelerator is driver.accelerator

    from_generator = generator.update()
    one_shot = calculate_acd_pz(
        tracking_df,
        model_details,
        config,
        closed_orbit_at_zero=measured_zero_reference_for_simulation(tracking_df),
        orbit_mode="dynamic",
    )

    pd.testing.assert_frame_equal(from_generator, one_shot)
    pd.testing.assert_frame_equal(from_generator.attrs["summary"], one_shot.attrs["summary"])
    assert generator.latest is from_generator


@pytest.mark.slow
def test_generator_repeated_update_is_deterministic(data_dir, acd_tracking_setup) -> None:
    """The frozen data means re-running with the same twiss is bit-for-bit stable."""
    tracking_df, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    model_details = ModelDetails(accelerator=driver.accelerator, pt=driver.pt)
    config = _config(bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    generator = ACDipolePzGenerator.build(
        data=tracking_df,
        closed_orbit_at_zero=measured_zero_reference_for_simulation(tracking_df),
        orbit_mode="dynamic",
        model_details=model_details,
        config=config,
    )
    assert isinstance(generator, ACDipolePzGenerator)
    first = generator.update()
    second = generator.update()

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.slow
def test_generator_pt_update_refreshes_acd_models(data_dir, acd_tracking_setup) -> None:
    """Updating pt refreshes both transport and driven optics inputs.

    The one-shot comparison receives the same explicit ``pt`` value.
    ``ModelDetails.pt`` is only the model probe coordinate; when the measurement
    offset is omitted, :func:`calculate_pz` estimates it from the data and then
    regenerates the model at that estimate. Supplying both values ensures the
    generator and one-shot paths reconstruct the same physical momentum.
    """
    tracking_df, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    model_details = ModelDetails(accelerator=driver.accelerator, pt=driver.pt)
    config = _config(bpm_upstream=bpm_up, bpm_downstream=bpm_dn)
    updated_pt = 1.0e-3

    generator = ACDipolePzGenerator.build(
        data=tracking_df,
        closed_orbit_at_zero=measured_zero_reference_for_simulation(tracking_df),
        orbit_mode="dynamic",
        model_details=model_details,
        config=config,
    )
    assert isinstance(generator, ACDipolePzGenerator)

    from_generator = generator.update(pt=updated_pt)
    one_shot = calculate_acd_pz(
        tracking_df,
        ModelDetails(
            accelerator=driver.accelerator,
            pt=updated_pt,
        ),
        config,
        closed_orbit_at_zero=measured_zero_reference_for_simulation(tracking_df),
        orbit_mode="dynamic",
    )

    pd.testing.assert_frame_equal(from_generator, one_shot)
    pd.testing.assert_frame_equal(from_generator.attrs["summary"], one_shot.attrs["summary"])
