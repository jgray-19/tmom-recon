"""Integration tests for :class:`tmom_recon.reconstruction.ACDipolePzGenerator`.

The generator must give *exactly* the same result as a one-shot
``calculate_pz(..., acd_only=True)`` for the same optics (it shares the same
``reconstruct_from_prepared`` code path), must freeze the input data so repeated
updates are deterministic, and must track changes to the model optics.

These run the real MAD-NG driver, mirroring the other ACD integration tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.reference_co import zero_momentum_reference
from tmom_recon import ACDipoleConfig, ACDipolePzGenerator, ModelDetails, calculate_pz

from .acd_test_helpers import AC_DIPOLE_ELEMENT, _ac_dipole_segment_around_element, _get_driver

SEQ_FILE = "lhcb1.seq"
DRIVEN_TUNES = (0.27, 0.322)


def _model_details(driver, tws) -> ModelDetails:
    return ModelDetails(
        accelerator=driver.accelerator,
        pt=driver.pt,
    )


def _config(*, bpm_upstream: str, bpm_downstream: str) -> ACDipoleConfig:
    return ACDipoleConfig(
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        driven_tunes=DRIVEN_TUNES,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
    )


def _setup(data_dir, acd_tracking_setup):
    setup = acd_tracking_setup(SEQ_FILE, data_dir, delta_p=0.0, flattop_turns=100)
    tracking_df = setup["tracking_df"]
    tws = setup["tws"]
    driver = _get_driver(data_dir / "sequences" / SEQ_FILE, debug=False)
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        driver.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )
    return tracking_df, tws, driver, bpm_upstream, bpm_downstream


@pytest.mark.slow
def test_generator_update_matches_acd_only(data_dir, acd_tracking_setup) -> None:
    """gen.update() is identical to a one-shot calculate_pz(acd_only=True)."""
    tracking_df, tws, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    model_details = _model_details(driver, tws)
    config = _config(bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    generator = calculate_pz(
        tracking_df,
        reference=zero_momentum_reference(tracking_df),
        model_details=model_details,
        acd=config,
        acd_only=True,
        generator=True,
    )
    assert isinstance(generator, ACDipolePzGenerator)
    assert generator.model.accelerator is driver.accelerator

    from_generator = generator.update()
    one_shot = calculate_pz(
        tracking_df,
        reference=zero_momentum_reference(tracking_df),
        model_details=model_details,
        acd=config,
        acd_only=True,
    )

    pd.testing.assert_frame_equal(from_generator, one_shot)
    pd.testing.assert_frame_equal(from_generator.attrs["summary"], one_shot.attrs["summary"])
    assert generator.latest is from_generator


@pytest.mark.slow
def test_generator_repeated_update_is_deterministic(data_dir, acd_tracking_setup) -> None:
    """The frozen data means re-running with the same twiss is bit-for-bit stable."""
    tracking_df, tws, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    model_details = _model_details(driver, tws)
    config = _config(bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    generator = calculate_pz(
        tracking_df,
        reference=zero_momentum_reference(tracking_df),
        model_details=model_details,
        acd=config,
        acd_only=True,
        generator=True,
    )
    assert isinstance(generator, ACDipolePzGenerator)
    first = generator.update()
    second = generator.update()

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.slow
def test_generator_pt_update_refreshes_acd_models(data_dir, acd_tracking_setup) -> None:
    """Updating pt refreshes both transport and driven optics inputs."""
    tracking_df, tws, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    model_details = _model_details(driver, tws)
    config = _config(bpm_upstream=bpm_up, bpm_downstream=bpm_dn)
    updated_pt = 1.0e-3

    generator = calculate_pz(
        tracking_df,
        reference=zero_momentum_reference(tracking_df),
        model_details=model_details,
        acd=config,
        acd_only=True,
        generator=True,
    )
    assert isinstance(generator, ACDipolePzGenerator)

    from_generator = generator.update(measurement_pt=updated_pt)
    one_shot = calculate_pz(
        tracking_df,
        reference=zero_momentum_reference(tracking_df),
        model_details=ModelDetails(
            accelerator=driver.accelerator,
            pt=updated_pt,
        ),
        acd=config,
        acd_only=True,
    )

    pd.testing.assert_frame_equal(from_generator, one_shot)
    pd.testing.assert_frame_equal(from_generator.attrs["summary"], one_shot.attrs["summary"])
