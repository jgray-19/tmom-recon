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

pytest.importorskip("pymadng_utils")
pytest.importorskip("xtrack_tools")

from tmom_recon import ACDipoleConfig, ACDipolePzGenerator, calculate_pz

from .acd_test_helpers import AC_DIPOLE_ELEMENT, _ac_dipole_segment_around_element, _get_driver

SEQ_FILE = "lhcb1.seq"
DRIVEN_TUNES = (0.27, 0.322)


def _config(driver, *, bpm_upstream: str, bpm_downstream: str) -> ACDipoleConfig:
    return ACDipoleConfig(
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        model=driver,
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
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
    """gen.update(tws) is identical to a one-shot calculate_pz(acd_only=True)."""
    tracking_df, tws, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    config = _config(driver, bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    generator = calculate_pz(tracking_df, model_tws=tws, acd=config, acd_only=True, generator=True)
    assert isinstance(generator, ACDipolePzGenerator)
    assert generator.model is driver

    from_generator = generator.update(tws)
    one_shot = calculate_pz(tracking_df, model_tws=tws, acd=config, acd_only=True)

    pd.testing.assert_frame_equal(from_generator, one_shot)
    pd.testing.assert_frame_equal(from_generator.attrs["summary"], one_shot.attrs["summary"])
    assert generator.latest is from_generator


@pytest.mark.slow
def test_generator_repeated_update_is_deterministic(data_dir, acd_tracking_setup) -> None:
    """The frozen data means re-running with the same twiss is bit-for-bit stable."""
    tracking_df, tws, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    config = _config(driver, bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    generator = calculate_pz(tracking_df, model_tws=tws, acd=config, acd_only=True, generator=True)
    first = generator.update(tws)
    second = generator.update(tws)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.slow
def test_generator_tracks_optics_change(data_dir, acd_tracking_setup) -> None:
    """Updating with new optics changes the result and matches a fresh one-shot."""
    tracking_df, tws, driver, bpm_up, bpm_dn = _setup(data_dir, acd_tracking_setup)
    config = _config(driver, bpm_upstream=bpm_up, bpm_downstream=bpm_dn)

    perturbed = tws.copy(deep=True)
    perturbed["beta11"] = perturbed["beta11"].to_numpy(dtype=float) * 1.05
    perturbed["beta22"] = perturbed["beta22"].to_numpy(dtype=float) * 0.95

    generator = calculate_pz(tracking_df, model_tws=tws, acd=config, acd_only=True, generator=True)
    baseline = generator.update(tws)
    changed = generator.update(perturbed)

    # The optics change must move the reconstructed momenta.
    assert not baseline["px"].equals(changed["px"])

    # ...and it must equal a fresh one-shot run with the perturbed optics.
    fresh = calculate_pz(tracking_df, model_tws=perturbed, acd=config, acd_only=True)
    pd.testing.assert_frame_equal(changed, fresh)
