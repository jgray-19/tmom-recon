"""Integration tests for :class:`tmom_recon.reconstruction.PzGenerator`."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.reference_co import measured_zero_reference_for_simulation
from tests.support.lhc import get_twiss, lhc_acd_barrier_s, lhc_model_details
from tmom_recon import PzGenerator, calculate_pz

pytest.importorskip("xtrack_tools")

pytestmark = [pytest.mark.lhc, pytest.mark.integration]


@pytest.mark.slow
def test_generator_update_matches_calculate_pz_and_accepts_bpm_subset(
    seq_b1,
    acd_tracking_setup,
) -> None:
    setup = acd_tracking_setup(seq_b1, delta_p=0.0, flattop_turns=100)
    tracking_df = setup.data
    tws = get_twiss(seq_b1, deltap=0.0)
    model_details = lhc_model_details(seq_b1, delta_p=0.0)
    barrier_s = lhc_acd_barrier_s(model_details.accelerator, model_details.pt)

    generator = calculate_pz(
        tracking_df,
        model_details,
        frame=measured_zero_reference_for_simulation(tracking_df),
        generator=True,
        barrier_s=barrier_s,
        info=False,
    )
    assert isinstance(generator, PzGenerator)

    from_generator = generator.update()
    one_shot = calculate_pz(
        tracking_df,
        model_details,
        frame=measured_zero_reference_for_simulation(tracking_df),
        barrier_s=barrier_s,
        info=False,
    )
    assert isinstance(one_shot, pd.DataFrame)

    pd.testing.assert_frame_equal(from_generator, one_shot)
    assert generator.latest is from_generator

    bpm_subset = [str(name) for name in tws.index[5:8]]
    subset = generator.update(bpm_names=bpm_subset)
    expected = one_shot[one_shot["name"].isin(bpm_subset)].reset_index(drop=True)

    pd.testing.assert_frame_equal(subset, expected)
    assert subset["name"].drop_duplicates().tolist() == bpm_subset
