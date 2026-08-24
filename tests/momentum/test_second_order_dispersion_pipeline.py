"""Second-order dispersion end to end through `calculate_pz`, without the AC dipole.

`resolve_model_details` runs the model twiss with `chrom=True`, so `ddx`/`ddpx`
reach both the pt estimate and the neighbour-pair momentum formula. This test
pins that the columns actually arrive and that they matter: off momentum, the
first-order fallback biases pt by ~2e-3 relative at dp/p = 8e-3, which the
second-order solve removes. Momentum reconstruction is checked separately for
finiteness and a broad regression bound; its neighbour fit is not expected to
improve monotonically with the dispersion order used by the orbit estimator.

The comparison is run at a single dp/p because each point costs a full tracking
run; the broader sweep is intentionally kept out of the test suite.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.psb_tracking import ACD_ELEMENT
from tests.support.acd_barrier import acd_barrier_s
from tests.support.assertions import merge_tracking_truth, rmse
from tests.support.truth import simulated_mixed_reference_from_model
from tmom_recon import ModelDetails, calculate_pz, reconstruction
from tmom_recon.model import resolve_model_details

pytestmark = [pytest.mark.psb, pytest.mark.integration]

DELTA_P = 8e-3
SECOND_ORDER_COLUMNS = ("ddx", "ddpx", "ddy", "ddpy")


def _reconstruct(tracking_df: pd.DataFrame, model, *, second_order: bool) -> pd.DataFrame:
    """Run the plain reconstruction, optionally stripped back to first order."""
    original = reconstruction.resolve_model_details
    nominal_details = ModelDetails(accelerator=model.accelerator, pt=0.0)

    def without_second_order(*args, **kwargs):
        resolved = original(*args, **kwargs)
        return replace(
            resolved,
            tws=resolved.tws.drop(columns=list(SECOND_ORDER_COLUMNS), errors="ignore"),
        )

    if not second_order:
        reconstruction.resolve_model_details = without_second_order
    try:
        df = calculate_pz(
            tracking_df.copy(deep=True),
            # pt=0: the model is built on momentum, so the reconstruction has to
            # recover the beam's momentum from the orbit rather than be told it.
            nominal_details,
            frame=simulated_mixed_reference_from_model(nominal_details, tracking_df),
            use_dispersion=True,
            barrier_s=acd_barrier_s(model, ACD_ELEMENT),
            info=False,
        )
        assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
        return df
    finally:
        reconstruction.resolve_model_details = original


@pytest.mark.slow
def test_generated_model_twiss_carries_second_order_dispersion(psb_tracking_setup) -> None:
    """`chrom=True` must reach the twiss `calculate_pz` actually reconstructs against.

    Note this is `resolve_model_details`' twiss, not the one the tracking setup
    builds for its own bookkeeping -- the latter is not what the pipeline uses.
    """
    model = psb_tracking_setup(0.0).machine.madng_model
    resolved = resolve_model_details(ModelDetails(accelerator=model.accelerator, pt=0.0))
    missing = [col for col in SECOND_ORDER_COLUMNS if col not in resolved.tws.columns]
    assert not missing, f"model twiss is missing chrom columns {missing}"


@pytest.mark.slow
def test_second_order_dispersion_improves_pt_off_momentum(psb_tracking_setup) -> None:
    setup = psb_tracking_setup(DELTA_P)
    tracking_df = setup.measurement.data
    physical_tracking_df = tracking_df.loc[
        tracking_df["name"].isin(setup.measurement.bpm_names)
    ].copy()
    pt_true = setup.measurement.pt

    results = {
        label: _reconstruct(
            physical_tracking_df, setup.machine.madng_model, second_order=second_order
        )
        for label, second_order in (("second", True), ("first", False))
    }
    errors = {}
    for label, result in results.items():
        merged = merge_tracking_truth(physical_tracking_df, result)
        finite = np.isfinite(merged[["px", "py"]]).all(axis=1)
        assert finite.any(), f"{label} reconstruction produced no finite BPM momenta"
        merged = merged.loc[finite]
        errors[label] = {
            "pt_rel": abs(result.attrs["PT_EST"] - pt_true) / abs(pt_true),
            "px_rmse": rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy()),
        }

    # First order leaves a pt bias of order pt*ddx/dx, which grows with dp/p.
    assert errors["first"]["pt_rel"] > 1e-3
    # Second order removes it almost entirely: ~180x at this dp/p.
    assert errors["second"]["pt_rel"] < 1e-4
    assert errors["second"]["pt_rel"] < errors["first"]["pt_rel"] / 50.0

    # The migrated physical-BPM-only input still produces usable momenta. The
    # neighbour fit and the orbit estimator exercise different approximations,
    # so their errors are not required to improve monotonically together.
    assert errors["second"]["px_rmse"] < 1e-5


@pytest.mark.slow
def test_second_order_dispersion_changes_nothing_on_momentum(psb_tracking_setup) -> None:
    """At pt = 0 the pt**2 terms vanish, so both paths must agree exactly."""
    setup = psb_tracking_setup(0.0)
    tracking_df = setup.measurement.data
    tracking_df = tracking_df.loc[tracking_df["name"].isin(setup.measurement.bpm_names)].copy()

    second = _reconstruct(tracking_df, setup.machine.madng_model, second_order=True)
    first = _reconstruct(tracking_df, setup.machine.madng_model, second_order=False)

    # pt is ~1e-9 here rather than exactly zero, so compare absolutely: a
    # relative tolerance on a number that small is meaningless.
    assert second.attrs["PT_EST"] == pytest.approx(first.attrs["PT_EST"], abs=1e-15)
    for col in ("px", "py"):
        got, want = second[col].to_numpy(), first[col].to_numpy()
        # Ring-edge and marker rows have no neighbour pair and stay NaN in both.
        finite = np.isfinite(want)
        assert np.array_equal(finite, np.isfinite(got))
        assert got[finite] == pytest.approx(want[finite], rel=1e-9, abs=1e-15)
