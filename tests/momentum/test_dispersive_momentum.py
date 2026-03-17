"""Integration tests for dispersive momentum reconstruction using xtrack data."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest
from xtrack_tools.acd import run_acd_track

from tmom_recon import ACDipoleConfig, inject_noise_xy_inplace
from tmom_recon import calculate_dispersive_pz as dispersive_calc
from tmom_recon import calculate_transverse_pz as transverse_calc
from tmom_recon.svd import svd_clean_measurements  # noqa: E402

from .acd_test_helpers import (
    AC_DIPOLE_ELEMENT,
    _ac_dipole_segment_around_element,
    _full_xsuite_to_ngtws,
    _get_driver,
)
from .momentum_test_utils import get_truth, rmse, xsuite_to_ngtws


@cache
def _cached_tracking_setup(
    seq_path: str,
    json_path: str,
    delta_p: float,
    ramp_turns: int,
    flattop_turns: int,
):
    tracking_df, tws, _baseline_line = run_acd_track(
        json_path=Path(json_path),
        sequence_file=Path(seq_path),
        delta_p=delta_p,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
    )
    tws = xsuite_to_ngtws(tws)
    truth = get_truth(tracking_df, tws)
    return tracking_df, tws, truth


def _get_tracking_setup(seq: Path, json_path: Path, delta_p: float) -> tuple:
    tracking_df, tws, truth = _cached_tracking_setup(
        str(seq),
        str(json_path),
        delta_p,
        1000,
        100,
    )
    return tracking_df.copy(deep=True), tws.copy(deep=True), truth.copy(deep=True)


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
def test_dispersive_momentum_on_momentum(seq_file, data_dir, xsuite_json_path):
    """Test dispersive momentum reconstruction for on-momentum beam.

    For on-momentum particles (δp=0), dispersive and transverse methods
    should produce nearly identical results.
    """
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)
    tracking_df, tws, truth = _get_tracking_setup(seq, json_path, 0.0)

    # Transverse reconstruction (baseline)
    trans_result = transverse_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_trans", "py": "py_trans"})

    # Dispersive reconstruction
    disp_result = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_disp", "py": "py_disp"})

    # Merge results
    merged = truth.merge(
        trans_result[["name", "turn", "px_trans", "py_trans"]],
        on=["name", "turn"],
    ).merge(
        disp_result[["name", "turn", "px_disp", "py_disp"]],
        on=["name", "turn"],
    )

    assert len(merged) == len(truth)

    # Compute RMSE for both methods
    px_rmse_trans = rmse(merged["px_true"].to_numpy(), merged["px_trans"].to_numpy())
    py_rmse_trans = rmse(merged["py_true"].to_numpy(), merged["py_trans"].to_numpy())
    px_rmse_disp = rmse(merged["px_true"].to_numpy(), merged["px_disp"].to_numpy())
    py_rmse_disp = rmse(merged["py_true"].to_numpy(), merged["py_disp"].to_numpy())

    # For on-momentum, both methods should give reasonable results
    # Note: Using driven motion with AC dipole introduces some systematic offset
    # from the natural optics model, so tolerances are relaxed
    assert px_rmse_trans < 3.6e-7, f"Transverse px RMSE {px_rmse_trans:.2e} > 3.6e-7"
    assert py_rmse_trans < 3e-7, f"Transverse py RMSE {py_rmse_trans:.2e} > 3e-7"
    assert px_rmse_disp < 3.2e-7, f"Dispersive px RMSE {px_rmse_disp:.2e} > 3.2e-7"
    assert py_rmse_disp < 3e-7, f"Dispersive py RMSE {py_rmse_disp:.2e} > 3e-7"

    # Both methods should be equivalent for on-momentum (dispersive <= transverse)
    assert px_rmse_disp < px_rmse_trans or np.isclose(px_rmse_disp, px_rmse_trans, rtol=1e-2), (
        f"Dispersive px RMSE {px_rmse_disp:.2e} > transverse {px_rmse_trans:.2e}"
    )
    assert py_rmse_disp < py_rmse_trans or np.isclose(py_rmse_disp, py_rmse_trans, rtol=1e-2), (
        f"Dispersive py RMSE {py_rmse_disp:.2e} > transverse {py_rmse_trans:.2e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
def test_dispersive_momentum_on_momentum_with_ac_dipole_config(
    seq_file,
    data_dir,
    xsuite_json_path,
):
    pytest.importorskip("pymadng_utils")

    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)
    tracking_df, tws_xsuite, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq,
        delta_p=0.0,
        ramp_turns=1000,
        flattop_turns=100,
    )
    tws = xsuite_to_ngtws(tws_xsuite)
    truth = get_truth(tracking_df, tws)
    full_tws = baseline_line.twiss(method="4d")
    full_ng_tws = _full_xsuite_to_ngtws(full_tws)

    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        full_tws,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )
    model = _get_driver(seq, deltap=0.0)

    baseline = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=full_ng_tws,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_base", "py": "py_base"})

    with_acd = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=full_ng_tws,
        inject_noise=False,
        info=False,
        ac_dipole_config=ACDipoleConfig(
            ac_dipole_marker=AC_DIPOLE_ELEMENT,
            model=model,
            bpm_upstream=bpm_upstream,
            bpm_downstream=bpm_downstream,
        ),
    ).rename(columns={"px": "px_acd", "py": "py_acd"})

    merged = truth.merge(
        baseline[["name", "turn", "px_base", "py_base"]],
        on=["name", "turn"],
    ).merge(
        with_acd[["name", "turn", "px_acd", "py_acd"]],
        on=["name", "turn"],
    )

    px_rmse_base = rmse(merged["px_true"].to_numpy(), merged["px_base"].to_numpy())
    py_rmse_base = rmse(merged["py_true"].to_numpy(), merged["py_base"].to_numpy())
    px_rmse_acd = rmse(merged["px_true"].to_numpy(), merged["px_acd"].to_numpy())
    py_rmse_acd = rmse(merged["py_true"].to_numpy(), merged["py_acd"].to_numpy())

    # Global quality should not be worse when ACD corrections are enabled.
    assert px_rmse_acd <= px_rmse_base
    assert py_rmse_acd <= py_rmse_base

    improved_bpms = {bpm_upstream, bpm_downstream}
    improved_rows = merged[merged["name"].isin(improved_bpms)]
    assert len(improved_rows) > 0

    px_rmse_base_local = rmse(
        improved_rows["px_true"].to_numpy(),
        improved_rows["px_base"].to_numpy(),
    )
    py_rmse_base_local = rmse(
        improved_rows["py_true"].to_numpy(),
        improved_rows["py_base"].to_numpy(),
    )
    px_rmse_acd_local = rmse(
        improved_rows["px_true"].to_numpy(),
        improved_rows["px_acd"].to_numpy(),
    )
    py_rmse_acd_local = rmse(
        improved_rows["py_true"].to_numpy(),
        improved_rows["py_acd"].to_numpy(),
    )

    # At the BPMs near the AC dipole, corrected estimates should improve.
    assert px_rmse_acd_local < px_rmse_base_local
    assert py_rmse_acd_local < py_rmse_base_local


@pytest.mark.slow
@pytest.mark.parametrize("seq_file", ["lhcb1.seq", "b1_120cm_crossing.seq"])
@pytest.mark.parametrize("delta_p", [-5e-4, 4e-4])
def test_dispersive_momentum_off_momentum_cases(seq_file, delta_p, data_dir, xsuite_json_path):
    """Validate off-momentum dispersive momentum reconstruction for clean, noisy, and SVD-cleaned data.

    This test uses a single tracked off-momentum beam (non-zero δp) and performs:

    * A baseline transverse reconstruction (no dispersion model) used for comparison.
    * A clean dispersive reconstruction (no injected noise) to validate the off-momentum
      dispersive method against the tracking truth and the transverse baseline.
    * A noisy dispersive reconstruction, where realistic BPM-like noise is injected into
      the measured coordinates to assess the degradation in reconstruction quality.
    * An SVD-cleaned dispersive reconstruction, where the noisy measurements are first
      passed through ``svd_clean_measurements`` to verify that SVD cleaning recovers
      performance close to the clean dispersive case and improves over the raw noisy case.

    The merged results allow comparison between transverse and dispersive methods and
    between clean, noisy, and SVD-cleaned dispersive reconstructions for each
    sequence and δp value.
    """
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)
    tracking_df, tws, truth = _get_tracking_setup(seq, json_path, delta_p)

    trans_result = transverse_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_trans", "py": "py_trans"})

    # Clean reconstruction (no noise)
    clean_result = dispersive_calc(
        tracking_df.copy(deep=True),
        tws=tws,
        inject_noise=False,
        info=True,
    ).rename(columns={"px": "px_clean", "py": "py_clean"})

    # Noisy reconstruction - inject noise manually then calculate
    rng = np.random.default_rng(42)
    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(noisy_df, tracking_df, rng)
    noisy_result = dispersive_calc(
        noisy_df,
        tws=tws,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_noisy", "py": "py_noisy"})

    # SVD cleaned reconstruction - apply SVD to noisy data
    cleaned_df = svd_clean_measurements(noisy_df)
    svd_result = dispersive_calc(
        cleaned_df,
        tws=tws,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_svd", "py": "py_svd"})

    # Merge all results
    merged = (
        truth.merge(
            trans_result[["name", "turn", "px_trans", "py_trans"]],
            on=["name", "turn"],
        )
        .merge(
            clean_result[["name", "turn", "px_clean", "py_clean"]],
            on=["name", "turn"],
        )
        .merge(
            noisy_result[["name", "turn", "px_noisy", "py_noisy"]],
            on=["name", "turn"],
        )
        .merge(
            svd_result[["name", "turn", "px_svd", "py_svd"]],
            on=["name", "turn"],
        )
    )

    assert len(merged) == len(truth)

    # Compute RMSE
    px_rmse_trans = rmse(merged["px_true"].to_numpy(), merged["px_trans"].to_numpy())
    py_rmse_trans = rmse(merged["py_true"].to_numpy(), merged["py_trans"].to_numpy())
    px_rmse_nonoise = rmse(merged["px_true"].to_numpy(), merged["px_clean"].to_numpy())
    py_rmse_nonoise = rmse(merged["py_true"].to_numpy(), merged["py_clean"].to_numpy())
    px_rmse_noisy = rmse(merged["px_true"].to_numpy(), merged["px_noisy"].to_numpy())
    py_rmse_noisy = rmse(merged["py_true"].to_numpy(), merged["py_noisy"].to_numpy())
    px_rmse_cleaned = rmse(merged["px_true"].to_numpy(), merged["px_svd"].to_numpy())
    py_rmse_cleaned = rmse(merged["py_true"].to_numpy(), merged["py_svd"].to_numpy())

    # Clean off-momentum behaviour should still beat the transverse baseline.
    assert py_rmse_trans < 3e-7, f"Transverse py RMSE {py_rmse_trans:.2e} > 2e-7"
    assert py_rmse_nonoise < 3e-7, f"Dispersive py RMSE {py_rmse_nonoise:.2e} > 2e-7"
    assert px_rmse_nonoise <= px_rmse_trans / 11, (
        f"Dispersive px RMSE {px_rmse_nonoise:.2e} should be <= transverse {px_rmse_trans:.2e}"
    )
    tol = 6e-6 if "crossing" not in seq_file else 7.2e-6
    assert px_rmse_nonoise < 5e-7, f"Dispersive px RMSE {px_rmse_nonoise:.2e} > 5e-7"
    assert px_rmse_trans < tol, f"Transverse px RMSE {px_rmse_trans:.2e} > {tol:.2e}"

    # Check clean reconstruction quality
    assert px_rmse_nonoise < 5e-7, f"No noise px RMSE {px_rmse_nonoise:.2e} should be < 3.5e-7"
    assert py_rmse_nonoise < 3e-7, f"No noise py RMSE {py_rmse_nonoise:.2e} should be < 3e-7"

    # Check noisy is worse than clean
    assert px_rmse_noisy > px_rmse_nonoise, (
        f"Noisy px RMSE {px_rmse_noisy:.2e} should be > clean {px_rmse_nonoise:.2e}"
    )
    assert py_rmse_noisy > py_rmse_nonoise, (
        f"Noisy py RMSE {py_rmse_noisy:.2e} should be > clean {py_rmse_nonoise:.2e}"
    )

    # Check SVD cleaned is better than noisy
    assert px_rmse_cleaned < px_rmse_noisy, (
        f"SVD px RMSE {px_rmse_cleaned:.2e} should be < noisy {px_rmse_noisy:.2e}"
    )
    assert py_rmse_cleaned < py_rmse_noisy, (
        f"SVD py RMSE {py_rmse_cleaned:.2e} should be < noisy {py_rmse_noisy:.2e}"
    )

    # Check SVD cleaned has acceptable absolute tolerance
    assert px_rmse_cleaned < 5e-7, f"SVD px RMSE {px_rmse_cleaned:.2e} should be < 5e-7"
    assert py_rmse_cleaned < 4e-7, f"SVD py RMSE {py_rmse_cleaned:.2e} should be < 4e-7"
