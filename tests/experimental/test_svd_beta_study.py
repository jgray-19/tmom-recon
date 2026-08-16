"""Experimental study: weighted vs plain SVD cleaning on AC-dipole optics.

These tests are slow and depend on the optional
``xtrack``/``xtrack_tools``/``pymadng_utils``/``omc3`` stack (and AFS
``acc-models``), so they are skipped when anything is missing. They are not part
of the fast gate; run explicitly and watch stdout for the verdict::

    pytest tests/experimental -s -m slow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xtrack")
pytest.importorskip("xtrack_tools")
pytest.importorskip("pymadng_utils")
pytest.importorskip("omc3")
pytest.importorskip("turn_by_turn")

from tmom_recon.experimental.svd_beta_study import (  # noqa: E402
    ACDScene,
    StudyConfig,
    add_noise,
    build_acd_scene,
    clean,
    prepare_model_dir,
    run_hole_in_one,
    run_study,
    summarise,
)


# --------------------------------------------------------------------------- #
# Fast, self-contained unit check (no tracking / harpy)
# --------------------------------------------------------------------------- #
def test_add_noise_homoscedastic_makes_weighting_a_noop():
    """With equal per-BPM noise, weighted and plain SVD must coincide.

    This is *why* the study uses a noise spread: uniform noise makes the
    weighting cancel, leaving nothing to compare.
    """
    from tmom_recon.svd import svd_clean_measurements, weighted_svd_clean_measurements

    rng = np.random.default_rng(0)
    names = [f"BPM{i}" for i in range(12)]
    turns = np.arange(64)
    mode = np.outer(np.cos(2 * np.pi * 0.31 * turns), rng.normal(size=len(names)))
    tbt = pd.DataFrame(
        {
            "name": np.repeat(names, len(turns)),
            "turn": np.tile(turns, len(names)),
            "x": mode.T.ravel(),
            "y": mode.T.ravel(),
        }
    )
    noisy = add_noise(tbt, sigma=1e-3, seed=1, sigma_spread=0.0)
    plain = svd_clean_measurements(noisy, rank=2).sort_values(["name", "turn"])
    weighted = weighted_svd_clean_measurements(noisy, rank=2).sort_values(["name", "turn"])
    assert np.allclose(plain["x"], weighted["x"], atol=1e-12)


# --------------------------------------------------------------------------- #
# Slow end-to-end study
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def scene(tmp_path_factory) -> ACDScene:
    workdir = tmp_path_factory.mktemp("svd_beta_study")
    try:
        model_dir = prepare_model_dir(workdir)
    except Exception as exc:  # noqa: BLE001  # acc-models / MAD-NG not available here
        pytest.skip(f"model creation unavailable: {exc}")
    return build_acd_scene(model_dir, rel_k1_std=5e-4, perturb_seed=1, workdir=workdir)


@pytest.mark.slow
def test_weighting_is_noop_under_uniform_noise_endtoend(scene: ACDScene):
    """Sanity: uniform noise -> svd and weighted produce the same beta tables."""
    noisy = add_noise(scene.flat_tbt, sigma=3e-4, seed=0, sigma_spread=0.0)
    betas_svd = run_hole_in_one(scene, clean(noisy, "svd"), "svd", tag="uni_svd")
    betas_w = run_hole_in_one(scene, clean(noisy, "weighted"), "weighted", tag="uni_weighted")
    for plane in ("x", "y"):
        col = f"DELTABET{plane.upper()}"
        a = betas_svd[plane]["phase"][col]
        b = betas_w[plane]["phase"].reindex(a.index)[col]
        assert np.allclose(a.dropna(), b.reindex(a.dropna().index), rtol=1e-4, atol=1e-4)


@pytest.mark.slow
def test_study_runs_and_reports(scene: ACDScene):
    """Run the full comparison under heteroscedastic noise and print the verdict."""
    config = StudyConfig(
        workdir=scene.workdir,
        sigma=3e-4,
        sigma_spread=2.0,
        noise_seeds=(0, 1, 2),
        scene=scene,
    )
    results = run_study(config)
    summary = summarise(results)

    print("\n=== per-method mean DELTABET error vs truth (lower is better) ===")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary)

    assert not summary.isna().to_numpy().any()
    assert set(summary.index) >= {"none", "svd", "weighted"}
