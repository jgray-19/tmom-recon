from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tfs
from pymadng_utils.accelerators import LHC
from pymadng_utils.mad import AcceleratorMadInterface
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles, run_acd_track
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

import tmom_recon.nbpm.reconstruction as nbpm_reconstruction_module
from tests.momentum.momentum_test_utils import get_truth, rmse
from tmom_recon import ACDipoleConfig
from tmom_recon.nbpm import (
    LHC_MAINFIELD_PATTERNS,
    BpmPair,
    build_bpm_longitudinal_covariance,
    build_bpm_s_response_matrix,
    build_dp_dphi,
    build_effective_k1_sources,
    build_measurement_jacobian,
    build_momentum_covariance,
    build_pair_catalog,
    build_phase_response_matrix,
    build_TK_block,
    build_Ts_block,
    calculate_transverse_pz_nbpm,
    combine_momentum_blue,
    diagnose_covariance_model,
    evaluate_momentum_estimates,
)
from tmom_recon.physics.transverse import calculate_pz as calculate_transverse_pz
from tmom_recon.physics.transverse import inject_noise_xy_inplace
from tmom_recon.svd import svd_clean_measurements

SEQ_FILE = "lhcb1.seq"


def _make_twiss_bpms() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "BPM.3", "BPM.4"],
            "s": [0.0, 10.0, 20.0, 30.0],
            "beta11": [100.0, 110.0, 120.0, 130.0],
            "beta22": [90.0, 95.0, 100.0, 105.0],
            "alfa11": [0.1, 0.2, -0.1, 0.05],
            "alfa22": [0.0, 0.1, -0.2, 0.15],
            "mu1": [0.00, 0.08, 0.16, 0.24],
            "mu2": [0.00, 0.07, 0.15, 0.23],
        }
    ).set_index("name")


def _make_twiss_elements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["MQXA.1", "MS.1", "MQY.1", "DRIFT.1"],
            "s": [5.0, 15.0, 25.0, 35.0],
            "beta11": [80.0, 70.0, 75.0, 60.0],
            "beta22": [85.0, 72.0, 76.0, 62.0],
            "mu1": [0.04, 0.12, 0.20, 0.28],
            "mu2": [0.035, 0.11, 0.19, 0.27],
            "k1l": [0.02, 0.0, 0.03, 0.0],
            "k2l": [0.0, 0.5, 0.0, 0.0],
        }
    )


def _xsuite_to_full_ng_tws(tbl) -> pd.DataFrame:
    df = tbl.to_pandas()
    df["beta11"] = df["betx"]
    df["beta22"] = df["bety"]
    df["alfa11"] = df["alfx"]
    df["alfa22"] = df["alfy"]
    df["mu1"] = df["mux"]
    df["mu2"] = df["muy"]
    df["name"] = df["name"].astype(str).str.upper()
    return tfs.TfsDataFrame(df, headers={"q1": tbl.qx, "q2": tbl.qy}).set_index("name")


def _select_local_bpm_window(
    tracking_df: pd.DataFrame,
    tws_bpm: pd.DataFrame,
    *,
    window_size: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    bpm_mask = tws_bpm.index.str.match(r"^BPM.*\.B1$")
    tws_bpms = tws_bpm.loc[bpm_mask].copy(deep=True)
    mid_start = max(0, len(tws_bpms) // 2 - window_size // 2)
    selected_bpms = tws_bpms.index[mid_start : mid_start + window_size]
    return (
        tracking_df[tracking_df["name"].isin(selected_bpms)].copy(deep=True),
        tws_bpms.loc[selected_bpms].copy(deep=True),
        selected_bpms,
    )


def _evaluate_nbpm_vs_baseline(
    truth: pd.DataFrame,
    baseline: pd.DataFrame,
    nbpm: pd.DataFrame,
    *,
    selected_bpms: pd.Index,
) -> tuple[pd.DataFrame, float, float, float, float]:
    merged = truth.merge(
        baseline[["name", "turn", "px_base", "py_base"]],
        on=["name", "turn"],
        how="inner",
    ).merge(
        nbpm[["name", "turn", "px_nbpm", "py_nbpm"]],
        on=["name", "turn"],
        how="inner",
    )
    merged = merged[merged["name"].isin(selected_bpms)]
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["px_true", "py_true", "px_base", "py_base", "px_nbpm", "py_nbpm"]
    )

    px_rmse_base = rmse(merged["px_true"].to_numpy(), merged["px_base"].to_numpy())
    py_rmse_base = rmse(merged["py_true"].to_numpy(), merged["py_base"].to_numpy())
    px_rmse_nbpm = rmse(merged["px_true"].to_numpy(), merged["px_nbpm"].to_numpy())
    py_rmse_nbpm = rmse(merged["py_true"].to_numpy(), merged["py_nbpm"].to_numpy())
    return merged, px_rmse_base, py_rmse_base, px_rmse_nbpm, py_rmse_nbpm


def _get_setup_with_magnetic_errors(
    data_dir,
    *,
    flattop_turns: int = 40,
    magnet_seed: int = 13,
    rel_k1_std_dev: float = 1e-4,
):
    sequence_file = data_dir / "sequences" / SEQ_FILE
    accelerator = LHC(beam=1, sequence_file=sequence_file, pc=6800)
    mad = AcceleratorMadInterface(accelerator)

    mad.observe()
    _tws = mad.run_twiss()
    mad.unobserve_elements(["BPM"])
    magnet_strengths, _ = mad.apply_magnet_perturbations(
        rel_error=rel_k1_std_dev,
        seed=magnet_seed,
        magnet_type="all",
    )
    assert magnet_strengths

    matched_tunes = mad.perform_orbit_correction(
        machine_deltap=0.0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=None,
    )
    env = initialise_env(
        matched_tunes,
        magnet_strengths,
        tfs.TfsDataFrame(columns=["name", "kind", "knl", "ksl"]),
        sequence_file=sequence_file,
        seq_name="lhcb1",
        strict_set=False,
    )
    baseline_line = env["lhcb1"].copy()
    full_tws = baseline_line.twiss(method="4d")
    monitored_line = run_ac_dipole_tracking_with_particles(
        line=baseline_line,
        tws=full_tws,
        beam=1,
        ramp_turns=1000,
        flattop_turns=flattop_turns,
        driven_tunes=[0.27, 0.322],
        bpm_pattern=r"(?i)bpm.*",
        particle_coords={
            "x": [0.0],
            "px": [0.0],
            "y": [0.0],
            "py": [0.0],
            "delta": [0.0],
        },
    )
    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=1000,
        flattop_turns=flattop_turns,
        add_variance_columns=True,
    )
    tracking_df["name"] = tracking_df["name"].astype(str).str.upper()
    tracking_df["turn"] = tracking_df["turn"].astype(int)
    ng_tws = _xsuite_to_full_ng_tws(full_tws)
    truth = get_truth(tracking_df, ng_tws)
    return tracking_df, ng_tws, truth, _xsuite_to_full_ng_tws(full_tws)


def _make_pairs() -> list[BpmPair]:
    return [
        BpmPair(
            pair_id="p1",
            plane="x",
            branch="next",
            eta=1,
            i_bpm_name="BPM.1",
            j_bpm_name="BPM.3",
            i_bpm_idx=0,
            j_bpm_idx=2,
            phi_model_rad=0.02 * np.pi,
            beta_i=100.0,
            beta_j=120.0,
            alpha_i=0.1,
            u_i=1.4e-3,
            u_j=-0.9e-3,
            D_i=1.5,
            D_j=1.7,
            Dp_i=0.12,
            delta=2e-4,
            s_i=0.0,
            s_j=20.0,
            mu_i=0.00,
            mu_j=0.16,
            measurement_id_i="BPM.1:0:x",
            measurement_id_j="BPM.3:0:x",
            var_u_i=1e-8,
            var_u_j=1e-8,
        ),
        BpmPair(
            pair_id="p2",
            plane="x",
            branch="next",
            eta=1,
            i_bpm_name="BPM.1",
            j_bpm_name="BPM.4",
            i_bpm_idx=0,
            j_bpm_idx=3,
            phi_model_rad=-0.02 * np.pi,
            beta_i=100.0,
            beta_j=130.0,
            alpha_i=0.1,
            u_i=1.4e-3,
            u_j=-0.4e-3,
            D_i=1.5,
            D_j=1.8,
            Dp_i=0.12,
            delta=2e-4,
            s_i=0.0,
            s_j=30.0,
            mu_i=0.00,
            mu_j=0.24,
            measurement_id_i="BPM.1:0:x",
            measurement_id_j="BPM.4:0:x",
            var_u_i=1e-8,
            var_u_j=1e-8,
        ),
        BpmPair(
            pair_id="p3",
            plane="x",
            branch="prev",
            eta=-1,
            i_bpm_name="BPM.4",
            j_bpm_name="BPM.2",
            i_bpm_idx=3,
            j_bpm_idx=1,
            phi_model_rad=0.01 * np.pi,
            beta_i=130.0,
            beta_j=110.0,
            alpha_i=0.05,
            u_i=-0.4e-3,
            u_j=0.7e-3,
            D_i=1.8,
            D_j=1.6,
            Dp_i=0.10,
            delta=2e-4,
            s_i=30.0,
            s_j=10.0,
            mu_i=0.24,
            mu_j=0.08,
            measurement_id_i="BPM.4:0:x",
            measurement_id_j="BPM.2:0:x",
            var_u_i=1e-8,
            var_u_j=1e-8,
        ),
    ]


def test_build_effective_k1_sources_uses_lhc_pattern_table() -> None:
    sources = build_effective_k1_sources(
        _make_twiss_elements(), optics_patterns=LHC_MAINFIELD_PATTERNS
    )

    assert set(sources["elem_name"]) == {"MQXA.1", "MS.1", "MQY.1"}

    mqxa = sources.set_index("elem_name").loc["MQXA.1"]
    assert np.isclose(mqxa["sigma_k1_quad"], abs(0.02) * 10e-4)
    assert np.isclose(mqxa["sigma_k1_long"], abs(0.02) * 0.003)

    sext = sources.set_index("elem_name").loc["MS.1"]
    assert np.isclose(sext["sigma_k1_sext"], abs(0.5) * 0.003)
    assert np.isclose(
        sext["sigma_k1_eff"] ** 2,
        sext["sigma_k1_quad"] ** 2 + sext["sigma_k1_sext"] ** 2 + sext["sigma_k1_long"] ** 2,
    )


def test_nbpm_pair_catalog_rejects_near_singular_phases() -> None:
    tws_bpm = pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "BPM.3"],
            "s": [0.0, 10.0, 20.0],
            "beta11": [100.0, 110.0, 120.0],
            "beta22": [90.0, 95.0, 100.0],
            "alfa11": [0.1, 0.2, -0.1],
            "alfa22": [0.0, 0.1, -0.2],
            "mu1": [0.00, 0.50, 0.24],
            "mu2": [0.00, 0.20, 0.23],
        }
    ).set_index("name")
    tws_bpm.attrs["q1"] = 1.0
    tws_bpm.attrs["q2"] = 1.0

    catalog = build_pair_catalog(tws_bpm, plane="x", max_bpm_distance=11, min_abs_cos=0.05)
    bpm1_neighbors = {row["j_bpm_name"] for row in catalog["BPM.1"]}

    assert "BPM.2" not in bpm1_neighbors
    assert "BPM.3" in bpm1_neighbors


def test_nbpm_pair_catalog_rejects_pairs_crossing_ac_dipole_marker() -> None:
    tws_bpm = pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "BPM.3", "BPM.4"],
            "s": [0.0, 10.0, 20.0, 30.0],
            "beta11": [100.0, 110.0, 120.0, 130.0],
            "beta22": [90.0, 95.0, 100.0, 105.0],
            "alfa11": [0.1, 0.2, -0.1, 0.05],
            "alfa22": [0.0, 0.1, -0.2, 0.15],
            "mu1": [0.00, 0.08, 0.16, 0.24],
            "mu2": [0.00, 0.07, 0.15, 0.23],
        }
    ).set_index("name")
    tws_bpm.attrs["q1"] = 1.0
    tws_bpm.attrs["q2"] = 1.0

    twiss_elements = pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "ACD.TEST", "BPM.3", "BPM.4"],
            "s": [0.0, 10.0, 15.0, 20.0, 30.0],
        }
    ).set_index("name")

    catalog = build_pair_catalog(
        tws_bpm,
        twiss_elements=twiss_elements,
        plane="x",
        max_bpm_distance=11,
        min_abs_cos=0.05,
        excluded_element_name="ACD.TEST",
    )
    bpm2_next_neighbors = {row["j_bpm_name"] for row in catalog["BPM.2"] if row["branch"] == "next"}
    bpm2_prev_neighbors = {row["j_bpm_name"] for row in catalog["BPM.2"] if row["branch"] == "prev"}

    assert "BPM.3" not in bpm2_next_neighbors
    assert "BPM.4" not in bpm2_next_neighbors
    assert "BPM.1" in bpm2_prev_neighbors
    assert "BPM.3" in bpm2_prev_neighbors


def test_nbpm_pair_catalog_can_include_crossing_pairs_when_kicks_are_known() -> None:
    tws_bpm = _make_twiss_bpms()
    tws_bpm.attrs["q1"] = 1.0
    tws_bpm.attrs["q2"] = 1.0

    twiss_elements = pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "ACD.TEST", "BPM.3", "BPM.4"],
            "s": [0.0, 10.0, 15.0, 20.0, 30.0],
        }
    ).set_index("name")

    catalog = build_pair_catalog(
        tws_bpm,
        twiss_elements=twiss_elements,
        plane="x",
        max_bpm_distance=11,
        min_abs_cos=0.05,
        excluded_element_name="ACD.TEST",
        allow_excluded_crossing=True,
    )

    bpm2_next = [
        (row["j_bpm_name"], bool(row["crosses_excluded_element"]))
        for row in catalog["BPM.2"]
        if row["branch"] == "next"
    ]

    assert ("BPM.3", True) in bpm2_next
    assert ("BPM.4", True) in bpm2_next


def test_evaluate_momentum_estimates_applies_known_crossing_kick() -> None:
    base_pair = replace(_make_pairs()[0], kick=0.0, kick_sign=0.0)
    kicked_pair = replace(base_pair, kick=2.5e-4, kick_sign=-1.0)

    base_value = evaluate_momentum_estimates([base_pair])[0]
    kicked_value = evaluate_momentum_estimates([kicked_pair])[0]

    assert kicked_value == pytest.approx(base_value - 2.5e-4)


def test_nbpm_ac_dipole_config_runs_acd_first_and_reuses_fitted_kicks(monkeypatch) -> None:
    tws_bpm = _make_twiss_bpms().assign(x=0.0, y=0.0, px=0.0, py=0.0)
    tws_bpm.attrs["q1"] = 1.0
    tws_bpm.attrs["q2"] = 1.0
    twiss_elements = pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "ACD.TEST", "BPM.3", "BPM.4"],
            "s": [0.0, 10.0, 15.0, 20.0, 30.0],
        }
    ).set_index("name")
    data = pd.DataFrame(
        {
            "name": ["BPM.1", "BPM.2", "BPM.3", "BPM.4"] * 2,
            "turn": [0] * 4 + [1] * 4,
            "x": [1.0e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.1e-3, 1.3e-3, 1.5e-3, 1.7e-3],
            "y": [-1.0e-3, -1.2e-3, -1.4e-3, -1.6e-3, -1.1e-3, -1.3e-3, -1.5e-3, -1.7e-3],
            "var_x": [1.0e-8] * 8,
            "var_y": [1.0e-8] * 8,
        }
    )

    acd_result = pd.DataFrame(
        {
            "turn": [0, 1],
            "bpm_upstream": ["BPM.2", "BPM.2"],
            "bpm_downstream": ["BPM.3", "BPM.3"],
            "dpx_fit_rad": [2.0e-4, -1.5e-4],
            "dpy_fit_rad": [-3.0e-4, 2.5e-4],
            "px_bpm_upstream_cleaned": [9.0e-4, 8.5e-4],
            "py_bpm_upstream_cleaned": [-7.0e-4, -6.5e-4],
            "px_bpm_downstream_cleaned": [5.0e-4, 4.5e-4],
            "py_bpm_downstream_cleaned": [3.0e-4, 2.5e-4],
        }
    )
    acd_result.attrs["acd_marker"] = "ACD.TEST"
    acd_result.attrs["bpm_upstream"] = "BPM.2"
    acd_result.attrs["bpm_downstream"] = "BPM.3"

    calls: list[tuple[list[str], list[str]]] = []

    def _fake_run_ac_dipole_reconstruction(
        acd_data: pd.DataFrame,
        acd_tws: pd.DataFrame,
        config: ACDipoleConfig,
    ) -> pd.DataFrame:
        calls.append((acd_data["name"].astype(str).tolist(), acd_tws.index.astype(str).tolist()))
        assert config.ac_dipole_marker == "ACD.TEST"
        return acd_result.copy(deep=True)

    monkeypatch.setattr(
        nbpm_reconstruction_module,
        "run_ac_dipole_reconstruction",
        _fake_run_ac_dipole_reconstruction,
    )

    explicit = calculate_transverse_pz_nbpm(
        data.copy(deep=True),
        tws=tws_bpm,
        twiss_elements=twiss_elements,
        acdipole_element_name="ACD.TEST",
        acd_kicks=acd_result,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    )
    with_config = calculate_transverse_pz_nbpm(
        data.copy(deep=True),
        tws=tws_bpm,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
        ac_dipole_config=ACDipoleConfig(
            ac_dipole_marker="ACD.TEST",
            model=SimpleNamespace(twiss_elements=twiss_elements),  # ty:ignore[invalid-argument-type]
            bpm_upstream="BPM.2",
            bpm_downstream="BPM.3",
        ),
    )

    assert len(calls) == 1
    assert explicit.attrs["ac_dipole_bpm_upstream"] == "BPM.2"
    assert explicit.attrs["ac_dipole_bpm_downstream"] == "BPM.3"
    assert with_config.attrs["ac_dipole_bpm_upstream"] == "BPM.2"
    assert with_config.attrs["ac_dipole_bpm_downstream"] == "BPM.3"

    unaffected = with_config[~with_config["name"].isin(["BPM.2", "BPM.3"])].merge(
        explicit[["name", "turn", "px", "py"]],
        on=["name", "turn"],
        suffixes=("_config", "_explicit"),
    )
    np.testing.assert_allclose(
        unaffected["px_config"].to_numpy(),
        unaffected["px_explicit"].to_numpy(),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        unaffected["py_config"].to_numpy(),
        unaffected["py_explicit"].to_numpy(),
        equal_nan=True,
    )

    upstream = with_config[with_config["name"] == "BPM.2"].sort_values("turn")
    downstream = with_config[with_config["name"] == "BPM.3"].sort_values("turn")
    upstream_explicit = explicit[explicit["name"] == "BPM.2"].sort_values("turn")
    downstream_explicit = explicit[explicit["name"] == "BPM.3"].sort_values("turn")
    np.testing.assert_allclose(
        upstream_explicit["px"].to_numpy(),
        acd_result["px_bpm_upstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        upstream_explicit["py"].to_numpy(),
        acd_result["py_bpm_upstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        downstream_explicit["px"].to_numpy(),
        acd_result["px_bpm_downstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        downstream_explicit["py"].to_numpy(),
        acd_result["py_bpm_downstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        upstream["px"].to_numpy(),
        acd_result["px_bpm_upstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        upstream["py"].to_numpy(),
        acd_result["py_bpm_upstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        downstream["px"].to_numpy(),
        acd_result["px_bpm_downstream_cleaned"].to_numpy(),
    )
    np.testing.assert_allclose(
        downstream["py"].to_numpy(),
        acd_result["py_bpm_downstream_cleaned"].to_numpy(),
    )


def test_nbpm_builds_sparse_optics_blocks_and_symmetric_covariance() -> None:
    bpm_pairs = _make_pairs()
    tws_bpm = _make_twiss_bpms()
    twiss_elements = _make_twiss_elements()
    sources = build_effective_k1_sources(twiss_elements)

    dp_dphi = build_dp_dphi(bpm_pairs)
    t_meas, c_meas, _ = build_measurement_jacobian(bpm_pairs)
    r_phase_k = build_phase_response_matrix(twiss_elements, bpm_pairs, sources, "x")
    r_phase_s, bpm_names = build_bpm_s_response_matrix(tws_bpm, bpm_pairs, "x")
    t_k = build_TK_block(dp_dphi, r_phase_k)
    t_s = build_Ts_block(dp_dphi, r_phase_s)
    c_k = np.diag(sources["sigma_k1_eff"].to_numpy(dtype=float) ** 2)
    c_s, bpm_s_names = build_bpm_longitudinal_covariance(tws_bpm)

    assert bpm_names == bpm_s_names
    assert np.count_nonzero(r_phase_s[0]) == 2
    assert np.count_nonzero(r_phase_s[1]) == 2
    assert np.count_nonzero(r_phase_s[2]) == 2

    source_index = {name: idx for idx, name in enumerate(sources["elem_name"].tolist())}
    assert np.isclose(r_phase_k[0, source_index["MQY.1"]], 0.0)
    assert r_phase_k[0, source_index["MQXA.1"]] != 0.0
    assert r_phase_k[1, source_index["MS.1"]] != 0.0

    v_p, _t_full, _c_full = build_momentum_covariance(
        t_meas=t_meas,
        t_k=t_k,
        t_s=t_s,
        c_meas=c_meas,
        c_k=c_k,
        c_s=c_s,
    )
    diagnose_covariance_model(
        bpm_pairs=bpm_pairs,
        t_k=t_k,
        t_s=t_s,
        v_p=v_p,
        eff_sources=sources,
    )

    assert np.all(np.diag(v_p) >= 0.0)


def test_nbpm_method_improves_noisy_reconstruction() -> None:
    rng = np.random.default_rng(12345)
    bpm_pairs = _make_pairs()
    tws_bpm = _make_twiss_bpms()
    twiss_elements = _make_twiss_elements()
    sources = build_effective_k1_sources(twiss_elements)

    p0 = np.array([1.02e-4, 0.98e-4, 1.01e-4], dtype=float)
    dp_dphi = build_dp_dphi(bpm_pairs)
    t_meas, c_meas, _ = build_measurement_jacobian(bpm_pairs)
    r_phase_k = build_phase_response_matrix(twiss_elements, bpm_pairs, sources, "x")
    r_phase_s, _ = build_bpm_s_response_matrix(tws_bpm, bpm_pairs, "x")
    t_k = build_TK_block(dp_dphi, r_phase_k)
    t_s = build_Ts_block(dp_dphi, r_phase_s)
    c_k = np.diag((5.0 * sources["sigma_k1_eff"].to_numpy(dtype=float)) ** 2)
    c_s_base, _ = build_bpm_longitudinal_covariance(tws_bpm)
    c_s = (5.0**2) * c_s_base
    v_p, _t_full, _c_full = build_momentum_covariance(
        t_meas=t_meas,
        t_k=t_k,
        t_s=t_s,
        c_meas=c_meas,
        c_k=c_k,
        c_s=c_s,
    )

    _blue_estimate, _blue_var, blue_weights = combine_momentum_blue(p0, v_p)
    naive_diag = np.diag(v_p)
    naive_weights = (1.0 / naive_diag) / np.sum(1.0 / naive_diag)

    draws = 4000
    nuis_meas = rng.multivariate_normal(np.zeros(c_meas.shape[0]), c_meas, size=draws)
    nuis_k = rng.multivariate_normal(np.zeros(c_k.shape[0]), c_k, size=draws)
    nuis_s = rng.multivariate_normal(np.zeros(c_s.shape[0]), c_s, size=draws)

    p_draws = p0[None, :] + nuis_meas @ t_meas.T + nuis_k @ t_k.T + nuis_s @ t_s.T
    truth = float(np.mean(p0))
    blue_estimates = p_draws @ blue_weights
    naive_estimates = p_draws @ naive_weights

    blue_rmse = float(np.sqrt(np.mean((blue_estimates - truth) ** 2)))
    naive_rmse = float(np.sqrt(np.mean((naive_estimates - truth) ** 2)))

    assert blue_rmse < naive_rmse
    assert blue_rmse < 0.98 * naive_rmse


def test_nbpm_optics_jacobians_match_finite_differences() -> None:
    bpm_pairs = _make_pairs()
    tws_bpm = _make_twiss_bpms()
    twiss_elements = _make_twiss_elements()
    sources = build_effective_k1_sources(twiss_elements)

    dp_dphi = build_dp_dphi(bpm_pairs)
    r_phase_k = build_phase_response_matrix(twiss_elements, bpm_pairs, sources, "x")
    r_phase_s, _bpm_names = build_bpm_s_response_matrix(tws_bpm, bpm_pairs, "x")
    t_k = build_TK_block(dp_dphi, r_phase_k)
    t_s = build_Ts_block(dp_dphi, r_phase_s)
    base_momenta = evaluate_momentum_estimates(bpm_pairs)
    step = 1.0e-7

    def _finite_difference_column(phase_shift: np.ndarray) -> np.ndarray:
        plus_pairs = [
            replace(pair, phi_model_rad=pair.phi_model_rad + float(delta_phi))
            for pair, delta_phi in zip(bpm_pairs, phase_shift, strict=False)
        ]
        minus_pairs = [
            replace(pair, phi_model_rad=pair.phi_model_rad - float(delta_phi))
            for pair, delta_phi in zip(bpm_pairs, phase_shift, strict=False)
        ]
        plus = evaluate_momentum_estimates(plus_pairs)
        minus = evaluate_momentum_estimates(minus_pairs)
        assert np.all(np.isfinite(base_momenta))
        return (plus - minus) / (2.0 * step)

    for col_idx in range(r_phase_k.shape[1]):
        fd_col = _finite_difference_column(step * r_phase_k[:, col_idx])
        np.testing.assert_allclose(fd_col, t_k[:, col_idx], rtol=5.0e-6, atol=1.0e-10)

    for col_idx in range(r_phase_s.shape[1]):
        fd_col = _finite_difference_column(step * r_phase_s[:, col_idx])
        np.testing.assert_allclose(fd_col, t_s[:, col_idx], rtol=5.0e-6, atol=1.0e-10)


@pytest.mark.slow
def test_nbpm_method_improves_noisy_reconstruction_full_flow(
    data_dir,
    xsuite_json_path,
) -> None:
    seq_file = SEQ_FILE
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)

    tracking_df, tws_xsuite, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq,
        delta_p=0.0,
        ramp_turns=1000,
        flattop_turns=40,
    )
    tracking_df["name"] = tracking_df["name"].astype(str).str.upper()
    tracking_df["turn"] = tracking_df["turn"].astype(int)
    tracking_df, tws_bpm, selected_bpms = _select_local_bpm_window(
        tracking_df,
        _xsuite_to_full_ng_tws(tws_xsuite),
    )
    truth = get_truth(tracking_df, tws_bpm)

    full_tws = _xsuite_to_full_ng_tws(baseline_line.twiss(method="4d"))
    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(noisy_df, tracking_df, np.random.default_rng(42), noise_std=1e-4)

    baseline = calculate_transverse_pz(
        noisy_df.copy(deep=True),
        tws=tws_bpm,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_base", "py": "py_base"})

    nbpm = calculate_transverse_pz_nbpm(
        noisy_df.copy(deep=True),
        tws=tws_bpm,
        twiss_elements=full_tws,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    ).rename(columns={"px": "px_nbpm", "py": "py_nbpm"})

    merged, px_rmse_base, py_rmse_base, px_rmse_nbpm, py_rmse_nbpm = _evaluate_nbpm_vs_baseline(
        truth,
        baseline,
        nbpm,
        selected_bpms=selected_bpms,
    )

    assert len(merged) > 0

    assert px_rmse_nbpm < px_rmse_base / 2
    assert py_rmse_nbpm < py_rmse_base / 2

    # Now if we did an svd clean on the data, the improvement should be less pronounced
    cleaned_df = svd_clean_measurements(noisy_df)
    nbpm_cleaned = calculate_transverse_pz_nbpm(
        cleaned_df.copy(deep=True),
        tws=tws_bpm,
        twiss_elements=full_tws,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    ).rename(columns={"px": "px_nbpm_cleaned", "py": "py_nbpm_cleaned"})

    baseline_cleaned = calculate_transverse_pz(
        cleaned_df.copy(deep=True),
        tws=tws_bpm,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_base_cleaned", "py": "py_base_cleaned"})
    merged_cleaned = truth.merge(
        baseline_cleaned[["name", "turn", "px_base_cleaned", "py_base_cleaned"]],
        on=["name", "turn"],
        how="inner",
    ).merge(
        nbpm_cleaned[["name", "turn", "px_nbpm_cleaned", "py_nbpm_cleaned"]],
        on=["name", "turn"],
        how="inner",
    )
    merged_cleaned = merged_cleaned[merged_cleaned["name"].isin(selected_bpms)]
    merged_cleaned = merged_cleaned.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "px_true",
            "py_true",
            "px_base_cleaned",
            "py_base_cleaned",
            "px_nbpm_cleaned",
            "py_nbpm_cleaned",
        ]
    )
    px_rmse_base_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_base_cleaned"].to_numpy(),
    )
    py_rmse_base_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_base_cleaned"].to_numpy(),
    )
    px_rmse_nbpm_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_nbpm_cleaned"].to_numpy(),
    )
    py_rmse_nbpm_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_nbpm_cleaned"].to_numpy(),
    )
    assert px_rmse_nbpm_cleaned < px_rmse_base_cleaned
    assert py_rmse_nbpm_cleaned < py_rmse_base_cleaned


@pytest.mark.slow
def test_nbpm_method_handles_magnetic_errors_and_bpm_noise_full_flow(data_dir) -> None:
    tracking_df, tws_full, truth, full_tws = _get_setup_with_magnetic_errors(
        data_dir,
        flattop_turns=30,
        magnet_seed=42,
        rel_k1_std_dev=1e-4,
    )
    tracking_df, tws_bpm, selected_bpms = _select_local_bpm_window(tracking_df, tws_full)
    truth = truth[truth["name"].isin(selected_bpms)].copy(deep=True)

    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(noisy_df, tracking_df, np.random.default_rng(123), noise_std=1e-4)

    baseline = calculate_transverse_pz(
        noisy_df.copy(deep=True),
        tws=tws_bpm,
        inject_noise=False,
        info=False,
    ).rename(columns={"px": "px_base", "py": "py_base"})

    nbpm = calculate_transverse_pz_nbpm(
        noisy_df.copy(deep=True),
        tws=tws_bpm,
        twiss_elements=full_tws,
        inject_noise=False,
        info=False,
        max_bpm_distance=11,
    ).rename(columns={"px": "px_nbpm", "py": "py_nbpm"})

    merged, px_rmse_base, py_rmse_base, px_rmse_nbpm, py_rmse_nbpm = _evaluate_nbpm_vs_baseline(
        truth,
        baseline,
        nbpm,
        selected_bpms=selected_bpms,
    )

    assert len(merged) > 0
    assert np.isfinite(px_rmse_base)
    assert np.isfinite(py_rmse_base)
    assert np.isfinite(px_rmse_nbpm)
    assert np.isfinite(py_rmse_nbpm)
    assert px_rmse_nbpm < 5.0 * px_rmse_base
    assert py_rmse_nbpm < 5.0 * py_rmse_base
