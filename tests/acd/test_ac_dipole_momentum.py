from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import pandas as pd
import pytest
import tfs
from matplotlib import pyplot as plt
from pymadng_utils.accelerators import LHC
from xtrack_tools.acd import run_acd_track

pytest.importorskip("pymadng_utils")
pytest.importorskip("xtrack_tools")

from pymadng_utils.mad.accelerator_mad_interface import AcceleratorMadInterface
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from tests.momentum.momentum_test_utils import get_truth, rmse, xsuite_to_ngtws
from tmom_recon import inject_noise_xy_inplace
from tmom_recon.acd import reconstruction as acd_reconstruction
from tmom_recon.acd.madng_driver import ACDipoleMadDriver
from tmom_recon.acd.models import (
    ACDipoleBPMWindow,
    ACDipoleHarmonicFit,
    ACDipoleStateEstimate,
    ACDipoleStateSeries,
)
from tmom_recon.acd.reconstruction import (
    calculate_ac_dipole_momentum,
    select_ac_dipole_bpm_window,
    select_ac_dipole_bpms,
)
from tmom_recon.data.schema import NEXT, PREV
from tmom_recon.svd import svd_clean_measurements

from .acd_test_helpers import (
    AC_DIPOLE_ELEMENT,
    _ac_dipole_segment_around_element,
    _get_driver,
)

SEQ_FILE = "lhcb1.seq"


def _should_plot_test_results() -> bool:
    return os.getenv("TMOM_RECON_PLOT_TESTS", "0") == "1"


class _IdentityAcdModel:
    def __init__(self, twiss_elements: pd.DataFrame) -> None:
        self.twiss_elements = twiss_elements

    def track_particles(
        self,
        source_name: str,
        marker_name: str,
        source_state: np.ndarray,
        *,
        direction: int,
    ) -> np.ndarray:
        del source_name, marker_name, direction
        return np.asarray(source_state, dtype=float)


def _get_setup(
    seq_file: str,
    data_dir: Path,
    xsuite_json_path,
    *,
    delta_p: float = 0.0,
    ramp_turns: int = 1000,
    flattop_turns: int = 1000,
):
    seq = data_dir / "sequences" / seq_file
    json_path = xsuite_json_path(seq_file)
    tracking_df, tws, baseline_line = run_acd_track(
        json_path=json_path,
        sequence_file=seq,
        delta_p=delta_p,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
    )
    tws = xsuite_to_ngtws(tws)
    truth = get_truth(tracking_df, tws)
    return tracking_df, tws, truth, baseline_line.twiss(method="4d")


def _get_setup_with_magnetic_errors(
    data_dir: Path,
    *,
    flattop_turns: int = 100,
    magnet_seed: int = 13,
    rel_k1_std_dev: float | None = None,
):
    """Build an ACD tracking dataset with lattice perturbations and orbit correction.

    This helper intentionally exercises the pymadng-based workflow used in the
    integration tests:
    1. apply random magnet perturbations,
    2. re-match tunes via orbit correction,
    3. export the resulting machine state to xtrack and generate turn-by-turn data.
    """

    sequence_file = data_dir / "sequences" / SEQ_FILE
    accelerator = LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800)
    mad = AcceleratorMadInterface(accelerator)

    mad.observe()
    tws = mad.run_twiss()
    tws = tws.loc[tws.index.str.upper().str.contains("BPM")]
    mad.unobserve_elements(["BPM"])

    magnet_strengths, _reference_strengths = mad.apply_magnet_perturbations(
        rel_error=rel_k1_std_dev,
        seed=magnet_seed,
        magnet_type="qd",
    )
    assert magnet_strengths, "Expected magnet perturbations to update strengths"

    matched_tunes = mad.perform_orbit_correction(
        machine_deltap=0.0,
        target_qx=0.28,
        target_qy=0.31,
        corrector_file=None,
    )

    # Build xtrack env with perturbed magnet strengths and no explicit corrector table.
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
    ng_tws = xsuite_to_ngtws(full_tws)
    truth = get_truth(tracking_df, ng_tws)
    return tracking_df, ng_tws, truth, full_tws


def _transport_rows_to_marker(
    rows: pd.DataFrame,
    model,
    *,
    source_name: str,
    marker_name: str,
    direction: int,
) -> np.ndarray:
    states = rows[["x", "px", "y", "py"]].to_numpy(dtype=float)
    return model.track_particles(source_name, marker_name, states, direction=direction)


def _raw_mad_track(
    model: ACDipoleMadDriver,
    *,
    range_name: str,
    direction: int,
    states: np.ndarray,
) -> pd.DataFrame:
    x0_particles = [
        {
            "x": float(x),
            "px": float(px),
            "y": float(y),
            "py": float(py),
            "t": 0.0,
            "pt": 0.0,
        }
        for x, px, y, py in np.asarray(states, dtype=float)
    ]
    model.mad.send(
        """
--begin
range = py:recv()
x0_particles = py:recv()
direction = py:recv()

tbl, flw = track {
    sequence=loaded_sequence,
    range=range,
    X0=x0_particles,
    save=true,
    nturn=1,
    dir=direction,
    observe=1,
    deltap=DELTAP
}
py:send(true)
--end
"""
    ).send(range_name).send(x0_particles).send(direction)
    assert model.mad.recv()
    track_df = model.mad.tbl.to_df(force_pandas=True)
    return (
        track_df.sort_values(["id", "turn", "s"], kind="stable")
        .groupby("id", sort=False, as_index=False)
        .tail(1)
        .sort_values("id", kind="stable")
        .reset_index(drop=True)
    )


def _build_truth_at_ac_dipole(
    tracking_df: pd.DataFrame,
    model,
    *,
    bpm_upstream: str,
    bpm_downstream: str,
    marker_name: str,
) -> pd.DataFrame:
    up_rows = (
        tracking_df.loc[tracking_df["name"] == bpm_upstream, ["turn", "x", "px", "y", "py"]]
        .sort_values("turn")
        .reset_index(drop=True)
    )
    down_rows = (
        tracking_df.loc[tracking_df["name"] == bpm_downstream, ["turn", "x", "px", "y", "py"]]
        .sort_values("turn")
        .reset_index(drop=True)
    )

    up_at_marker = _transport_rows_to_marker(
        up_rows,
        model,
        source_name=bpm_upstream,
        marker_name=marker_name,
        direction=1,
    )
    down_at_marker = _transport_rows_to_marker(
        down_rows,
        model,
        source_name=bpm_downstream,
        marker_name=marker_name,
        direction=-1,
    )

    truth = up_rows[["turn", "px", "py"]].rename(
        columns={"px": "px_bpm_upstream_true", "py": "py_bpm_upstream_true"}
    )
    truth = truth.merge(
        down_rows[["turn", "px", "py"]].rename(
            columns={"px": "px_bpm_downstream_true", "py": "py_bpm_downstream_true"}
        ),
        on="turn",
        how="inner",
    )
    truth["x_acd_upstream_true"] = up_at_marker[:, 0]
    truth["px_acd_upstream_true"] = up_at_marker[:, 1]
    truth["y_acd_upstream_true"] = up_at_marker[:, 2]
    truth["py_acd_upstream_true"] = up_at_marker[:, 3]
    truth["x_acd_downstream_true"] = down_at_marker[:, 0]
    truth["px_acd_downstream_true"] = down_at_marker[:, 1]
    truth["y_acd_downstream_true"] = down_at_marker[:, 2]
    truth["py_acd_downstream_true"] = down_at_marker[:, 3]
    truth["dpx_rad_true"] = truth["px_acd_downstream_true"] - truth["px_acd_upstream_true"]
    truth["dpy_rad_true"] = truth["py_acd_downstream_true"] - truth["py_acd_upstream_true"]
    return truth


def _plot_ac_dipole_reconstruction(merged: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(merged["turn"], merged["px_bpm_upstream_true"], label="px upstream true", lw=1.3)
    axes[0].plot(merged["turn"], merged["px_bpm_upstream"], "--", label="px upstream reco", lw=1.1)
    if "px_bpm_upstream_cleaned" in merged.columns:
        axes[0].plot(
            merged["turn"],
            merged["px_bpm_upstream_cleaned"],
            ":",
            label="px upstream cleaned",
            lw=1.2,
        )
    axes[0].plot(
        merged["turn"], merged["px_bpm_downstream_true"], label="px downstream true", lw=1.3
    )
    axes[0].plot(
        merged["turn"],
        merged["px_bpm_downstream"],
        "--",
        label="px downstream reco",
        lw=1.1,
    )
    if "px_bpm_downstream_cleaned" in merged.columns:
        axes[0].plot(
            merged["turn"],
            merged["px_bpm_downstream_cleaned"],
            ":",
            label="px downstream cleaned",
            lw=1.2,
        )
    axes[0].set_ylabel("px [rad]")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(merged["turn"], merged["py_bpm_upstream_true"], label="py upstream true", lw=1.3)
    axes[1].plot(merged["turn"], merged["py_bpm_upstream"], "--", label="py upstream reco", lw=1.1)
    if "py_bpm_upstream_cleaned" in merged.columns:
        axes[1].plot(
            merged["turn"],
            merged["py_bpm_upstream_cleaned"],
            ":",
            label="py upstream cleaned",
            lw=1.2,
        )
    axes[1].plot(
        merged["turn"], merged["py_bpm_downstream_true"], label="py downstream true", lw=1.3
    )
    axes[1].plot(
        merged["turn"],
        merged["py_bpm_downstream"],
        "--",
        label="py downstream reco",
        lw=1.1,
    )
    if "py_bpm_downstream_cleaned" in merged.columns:
        axes[1].plot(
            merged["turn"],
            merged["py_bpm_downstream_cleaned"],
            ":",
            label="py downstream cleaned",
            lw=1.2,
        )
    axes[1].set_ylabel("py [rad]")
    axes[1].legend(loc="upper right", ncol=2)

    axes[2].plot(merged["turn"], merged["dpx_rad_true"], label="dpx true", lw=1.3)
    axes[2].plot(merged["turn"], merged["dpx"], "--", label="dpx reco", lw=1.1)
    if "dpx_fit_rad" in merged.columns:
        axes[2].plot(merged["turn"], merged["dpx_fit_rad"], ":", label="dpx fit", lw=1.2)
    axes[2].plot(merged["turn"], merged["dpy_rad_true"], label="dpy true", lw=1.3)
    axes[2].plot(merged["turn"], merged["dpy"], "--", label="dpy reco", lw=1.1)
    if "dpy_fit_rad" in merged.columns:
        axes[2].plot(merged["turn"], merged["dpy_fit_rad"], ":", label="dpy fit", lw=1.2)
    axes[2].set_ylabel("kick [rad]")
    axes[2].legend(loc="upper right", ncol=2)

    axes[3].plot(
        merged["turn"], merged["px_acd_upstream_true"], label="px ACD true from up", lw=1.3
    )
    axes[3].plot(
        merged["turn"], merged["px_acd_upstream"], "--", label="px ACD reco from up", lw=1.1
    )
    if "px_acd_upstream_cleaned" in merged.columns:
        axes[3].plot(
            merged["turn"],
            merged["px_acd_upstream_cleaned"],
            ":",
            label="px ACD cleaned from up",
            lw=1.2,
        )
    axes[3].plot(
        merged["turn"], merged["px_acd_downstream_true"], label="px ACD true from down", lw=1.3
    )
    axes[3].plot(
        merged["turn"],
        merged["px_acd_downstream"],
        "--",
        label="px ACD reco from down",
        lw=1.1,
    )
    if "px_acd_downstream_cleaned" in merged.columns:
        axes[3].plot(
            merged["turn"],
            merged["px_acd_downstream_cleaned"],
            ":",
            label="px ACD cleaned from down",
            lw=1.2,
        )
    axes[3].set_ylabel("ACD px [rad]")
    axes[3].set_xlabel("turn")
    axes[3].legend(loc="upper right", ncol=2)
    if _should_plot_test_results():
        plt.show()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)


def _minimal_bpm_state_frame(name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "turn": [0, 1],
            "x": [0.0, 0.0],
            "px": [1.0e-4, 2.0e-4],
            "y": [0.0, 0.0],
            "py": [3.0e-4, 4.0e-4],
            "var_x": [1.0, 1.0],
            "var_px": [1.0, 1.0],
            "var_y": [1.0, 1.0],
            "var_py": [1.0, 1.0],
            "source_bpm": [name, name],
        }
    )


def _estimate_from_table(table: pd.DataFrame) -> ACDipoleStateEstimate:
    ordered = table.sort_values("turn").reset_index(drop=True)
    return ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            x=ordered["x"].to_numpy(dtype=float),
            px=ordered["px"].to_numpy(dtype=float),
            y=ordered["y"].to_numpy(dtype=float),
            py=ordered["py"].to_numpy(dtype=float),
        ),
        var_x=ordered["var_x"].to_numpy(dtype=float),
        var_px=ordered["var_px"].to_numpy(dtype=float),
        var_y=ordered["var_y"].to_numpy(dtype=float),
        var_py=ordered["var_py"].to_numpy(dtype=float),
    )


def test_calculate_ac_dipole_momentum_uses_direct_bpm_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = cast(
        ACDipoleMadDriver,
        _IdentityAcdModel(pd.DataFrame(index=pd.Index(["BPMU", "MKQA.6L4.B1", "BPMD"], dtype=str))),
    )
    tws = pd.DataFrame(
        {
            "px": [0.0, 0.0],
            "py": [0.0, 0.0],
        },
        index=pd.Index(["BPMU", "BPMD"], dtype=str),
    )
    data = pd.DataFrame(
        {
            "name": ["BPMU", "BPMD", "BPMU", "BPMD"],
            "turn": [0, 0, 1, 1],
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "var_x": [1.0, 1.0, 1.0, 1.0],
            "var_y": [1.0, 1.0, 1.0, 1.0],
        }
    )
    counts = {"prev": 0, "next": 0}

    monkeypatch.setattr(
        acd_reconstruction,
        "select_ac_dipole_bpm_window",
        lambda *args, **kwargs: ACDipoleBPMWindow(("BPMU",), ("BPMD",)),
    )
    monkeypatch.setattr(acd_reconstruction, "remove_closed_orbit_inplace", lambda data, tws: None)

    def fake_prepare_neighbor_tables(
        tws_bpm: pd.DataFrame,
        use_immediate_neighbors_for_bpms: bool = False,
    ) -> tuple[object, object, object, object]:
        del tws_bpm, use_immediate_neighbors_for_bpms
        return object(), object(), object(), object()

    def fake_prepare_prev_reconstruction(*args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        counts["prev"] += 1
        return _minimal_bpm_state_frame("BPMU")

    def fake_prepare_next_reconstruction(*args, **kwargs) -> pd.DataFrame:
        del args, kwargs
        counts["next"] += 1
        return _minimal_bpm_state_frame("BPMD")

    def fake_clean_ac_dipole_states(
        turns: np.ndarray,
        upstream_tables: list[pd.DataFrame],
        downstream_tables: list[pd.DataFrame],
        *,
        dpx_tune: float,
        dpy_tune: float,
        smooth_lambda: float,
    ) -> tuple[
        ACDipoleStateEstimate, ACDipoleStateEstimate, ACDipoleHarmonicFit, ACDipoleHarmonicFit
    ]:
        del turns, dpx_tune, dpy_tune, smooth_lambda
        fit = ACDipoleHarmonicFit(
            tune=0.1,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=np.zeros(2, dtype=float),
        )
        return (
            _estimate_from_table(upstream_tables[0]),
            _estimate_from_table(downstream_tables[0]),
            fit,
            fit,
        )

    monkeypatch.setattr(
        acd_reconstruction, "_prepare_neighbor_tables", fake_prepare_neighbor_tables
    )
    monkeypatch.setattr(
        acd_reconstruction, "_prepare_prev_reconstruction", fake_prepare_prev_reconstruction
    )
    monkeypatch.setattr(
        acd_reconstruction, "_prepare_next_reconstruction", fake_prepare_next_reconstruction
    )
    monkeypatch.setattr(acd_reconstruction, "_clean_ac_dipole_states", fake_clean_ac_dipole_states)

    result = calculate_ac_dipole_momentum(
        data,
        tws,
        ac_dipole_marker="MKQA.6L4.B1",
        model=model,
        inject_noise=False,
    )

    assert counts == {"prev": 1, "next": 1}
    assert result.attrs["bpm_upstream"] == "BPMU"
    assert result.attrs["bpm_downstream"] == "BPMD"


@pytest.mark.parametrize("use_immediate_neighbors_for_bpms", [False, True])
def test_prepare_neighbor_tables_immediate_bpm_flag(
    use_immediate_neighbors_for_bpms: bool,
) -> None:
    bpm_names = [f"BPM{i}" for i in range(1, 6)]
    mu = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    tws_bpm = tfs.TfsDataFrame(
        {
            "mu1": mu,
            "mu2": mu,
            "s": np.arange(len(bpm_names), dtype=float),
        },
        index=pd.Index(bpm_names, dtype=str),
        headers={"q1": 1.0, "q2": 1.0},
    )

    prev_x, prev_y, next_x, next_y = acd_reconstruction._prepare_neighbor_tables(
        tws_bpm,
        use_immediate_neighbors_for_bpms=use_immediate_neighbors_for_bpms,
    )

    if use_immediate_neighbors_for_bpms:
        expected_prev = ["BPM5", "BPM1", "BPM2", "BPM3", "BPM4"]
        expected_next = ["BPM2", "BPM3", "BPM4", "BPM5", "BPM1"]
        expected_prev_delta = [0.35, -0.15, -0.15, -0.15, -0.15]
        expected_next_delta = [-0.15, -0.15, -0.15, -0.15, 0.35]
    else:
        expected_prev_table = acd_reconstruction.prev_bpm_to_pi_2(tws_bpm["mu1"], 1.0)
        expected_next_table = acd_reconstruction.next_bpm_to_pi_2(tws_bpm["mu1"], 1.0)
        expected_prev = expected_prev_table["prev_bpm"].tolist()
        expected_next = expected_next_table["next_bpm"].tolist()
        expected_prev_delta = expected_prev_table["delta"].tolist()
        expected_next_delta = expected_next_table["delta"].tolist()

    assert prev_x[PREV.bpm_x].tolist() == expected_prev
    assert prev_y[PREV.bpm_y].tolist() == expected_prev
    assert next_x[NEXT.bpm_x].tolist() == expected_next
    assert next_y[NEXT.bpm_y].tolist() == expected_next
    assert np.allclose(prev_x[PREV.delta_x].to_numpy(dtype=float), expected_prev_delta)
    assert np.allclose(prev_y[PREV.delta_y].to_numpy(dtype=float), expected_prev_delta)
    assert np.allclose(next_x[NEXT.delta_x].to_numpy(dtype=float), expected_next_delta)
    assert np.allclose(next_y[NEXT.delta_y].to_numpy(dtype=float), expected_next_delta)


@pytest.mark.slow
def test_select_ac_dipole_bpms_matches_real_lattice_neighbors(data_dir, xsuite_json_path) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        xsuite_json_path,
        flattop_turns=64,
    )
    model = _get_driver(data_dir / "sequences" / SEQ_FILE, debug=False)
    expected_up, expected_down = _ac_dipole_segment_around_element(
        model.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )

    selection = select_ac_dipole_bpms(
        model.twiss_elements,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        bpm_names=tracking_df["name"].unique(),
    )

    assert selection.upstream == expected_up
    assert selection.downstream == expected_down


@pytest.mark.slow
def test_madng_track_range_and_direction_match_source_target_convention(
    data_dir,
    xsuite_json_path,
) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        xsuite_json_path,
        flattop_turns=20,
    )
    model = _get_driver(
        data_dir / "sequences" / SEQ_FILE,
        debug=False,
    )
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        model.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )
    up_rows = (
        tracking_df.loc[tracking_df["name"] == bpm_upstream, ["turn", "x", "px", "y", "py"]]
        .sort_values("turn")
        .reset_index(drop=True)
    )
    down_rows = (
        tracking_df.loc[tracking_df["name"] == bpm_downstream, ["turn", "x", "px", "y", "py"]]
        .sort_values("turn")
        .reset_index(drop=True)
    )
    count = min(len(up_rows), len(down_rows), 20)
    up_states = up_rows[["x", "px", "y", "py"]].to_numpy(dtype=float)[:count]
    down_states = down_rows[["x", "px", "y", "py"]].to_numpy(dtype=float)[:count]

    forward = _raw_mad_track(
        model,
        range_name=f"{bpm_upstream}/{bpm_downstream}",
        direction=1,
        states=up_states,
    )
    backward_same_range = _raw_mad_track(
        model,
        range_name=f"{bpm_downstream}/{bpm_upstream}",
        direction=-1,
        states=down_states,
    )
    backward_swapped_range = _raw_mad_track(
        model,
        range_name=f"{bpm_upstream}/{bpm_downstream}",
        direction=-1,
        states=down_states,
    )

    forward_rmse = rmse(
        down_states.reshape(-1),
        forward[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
    )
    backward_same_range_rmse = rmse(
        up_states.reshape(-1),
        backward_same_range[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
    )
    backward_swapped_range_rmse = rmse(
        up_states.reshape(-1),
        backward_swapped_range[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
    )

    assert forward_rmse < 1e-4
    assert backward_same_range_rmse < backward_swapped_range_rmse


@pytest.mark.slow
def test_select_ac_dipole_bpm_window_returns_primary_pair(data_dir, xsuite_json_path) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        xsuite_json_path,
        flattop_turns=64,
    )
    model = _get_driver(data_dir / "sequences" / SEQ_FILE, debug=False)

    window = select_ac_dipole_bpm_window(
        model.twiss_elements,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        bpm_names=tracking_df["name"].unique(),
    )

    assert len(window.upstream) == 1
    assert len(window.downstream) == 1
    assert window.primary == select_ac_dipole_bpms(
        model.twiss_elements,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        bpm_names=tracking_df["name"].unique(),
    )


@pytest.mark.slow
def test_calculate_ac_dipole_momentum_uses_real_tracking_setup(
    data_dir,
    xsuite_json_path,
    tmp_path,
) -> None:
    tracking_df, tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        xsuite_json_path,
        flattop_turns=100,
    )
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        _get_driver(data_dir / "sequences" / SEQ_FILE, debug=False).twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )

    reco_log = tmp_path / "acd_madng_reco.log"
    model = _get_driver(
        data_dir / "sequences" / SEQ_FILE,
        debug=True,
        mad_logfile=reco_log,
    )
    tracked_up = model.track_particles(
        bpm_upstream,
        AC_DIPOLE_ELEMENT,
        np.zeros((2, 4), dtype=float),
        direction=1,
    )
    tracked_down = model.track_particles(
        bpm_downstream,
        AC_DIPOLE_ELEMENT,
        np.zeros((2, 4), dtype=float),
        direction=-1,
    )
    assert tracked_up.shape == (2, 4)
    assert tracked_down.shape == (2, 4)

    truth = _build_truth_at_ac_dipole(
        tracking_df,
        model,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        marker_name=AC_DIPOLE_ELEMENT,
    )

    result = calculate_ac_dipole_momentum(
        tracking_df,
        tws,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        model=model,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        inject_noise=False,
    )

    merged = result.merge(truth, on="turn", how="inner")
    assert len(merged) == len(truth)
    assert result.attrs["acd_marker"] == AC_DIPOLE_ELEMENT
    assert result.attrs["acd_element"] == AC_DIPOLE_ELEMENT
    assert result.attrs["bpm_upstream"] == bpm_upstream
    assert result.attrs["bpm_downstream"] == bpm_downstream
    assert result.attrs["bpms_upstream_used"] == (bpm_upstream,)
    assert result.attrs["bpms_downstream_used"] == (bpm_downstream,)
    assert result.headers["ACD_MARKER"] == AC_DIPOLE_ELEMENT
    assert result.headers["ACD_ELEMENT"] == AC_DIPOLE_ELEMENT
    assert result.headers["ACD_BPM_UPSTREAM"] == bpm_upstream
    assert result.headers["ACD_BPM_DOWNSTREAM"] == bpm_downstream
    assert reco_log.exists()

    px_up_rmse = rmse(
        merged["px_bpm_upstream_true"].to_numpy(), merged["px_bpm_upstream"].to_numpy()
    )
    py_up_rmse = rmse(
        merged["py_bpm_upstream_true"].to_numpy(), merged["py_bpm_upstream"].to_numpy()
    )
    px_down_rmse = rmse(
        merged["px_bpm_downstream_true"].to_numpy(),
        merged["px_bpm_downstream"].to_numpy(),
    )
    py_down_rmse = rmse(
        merged["py_bpm_downstream_true"].to_numpy(),
        merged["py_bpm_downstream"].to_numpy(),
    )
    dpx_rmse = rmse(merged["dpx_rad_true"].to_numpy(), merged["dpx"].to_numpy())
    dpy_rmse = rmse(merged["dpy_rad_true"].to_numpy(), merged["dpy"].to_numpy())

    assert px_up_rmse < 1e-10
    assert py_up_rmse < 5e-12
    assert px_down_rmse < 5e-15
    assert py_down_rmse < 1e-14
    assert dpx_rmse < 1e-9
    assert dpy_rmse < 1e-10

    plot_path = tmp_path / "ac_dipole_momentum_debug.png"
    _plot_ac_dipole_reconstruction(merged, plot_path)
    assert plot_path.exists()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("use_svd_cleaning", "include_magnetic_errors", "ratio_limit"),
    [
        (False, False, 0.8),
        (True, False, 0.6),
        (True, True, 0.6),
    ],
    ids=["raw_noisy", "svd_cleaned_noisy", "svd_cleaned_noisy_with_magnet_errors"],
)
def test_ac_dipole_kick_fit_improves_noisy_reconstruction(
    data_dir,
    xsuite_json_path,
    tmp_path,
    use_svd_cleaning: bool,
    include_magnetic_errors: bool,
    ratio_limit: float,
) -> None:
    if include_magnetic_errors:
        tracking_df, tws, _truth, _full_tws = _get_setup_with_magnetic_errors(
            data_dir,
            flattop_turns=100,
        )
    else:
        tracking_df, tws, _truth, _full_tws = _get_setup(
            SEQ_FILE,
            data_dir,
            xsuite_json_path,
            flattop_turns=100,
        )
    model = _get_driver(
        data_dir / "sequences" / SEQ_FILE,
        debug=True,
        mad_logfile=tmp_path / "acd_madng_reco_noisy.log",
    )
    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        model.twiss_elements,
        available_bpms=tracking_df["name"].unique().tolist(),
        element_name=AC_DIPOLE_ELEMENT,
    )

    truth = _build_truth_at_ac_dipole(
        tracking_df,
        model,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        marker_name=AC_DIPOLE_ELEMENT,
    )

    noisy_df = tracking_df.copy(deep=True)
    inject_noise_xy_inplace(
        noisy_df,
        tracking_df,
        np.random.default_rng(42),
        noise_std=1e-4,
    )
    input_df = svd_clean_measurements(noisy_df) if use_svd_cleaning else noisy_df

    result = calculate_ac_dipole_momentum(
        input_df,
        tws,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        model=model,
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        inject_noise=False,
    )

    merged = result.merge(truth, on="turn", how="inner")
    dpx_rmse_raw = rmse(merged["dpx_rad_true"].to_numpy(), merged["dpx"].to_numpy())
    dpy_rmse_raw = rmse(merged["dpy_rad_true"].to_numpy(), merged["dpy"].to_numpy())
    dpx_rmse_fit = rmse(merged["dpx_rad_true"].to_numpy(), merged["dpx_fit_rad"].to_numpy())
    dpy_rmse_fit = rmse(merged["dpy_rad_true"].to_numpy(), merged["dpy_fit_rad"].to_numpy())

    assert dpx_rmse_fit < dpx_rmse_raw
    assert dpy_rmse_fit < dpy_rmse_raw

    if use_svd_cleaning or include_magnetic_errors:
        assert dpx_rmse_fit < ratio_limit * dpx_rmse_raw
        assert dpy_rmse_fit < 0.999 * dpy_rmse_raw
    else:
        assert dpx_rmse_fit < ratio_limit * dpx_rmse_raw
        assert dpy_rmse_fit < ratio_limit * dpy_rmse_raw
    assert np.all(np.isfinite(merged["px_bpm_upstream_cleaned"].to_numpy()))
    assert np.all(np.isfinite(merged["py_bpm_upstream_cleaned"].to_numpy()))
    assert np.all(np.isfinite(merged["px_bpm_downstream_cleaned"].to_numpy()))
    assert np.all(np.isfinite(merged["py_bpm_downstream_cleaned"].to_numpy()))

    fit_attrs_x = result.attrs["dpx_fit"]
    fit_attrs_y = result.attrs["dpy_fit"]
    assert result.headers["ACD_DPX_TUNE"] == pytest.approx(fit_attrs_x["tune"])
    assert result.headers["ACD_DPX_AMPLITUDE"] == pytest.approx(fit_attrs_x["amplitude"])
    assert result.headers["ACD_DPX_PHASE"] == pytest.approx(fit_attrs_x["phase"])
    assert result.headers["ACD_DPX_OFFSET"] == pytest.approx(fit_attrs_x["offset"])
    assert result.headers["ACD_DPY_TUNE"] == pytest.approx(fit_attrs_y["tune"])
    assert result.headers["ACD_DPY_AMPLITUDE"] == pytest.approx(fit_attrs_y["amplitude"])
    assert result.headers["ACD_DPY_PHASE"] == pytest.approx(fit_attrs_y["phase"])
    assert result.headers["ACD_DPY_OFFSET"] == pytest.approx(fit_attrs_y["offset"])
    assert fit_attrs_x["amplitude"] > 0.0
    assert fit_attrs_y["amplitude"] > 0.0
    assert 0.0 < fit_attrs_x["tune"] < 0.5
    assert 0.0 < fit_attrs_y["tune"] < 0.5

    plot_path = tmp_path / "ac_dipole_momentum_noisy_cleaned.png"
    _plot_ac_dipole_reconstruction(merged, plot_path)
    assert plot_path.exists()


@pytest.mark.slow
def test_ac_dipole_kick_fit_improves_with_more_turns(
    data_dir,
    xsuite_json_path,
    tmp_path,
) -> None:
    flattop_turns_grid = [50, 100, 200]
    dpx_fit_errors: list[float] = []
    dpy_fit_errors: list[float] = []

    for flattop_turns in flattop_turns_grid:
        tracking_df, tws, _truth, _full_tws = _get_setup(
            SEQ_FILE,
            data_dir,
            xsuite_json_path,
            flattop_turns=flattop_turns,
        )
        model = _get_driver(
            data_dir / "sequences" / SEQ_FILE,
            debug=True,
            mad_logfile=tmp_path / f"acd_madng_reco_turns_{flattop_turns}.log",
        )
        bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
            model.twiss_elements,
            available_bpms=tracking_df["name"].unique().tolist(),
            element_name=AC_DIPOLE_ELEMENT,
        )
        truth = _build_truth_at_ac_dipole(
            tracking_df,
            model,
            bpm_upstream=bpm_upstream,
            bpm_downstream=bpm_downstream,
            marker_name=AC_DIPOLE_ELEMENT,
        )

        noisy_df = tracking_df.copy(deep=True)
        inject_noise_xy_inplace(
            noisy_df,
            tracking_df,
            np.random.default_rng(42),
            noise_std=1e-4,
        )
        result = calculate_ac_dipole_momentum(
            noisy_df,
            tws,
            ac_dipole_marker=AC_DIPOLE_ELEMENT,
            model=model,
            bpm_upstream=bpm_upstream,
            bpm_downstream=bpm_downstream,
            inject_noise=False,
        )
        merged = result.merge(truth, on="turn", how="inner")
        dpx_fit_errors.append(
            rmse(merged["dpx_rad_true"].to_numpy(), merged["dpx_fit_rad"].to_numpy())
        )
        dpy_fit_errors.append(
            rmse(merged["dpy_rad_true"].to_numpy(), merged["dpy_fit_rad"].to_numpy())
        )

    assert dpx_fit_errors[-1] < dpx_fit_errors[0]
    assert dpy_fit_errors[-1] < dpy_fit_errors[0]


@pytest.mark.slow
def test_ac_dipole_reports_selected_bpms(
    data_dir,
    xsuite_json_path,
    tmp_path,
) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        xsuite_json_path,
        flattop_turns=100,
    )
    model = _get_driver(
        data_dir / "sequences" / SEQ_FILE,
        debug=True,
        mad_logfile=tmp_path / "acd_madng_reco_multi.log",
    )
    expected_window = select_ac_dipole_bpm_window(
        model.twiss_elements,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        bpm_names=tracking_df["name"].unique(),
    )
    result = calculate_ac_dipole_momentum(
        tracking_df,
        _tws,
        ac_dipole_marker=AC_DIPOLE_ELEMENT,
        model=model,
        inject_noise=False,
    )

    assert result.attrs["bpms_upstream_used"] == expected_window.upstream
    assert result.attrs["bpms_downstream_used"] == expected_window.downstream
    assert result.headers["ACD_BPMS_UPSTREAM_USED"] == ",".join(expected_window.upstream)
    assert result.headers["ACD_BPMS_DOWNSTREAM_USED"] == ",".join(expected_window.downstream)
    assert "px_bpm_upstream_cleaned" in result.columns
    assert "py_bpm_upstream_cleaned" in result.columns
    assert "px_bpm_downstream_cleaned" in result.columns
    assert "py_bpm_downstream_cleaned" in result.columns
