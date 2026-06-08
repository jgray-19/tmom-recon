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

pytest.importorskip("pymadng_utils")
pytest.importorskip("xtrack_tools")

from pymadng_utils.mad.accelerator_mad_interface import AcceleratorMadInterface
from xtrack_tools.acd import run_ac_dipole_tracking_with_particles
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from tests.momentum.momentum_test_utils import get_truth, rmse, xsuite_to_ngtws
from tmom_recon import inject_noise_xy_inplace
from tmom_recon.acd import reconstruction as acd_reconstruction
from tmom_recon.acd.cleaning import _clean_ac_dipole_states, _refine_known_kick_fit
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
DRIVEN_TUNES = (0.27, 0.322)


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
        deltap: float | None = None,
    ) -> np.ndarray:
        del source_name, marker_name, direction, deltap
        return np.asarray(source_state, dtype=float)


class _ShiftedAcdModel(_IdentityAcdModel):
    def track_particles(
        self,
        source_name: str,
        marker_name: str,
        source_state: np.ndarray,
        *,
        direction: int,
        deltap: float | None = None,
    ) -> np.ndarray:
        del source_name, marker_name, deltap
        state = np.asarray(source_state, dtype=float).copy()
        state[:, 1] += 3.0 * float(direction)
        state[:, 3] -= 2.0 * float(direction)
        return state


def _get_setup(
    seq_file: str,
    data_dir: Path,
    acd_tracking_setup,
    *,
    delta_p: float = 0.0,
    ramp_turns: int = 1000,
    flattop_turns: int = 1000,
):
    setup = acd_tracking_setup(
        seq_file,
        data_dir,
        delta_p=delta_p,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
    )
    return (
        setup["tracking_df"],
        setup["tws"],
        setup["truth"],
        setup["baseline_twiss_4d"],
    )


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
    observe=0,
    deltap=DELTAP
}
py:send(true)
--end
"""
    ).send(range_name).send(x0_particles).send(direction)
    assert model.mad.recv()
    track_df = model.mad.tbl.to_df(force_pandas=True)
    return (
        track_df.reset_index(drop=True)
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


def _harmonic_bpm_state_frame(
    name: str,
    *,
    turns: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> pd.DataFrame:
    turns = np.asarray(turns, dtype=int)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    zeros = np.zeros_like(px, dtype=float)
    ones = np.ones_like(px, dtype=float)
    return pd.DataFrame(
        {
            "turn": turns,
            "x": zeros,
            "px": px,
            "y": zeros,
            "py": py,
            "var_x": ones,
            "var_px": ones,
            "var_y": ones,
            "var_py": ones,
            "source_bpm": [name] * len(turns),
        }
    )


def test_refine_known_kick_fit_refines_tune_to_improve_phase_and_amplitude() -> None:
    turns = np.arange(256, dtype=int)
    true_tune = 0.322
    tune_hint = true_tune - 0.003
    true_amplitude = 7.5e-6
    true_phase = 0.63
    true_offset = -1.2e-6
    signal = true_amplitude * np.sin(2.0 * np.pi * true_tune * turns + true_phase) + true_offset

    upstream = _harmonic_bpm_state_frame(
        "BPMU",
        turns=turns,
        px=np.zeros_like(signal),
        py=np.zeros_like(signal),
    )
    downstream = _harmonic_bpm_state_frame(
        "BPMD",
        turns=turns,
        px=signal,
        py=signal,
    )

    fit = _refine_known_kick_fit(
        turns.astype(float),
        [upstream],
        [downstream],
        value_col="py",
        variance_col="var_py",
        tune_hint=tune_hint,
    )

    hint_design = np.column_stack(
        [
            np.sin(2.0 * np.pi * tune_hint * turns),
            np.cos(2.0 * np.pi * tune_hint * turns),
            np.ones_like(turns, dtype=float),
        ]
    )
    hint_coeffs, *_ = np.linalg.lstsq(hint_design, signal, rcond=None)
    hint_fit = hint_design @ hint_coeffs

    refined_rmse = rmse(signal, fit.fitted)
    hinted_rmse = rmse(signal, hint_fit)
    phase_error = np.angle(np.exp(1j * (fit.phase - true_phase)))

    assert fit.tune == pytest.approx(true_tune, abs=5.0e-4)
    assert fit.amplitude == pytest.approx(true_amplitude, rel=0.02)
    assert phase_error == pytest.approx(0.0, abs=0.03)
    assert fit.offset == pytest.approx(true_offset, abs=5.0e-8)
    assert refined_rmse < hinted_rmse
    assert refined_rmse < 1.0e-7


def test_clean_ac_dipole_states_uses_one_common_marker_xy_per_turn() -> None:
    turns = np.array([0.0, 1.0], dtype=float)
    upstream = pd.DataFrame(
        {
            "turn": [0, 1],
            "x": [1.0, 3.0],
            "px": [0.0, 0.0],
            "y": [2.0, 4.0],
            "py": [0.0, 0.0],
            "var_x": [1.0, 1.0],
            "var_px": [1.0, 1.0],
            "var_y": [1.0, 1.0],
            "var_py": [1.0, 1.0],
            "source_bpm": ["BPMU", "BPMU"],
        }
    )
    downstream = pd.DataFrame(
        {
            "turn": [0, 1],
            "x": [5.0, 7.0],
            "px": [0.0, 0.0],
            "y": [8.0, 10.0],
            "py": [0.0, 0.0],
            "var_x": [1.0, 1.0],
            "var_px": [1.0, 1.0],
            "var_y": [1.0, 1.0],
            "var_py": [1.0, 1.0],
            "source_bpm": ["BPMD", "BPMD"],
        }
    )

    pre_kick, post_kick, dpx_fit, dpy_fit = _clean_ac_dipole_states(
        turns,
        [upstream],
        [downstream],
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        smooth_lambda=0.0,
    )

    expected_x = np.array([3.0, 5.0])
    expected_y = np.array([5.0, 7.0])

    assert np.allclose(pre_kick.state.x, expected_x)
    assert np.allclose(post_kick.state.x, expected_x)
    assert np.allclose(pre_kick.state.y, expected_y)
    assert np.allclose(post_kick.state.y, expected_y)
    assert np.allclose(pre_kick.state.x, post_kick.state.x)
    assert np.allclose(pre_kick.state.y, post_kick.state.y)
    assert np.allclose(dpx_fit.fitted, 0.0)
    assert np.allclose(dpy_fit.fitted, 0.0)


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
    monkeypatch.setattr(acd_reconstruction, "_resolve_name", lambda name, candidates: name)
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
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        inject_noise=False,
    )

    assert counts == {"prev": 1, "next": 1}
    assert result.attrs["bpm_upstream"] == "BPMU"
    assert result.attrs["bpm_downstream"] == "BPMD"


def test_reconstruct_bpm_momentum_from_cleaned_acd_backtracks_from_marker() -> None:
    model = _ShiftedAcdModel(
        pd.DataFrame(index=pd.Index(["BPMU", "MKQA.6L4.B1", "BPMD"], dtype=str))
    )
    tws = pd.DataFrame(
        {
            "px": [0.25, 0.0, -0.5],
            "py": [-0.1, 0.0, 0.75],
        },
        index=pd.Index(["BPMU", "MKQA.6L4.B1", "BPMD"], dtype=str),
    )
    cleaned_upstream = ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            x=np.array([0.0, 0.0]),
            px=np.array([1.0, 2.0]),
            y=np.array([0.0, 0.0]),
            py=np.array([5.0, 6.0]),
        ),
        var_x=np.ones(2),
        var_px=np.ones(2),
        var_y=np.ones(2),
        var_py=np.ones(2),
    )
    cleaned_downstream = ACDipoleStateEstimate(
        state=ACDipoleStateSeries(
            x=np.array([0.0, 0.0]),
            px=np.array([1.0, 2.0]),
            y=np.array([0.0, 0.0]),
            py=np.array([5.0, 6.0]),
        ),
        var_x=np.ones(2),
        var_px=np.ones(2),
        var_y=np.ones(2),
        var_py=np.ones(2),
    )

    px_up, py_up = acd_reconstruction._reconstruct_bpm_momentum_from_cleaned_acd(
        tws,
        model,
        bpm_name="BPMU",
        cleaned_acd=cleaned_upstream,
        marker_name="MKQA.6L4.B1",
        direction=-1,
    )
    px_down, py_down = acd_reconstruction._reconstruct_bpm_momentum_from_cleaned_acd(
        tws,
        model,
        bpm_name="BPMD",
        cleaned_acd=cleaned_downstream,
        marker_name="MKQA.6L4.B1",
        direction=1,
    )

    assert np.allclose(px_up, [1.0 - 3.0 + 0.25, 2.0 - 3.0 + 0.25])
    assert np.allclose(py_up, [5.0 + 2.0 - 0.1, 6.0 + 2.0 - 0.1])
    assert np.allclose(px_down, [1.0 + 3.0 - 0.5, 2.0 + 3.0 - 0.5])
    assert np.allclose(py_down, [5.0 - 2.0 + 0.75, 6.0 - 2.0 + 0.75])


def test_calculate_ac_dipole_momentum_uses_explicit_acd_tunes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tws = tfs.TfsDataFrame(
        {
            "s": [0.0, 1.0],
            "x": [0.0, 0.0],
            "px": [0.0, 0.0],
            "y": [0.0, 0.0],
            "py": [0.0, 0.0],
            "betx": [1.0, 1.0],
            "bety": [1.0, 1.0],
            "mux": [0.0, 0.1],
            "muy": [0.0, 0.1],
            "mu1": [0.0, 0.1],
            "mu2": [0.0, 0.1],
        },
        index=pd.Index(["BPMU", "BPMD"], dtype=str),
        headers={"q1": 0.28, "q2": 0.31},
    )
    model = _IdentityAcdModel(tws)
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
    captured_tunes: dict[str, float] = {}

    monkeypatch.setattr(
        acd_reconstruction,
        "select_ac_dipole_bpm_window",
        lambda *args, **kwargs: ACDipoleBPMWindow(("BPMU",), ("BPMD",)),
    )
    monkeypatch.setattr(acd_reconstruction, "_resolve_name", lambda name, candidates: name)
    monkeypatch.setattr(acd_reconstruction, "remove_closed_orbit_inplace", lambda data, tws: None)
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_neighbor_tables",
        lambda *args, **kwargs: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_prev_reconstruction",
        lambda *args, **kwargs: _minimal_bpm_state_frame("BPMU"),
    )
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_next_reconstruction",
        lambda *args, **kwargs: _minimal_bpm_state_frame("BPMD"),
    )

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
        del turns, smooth_lambda
        captured_tunes["dpx"] = dpx_tune
        captured_tunes["dpy"] = dpy_tune
        fit_x = ACDipoleHarmonicFit(
            tune=dpx_tune,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=np.zeros(2, dtype=float),
        )
        fit_y = ACDipoleHarmonicFit(
            tune=dpy_tune,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=np.zeros(2, dtype=float),
        )
        return (
            _estimate_from_table(upstream_tables[0]),
            _estimate_from_table(downstream_tables[0]),
            fit_x,
            fit_y,
        )

    monkeypatch.setattr(acd_reconstruction, "_clean_ac_dipole_states", fake_clean_ac_dipole_states)

    result = calculate_ac_dipole_momentum(
        data,
        tws,
        ac_dipole_marker="MKQA.6L4.B1",
        model=model,
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        inject_noise=False,
    )

    assert captured_tunes == {
        "dpx": pytest.approx(DRIVEN_TUNES[0]),
        "dpy": pytest.approx(DRIVEN_TUNES[1]),
    }
    assert result.headers["ACD_DPX_TUNE"] == pytest.approx(DRIVEN_TUNES[0])
    assert result.headers["ACD_DPY_TUNE"] == pytest.approx(DRIVEN_TUNES[1])


def test_calculate_ac_dipole_momentum_uses_supplied_tunes_without_fft_precheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = np.arange(64, dtype=int)
    dpx_signal_tune = 0.27
    dpy_signal_tune = 0.322
    dpx_signal = np.sin(2.0 * np.pi * dpx_signal_tune * turns)
    dpy_signal = np.sin(2.0 * np.pi * dpy_signal_tune * turns)

    tws = tfs.TfsDataFrame(
        {
            "s": [0.0, 1.0],
            "x": [0.0, 0.0],
            "px": [0.0, 0.0],
            "y": [0.0, 0.0],
            "py": [0.0, 0.0],
            "betx": [1.0, 1.0],
            "bety": [1.0, 1.0],
            "mux": [0.0, 0.1],
            "muy": [0.0, 0.1],
            "mu1": [0.0, 0.1],
            "mu2": [0.0, 0.1],
        },
        index=pd.Index(["BPMU", "BPMD"], dtype=str),
        headers={"q1": 0.28, "q2": 0.31},
    )
    model = _IdentityAcdModel(tws)
    data = pd.DataFrame(
        {
            "name": np.repeat(["BPMU", "BPMD"], len(turns)),
            "turn": np.tile(turns, 2),
            "x": 0.0,
            "y": 0.0,
            "var_x": 1.0,
            "var_y": 1.0,
        }
    )

    monkeypatch.setattr(
        acd_reconstruction,
        "select_ac_dipole_bpm_window",
        lambda *args, **kwargs: ACDipoleBPMWindow(("BPMU",), ("BPMD",)),
    )
    monkeypatch.setattr(acd_reconstruction, "_resolve_name", lambda name, candidates: name)
    monkeypatch.setattr(acd_reconstruction, "remove_closed_orbit_inplace", lambda data, tws: None)
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_neighbor_tables",
        lambda *args, **kwargs: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_prev_reconstruction",
        lambda *args, **kwargs: _harmonic_bpm_state_frame(
            "BPMU",
            turns=turns,
            px=np.zeros_like(dpx_signal),
            py=np.zeros_like(dpy_signal),
        ),
    )
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_next_reconstruction",
        lambda *args, **kwargs: _harmonic_bpm_state_frame(
            "BPMD",
            turns=turns,
            px=dpx_signal,
            py=dpy_signal,
        ),
    )

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
        del turns, smooth_lambda
        fit_x = ACDipoleHarmonicFit(
            tune=dpx_tune,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=dpx_signal,
        )
        fit_y = ACDipoleHarmonicFit(
            tune=dpy_tune,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=dpy_signal,
        )
        return (
            _estimate_from_table(upstream_tables[0]),
            _estimate_from_table(downstream_tables[0]),
            fit_x,
            fit_y,
        )

    monkeypatch.setattr(acd_reconstruction, "_clean_ac_dipole_states", fake_clean_ac_dipole_states)

    result = calculate_ac_dipole_momentum(
        data,
        tws,
        ac_dipole_marker="MKQA.6L4.B1",
        model=model,
        dpx_tune=0.27,
        dpy_tune=0.322,
        inject_noise=False,
    )

    assert result.headers["ACD_DPX_TUNE"] == pytest.approx(0.27)
    assert result.headers["ACD_DPY_TUNE"] == pytest.approx(0.322)


def test_calculate_ac_dipole_momentum_does_not_reject_mismatched_raw_signal_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    turns = np.arange(64, dtype=int)
    dpx_signal = np.sin(2.0 * np.pi * 0.27 * turns)
    dpy_signal = np.sin(2.0 * np.pi * 0.322 * turns)

    tws = tfs.TfsDataFrame(
        {
            "s": [0.0, 1.0],
            "x": [0.0, 0.0],
            "px": [0.0, 0.0],
            "y": [0.0, 0.0],
            "py": [0.0, 0.0],
            "betx": [1.0, 1.0],
            "bety": [1.0, 1.0],
            "mux": [0.0, 0.1],
            "muy": [0.0, 0.1],
            "mu1": [0.0, 0.1],
            "mu2": [0.0, 0.1],
        },
        index=pd.Index(["BPMU", "BPMD"], dtype=str),
        headers={"q1": 0.28, "q2": 0.31},
    )
    model = _IdentityAcdModel(tws)
    data = pd.DataFrame(
        {
            "name": np.repeat(["BPMU", "BPMD"], len(turns)),
            "turn": np.tile(turns, 2),
            "x": 0.0,
            "y": 0.0,
            "var_x": 1.0,
            "var_y": 1.0,
        }
    )

    monkeypatch.setattr(
        acd_reconstruction,
        "select_ac_dipole_bpm_window",
        lambda *args, **kwargs: ACDipoleBPMWindow(("BPMU",), ("BPMD",)),
    )
    monkeypatch.setattr(acd_reconstruction, "_resolve_name", lambda name, candidates: name)
    monkeypatch.setattr(acd_reconstruction, "remove_closed_orbit_inplace", lambda data, tws: None)
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_neighbor_tables",
        lambda *args, **kwargs: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_prev_reconstruction",
        lambda *args, **kwargs: _harmonic_bpm_state_frame(
            "BPMU",
            turns=turns,
            px=np.zeros_like(dpx_signal),
            py=np.zeros_like(dpy_signal),
        ),
    )
    monkeypatch.setattr(
        acd_reconstruction,
        "_prepare_next_reconstruction",
        lambda *args, **kwargs: _harmonic_bpm_state_frame(
            "BPMD",
            turns=turns,
            px=dpx_signal,
            py=dpy_signal,
        ),
    )

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
        del turns, smooth_lambda
        fit_x = ACDipoleHarmonicFit(
            tune=dpx_tune,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=dpx_signal,
        )
        fit_y = ACDipoleHarmonicFit(
            tune=dpy_tune,
            amplitude=1.0,
            phase=0.0,
            offset=0.0,
            fitted=dpy_signal,
        )
        return (
            _estimate_from_table(upstream_tables[0]),
            _estimate_from_table(downstream_tables[0]),
            fit_x,
            fit_y,
        )

    monkeypatch.setattr(acd_reconstruction, "_clean_ac_dipole_states", fake_clean_ac_dipole_states)

    result = calculate_ac_dipole_momentum(
        data,
        tws,
        ac_dipole_marker="MKQA.6L4.B1",
        model=model,
        dpx_tune=0.20,
        dpy_tune=0.322,
        inject_noise=False,
    )

    assert result.headers["ACD_DPX_TUNE"] == pytest.approx(0.20)
    assert result.headers["ACD_DPY_TUNE"] == pytest.approx(0.322)


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
        mu1_series = pd.Series(np.asarray(tws_bpm["mu1"], dtype=float), index=tws_bpm.index)
        expected_prev_table = acd_reconstruction.prev_bpm_to_pi_2(mu1_series, 1.0)
        expected_next_table = acd_reconstruction.next_bpm_to_pi_2(mu1_series, 1.0)
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
def test_select_ac_dipole_bpms_matches_real_lattice_neighbors(data_dir, acd_tracking_setup) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        acd_tracking_setup,
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
def test_madng_track_range_and_direction_follow_shortest_path_convention(
    data_dir,
    acd_tracking_setup,
) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        acd_tracking_setup,
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

    acd_forward = _raw_mad_track(
        model,
        range_name=f"{bpm_upstream}/{AC_DIPOLE_ELEMENT}",
        direction=1,
        states=up_states,
    )
    acd_backward = _raw_mad_track(
        model,
        range_name=f"{bpm_downstream}/{AC_DIPOLE_ELEMENT}",
        direction=-1,
        states=down_states,
    )
    acd_wrong_forward = _raw_mad_track(
        model,
        range_name=f"{bpm_downstream}/{AC_DIPOLE_ELEMENT}",
        direction=1,
        states=down_states,
    )

    model_forward = model.track_particles(
        bpm_upstream,
        AC_DIPOLE_ELEMENT,
        up_states,
        direction=1,
    )
    model_backward = model.track_particles(
        bpm_downstream,
        AC_DIPOLE_ELEMENT,
        down_states,
        direction=-1,
    )

    assert forward_rmse < 1e-4
    assert backward_same_range_rmse < backward_swapped_range_rmse
    assert (
        rmse(
            model_forward.reshape(-1),
            acd_forward[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
        )
        < 1e-12
    )
    assert (
        rmse(
            model_backward.reshape(-1),
            acd_backward[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
        )
        < 1e-12
    )
    assert rmse(
        model_backward.reshape(-1),
        acd_backward[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
    ) < rmse(
        model_backward.reshape(-1),
        acd_wrong_forward[["x", "px", "y", "py"]].to_numpy(dtype=float).reshape(-1),
    )


@pytest.mark.slow
def test_select_ac_dipole_bpm_window_returns_primary_pair(data_dir, acd_tracking_setup) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        acd_tracking_setup,
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
    acd_tracking_setup,
    tmp_path,
) -> None:
    tracking_df, tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        acd_tracking_setup,
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
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
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
    assert np.allclose(
        merged["x_acd_upstream_cleaned"].to_numpy(),
        merged["x_acd_downstream_cleaned"].to_numpy(),
    )
    assert np.allclose(
        merged["y_acd_upstream_cleaned"].to_numpy(),
        merged["y_acd_downstream_cleaned"].to_numpy(),
    )

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
    acd_tracking_setup,
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
            acd_tracking_setup,
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
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
        bpm_upstream=bpm_upstream,
        bpm_downstream=bpm_downstream,
        inject_noise=False,
    )

    merged = result.merge(truth, on="turn", how="inner")
    dpx_rmse_raw = rmse(merged["dpx_rad_true"].to_numpy(), merged["dpx"].to_numpy())
    dpy_rmse_raw = rmse(merged["dpy_rad_true"].to_numpy(), merged["dpy"].to_numpy())
    dpx_rmse_fit = rmse(merged["dpx_rad_true"].to_numpy(), merged["dpx_fit_rad"].to_numpy())
    dpy_rmse_fit = rmse(merged["dpy_rad_true"].to_numpy(), merged["dpy_fit_rad"].to_numpy())
    bpm_rmse_raw = np.mean(
        [
            rmse(merged["px_bpm_upstream_true"].to_numpy(), merged["px_bpm_upstream"].to_numpy()),
            rmse(merged["py_bpm_upstream_true"].to_numpy(), merged["py_bpm_upstream"].to_numpy()),
            rmse(
                merged["px_bpm_downstream_true"].to_numpy(),
                merged["px_bpm_downstream"].to_numpy(),
            ),
            rmse(
                merged["py_bpm_downstream_true"].to_numpy(),
                merged["py_bpm_downstream"].to_numpy(),
            ),
        ]
    )
    bpm_rmse_cleaned = np.mean(
        [
            rmse(
                merged["px_bpm_upstream_true"].to_numpy(),
                merged["px_bpm_upstream_cleaned"].to_numpy(),
            ),
            rmse(
                merged["py_bpm_upstream_true"].to_numpy(),
                merged["py_bpm_upstream_cleaned"].to_numpy(),
            ),
            rmse(
                merged["px_bpm_downstream_true"].to_numpy(),
                merged["px_bpm_downstream_cleaned"].to_numpy(),
            ),
            rmse(
                merged["py_bpm_downstream_true"].to_numpy(),
                merged["py_bpm_downstream_cleaned"].to_numpy(),
            ),
        ]
    )

    assert dpx_rmse_fit < dpx_rmse_raw
    assert dpy_rmse_fit < dpy_rmse_raw
    assert bpm_rmse_cleaned < bpm_rmse_raw
    assert np.allclose(
        merged["x_acd_upstream_cleaned"].to_numpy(),
        merged["x_acd_downstream_cleaned"].to_numpy(),
    )
    assert np.allclose(
        merged["y_acd_upstream_cleaned"].to_numpy(),
        merged["y_acd_downstream_cleaned"].to_numpy(),
    )

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
    acd_tracking_setup,
    tmp_path,
) -> None:
    flattop_turns_grid = [50, 100, 200]
    dpx_fit_errors: list[float] = []
    dpy_fit_errors: list[float] = []

    for flattop_turns in flattop_turns_grid:
        tracking_df, tws, _truth, _full_tws = _get_setup(
            SEQ_FILE,
            data_dir,
            acd_tracking_setup,
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
            dpx_tune=DRIVEN_TUNES[0],
            dpy_tune=DRIVEN_TUNES[1],
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
    acd_tracking_setup,
    tmp_path,
) -> None:
    tracking_df, _tws, _truth, _full_tws = _get_setup(
        SEQ_FILE,
        data_dir,
        acd_tracking_setup,
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
        dpx_tune=DRIVEN_TUNES[0],
        dpy_tune=DRIVEN_TUNES[1],
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
