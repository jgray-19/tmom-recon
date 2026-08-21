"""LHC tracking artifact construction for integration-test fixtures."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import tfs
from pymadng_utils.accelerators import LHC
from pymadng_utils.mad import AcceleratorMadInterface
from xtrack_tools.acd import run_ac_dipole_tracking, run_acd_track
from xtrack_tools.env import initialise_env
from xtrack_tools.monitors import process_tracking_data

from tests.support.acd_barrier import acd_barrier_s
from tests.support.truth import model_details_for
from tmom_recon import ModelDetails
from tmom_recon.acd.madng_driver import ACDipoleMadDriver

AC_DIPOLE_MARKER = "MKQA.6L4.B1"
AC_DIPOLE_DRIVEN_TUNES = (0.27, 0.322)
# A tracked LHC dataframe measured about 5 MiB, unlike its 366--534 MiB line.
# Keep the useful lightweight products within a fixed 256 MiB session budget.
TRACKING_DATA_CACHE_BYTES = 256 * 2**20

_LHC_HORIZONTAL_EXCITATION = 2 * 0.042 / 180.0**0.5
_LHC_VERTICAL_EXCITATION = 2 * 0.042 / 177.0**0.5
_SEQUENCE_DECLARATION = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*:\s*sequence\b", re.IGNORECASE)
_NATURAL_TUNES = (0.28, 0.31)


@dataclass(frozen=True)
class TrackingArtifacts:
    """Tracking data plus optional short-lived Xsuite optics objects."""

    data: Any
    baseline_line: Any | None = None
    baseline_twiss_4d: Any | None = None


def _copy_tracking_artifacts(artifacts: TrackingArtifacts) -> TrackingArtifacts:
    return TrackingArtifacts(
        data=artifacts.data.copy(deep=True),
        baseline_line=(
            artifacts.baseline_line.copy() if artifacts.baseline_line is not None else None
        ),
        baseline_twiss_4d=artifacts.baseline_twiss_4d,
    )


def _sequence_name_from_file(seq: Path) -> str:
    """Return the MAD-X sequence name declared in a file."""
    with seq.open(encoding="utf-8") as handle:
        for line in handle:
            match = _SEQUENCE_DECLARATION.match(line)
            if match:
                return match.group(1)
    return seq.stem


def create_loaded_mad_interface(sequence_file: Path) -> AcceleratorMadInterface:
    accelerator = LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800)
    return AcceleratorMadInterface(accelerator)


def get_twiss(sequence_file: Path, deltap: float) -> pd.DataFrame:
    """Return the twiss DataFrame for a given sequence and delta_p."""
    acciface = create_loaded_mad_interface(sequence_file)
    acciface.observe()
    return acciface.run_twiss(coupling=True, deltap=deltap)


def lhc_model_details(sequence_file: Path, *, delta_p: float = 0.0) -> ModelDetails:
    """Build LHC model details at the tracked absolute momentum."""
    accelerator = LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800)
    return model_details_for(accelerator, pt=accelerator.dp2pt(delta_p))


def tracking_artifacts_loader(xsuite_json_path):
    """Return a bounded data cache with opt-in, non-cached Xsuite lines."""
    data_cache: OrderedDict[tuple, Any] = OrderedDict()
    cached_bytes = 0

    def data_size(data: Any) -> int:
        return int(data.memory_usage(index=True, deep=True).sum())

    def cache_data(key: tuple, data: Any) -> None:
        nonlocal cached_bytes
        data_cache[key] = data
        cached_bytes += data_size(data)
        while cached_bytes > TRACKING_DATA_CACHE_BYTES and data_cache:
            _, evicted = data_cache.popitem(last=False)
            cached_bytes -= data_size(evicted)

    def load(
        seq_path: Path,
        json_path: Path,
        acd_marker: str,
        sequence_name: str,
        delta_p: float,
        ramp_turns: int,
        flattop_turns: int,
        state_markers: bool,
        include_line: bool,
    ) -> TrackingArtifacts:
        key = (
            seq_path,
            json_path,
            acd_marker,
            sequence_name,
            delta_p,
            ramp_turns,
            flattop_turns,
            state_markers,
        )
        if key not in data_cache or include_line:
            tracking_df, tws_xsuite, baseline_line = run_acd_track(
                sequence_file=seq_path,
                acd_marker=acd_marker,
                sequence_name=sequence_name,
                json_path=Path(json_path),
                delta_p=delta_p,
                ramp_turns=ramp_turns,
                flattop_turns=flattop_turns,
                state_markers=state_markers,
                horizontal_excitation=_LHC_HORIZONTAL_EXCITATION,
                vertical_excitation=_LHC_VERTICAL_EXCITATION,
            )
            if key not in data_cache:
                cache_data(key, tracking_df)
            if include_line:
                return TrackingArtifacts(
                    data=data_cache[key],
                    baseline_line=baseline_line,
                    baseline_twiss_4d=baseline_line.twiss(method="4d"),
                )
        else:
            data_cache.move_to_end(key)
        return TrackingArtifacts(data=data_cache[key])

    def get(
        seq_file: Path,
        *,
        delta_p: float = 0.0,
        ramp_turns: int = 1000,
        flattop_turns: int = 100,
        state_markers: bool = False,
        acd_marker: str = AC_DIPOLE_MARKER,
        include_line: bool = False,
    ) -> TrackingArtifacts:
        return _copy_tracking_artifacts(
            load(
                seq_file,
                xsuite_json_path(seq_file),
                acd_marker,
                "lhcb1",
                delta_p,
                ramp_turns,
                flattop_turns,
                state_markers,
                include_line,
            )
        )

    return get


def lhc_acd_barrier_s(accelerator, pt: float) -> float:
    """Resolve the tracked LHC AC-dipole position from MAD-NG element twiss."""
    model = ACDipoleMadDriver(accelerator=accelerator, pt=pt, observed_elements=AC_DIPOLE_MARKER)
    return acd_barrier_s(model, AC_DIPOLE_MARKER)


def setup_xsuite_simulation(
    delta_p: float,
    magnets_to_perturb: str | list[str],
    magnet_seed: int,
    json_path: Path,
    sequence_file: Path,
    tmp_path: Path,
    test_id: str,
    rel_k1_std_dev: float = 1e-4,
    flattop_turns: int = 100,
    initial_tune_guess: dict[str, float] | None = None,
    track_delta_p: float | None = None,
):
    """Build the shared LHC tracking scenario used by specialist contracts."""
    corrector_file = tmp_path / f"correctors_{test_id}.tfs"
    mad = create_loaded_mad_interface(sequence_file)
    mad.mad["zero_twiss", "_"] = mad.mad.twiss(sequence="loaded_sequence")
    mad.observe()
    mad.run_twiss()
    mad.unobserve_elements(["BPM"])
    magnet_strengths = {}
    if magnets_to_perturb:
        magnet_strengths = mad.apply_magnet_perturbations(
            rel_error=rel_k1_std_dev, seed=magnet_seed, magnet_type=magnets_to_perturb
        )
        if isinstance(magnet_strengths, tuple):
            magnet_strengths = magnet_strengths[0]
        assert magnet_strengths
    if initial_tune_guess is not None:
        mad.set_madx_variables(**initial_tune_guess)
    matched_tunes = mad.perform_orbit_correction(
        machine_deltap=delta_p,
        target_qx=_NATURAL_TUNES[0],
        target_qy=_NATURAL_TUNES[1],
        corrector_file=corrector_file,
    )
    corrector_table = cast(pd.DataFrame, tfs.read(corrector_file))
    corrector_table = corrector_table.loc[
        ~corrector_table["kind"].astype(str).str.lower().isin({"monitor", "hmonitor", "vmonitor"})
    ]
    env = initialise_env(
        matched_tunes,
        magnet_strengths,
        corrector_table,
        sequence_file=sequence_file,
        seq_name="lhcb1",
    )
    baseline_line = env["lhcb1"].copy()
    xsuite_tws = baseline_line.twiss(method="4d", delta0=delta_p)
    tracked_delta_p = delta_p if track_delta_p is None else track_delta_p
    monitored_line = run_ac_dipole_tracking(
        line=baseline_line,
        acd_marker=AC_DIPOLE_MARKER,
        sequence_name="lhcb1",
        tws=xsuite_tws,
        deltap=tracked_delta_p,
        ramp_turns=1000,
        flattop_turns=flattop_turns,
        driven_tunes=AC_DIPOLE_DRIVEN_TUNES,
        bpm_pattern=r"(?i)bpm.*",
        horizontal_excitation=_LHC_HORIZONTAL_EXCITATION,
        vertical_excitation=_LHC_VERTICAL_EXCITATION,
    )
    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=1000,
        flattop_turns=flattop_turns,
        add_variance_columns=False,
    )
    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0
    truth = tracking_df[["name", "turn", "x", "px", "y", "py"]].rename(
        columns={"px": "px_true", "py": "py_true", "x": "x_true", "y": "y_true"}
    )
    model_details = ModelDetails(
        accelerator=LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800),
        pt=LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800).dp2pt(tracked_delta_p),
        magnet_strengths=magnet_strengths or None,
        corrector_knobs=corrector_file,
    )
    return tracking_df, truth, model_details, xsuite_tws, baseline_line
