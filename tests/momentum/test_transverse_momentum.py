"""Integration tests for transverse momentum reconstruction using xtrack data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest
import tfs
from pymadng_utils.accelerators import LHC
from pymadng_utils.mad.accelerator_mad_interface import AcceleratorMadInterface
from xtrack_tools.acd import run_ac_dipole_tracking
from xtrack_tools.env import create_xsuite_environment, initialise_env
from xtrack_tools.monitors import process_tracking_data

from tests.acd.acd_test_helpers import AC_DIPOLE_ELEMENT
from tmom_recon import ModelDetails

from .momentum_test_utils import (  # noqa: E402
    transverse_calc as calculate_pz,
)
from .momentum_test_utils import (  # noqa: E402
    verify_pz_reconstruction,
)

if TYPE_CHECKING:
    from pathlib import Path

    from xtrack import Line


NAT_TUNES = [0.28, 0.31]
DRV_TUNES = [0.27, 0.322]
# Excitation amplitudes for the horizontal and vertical AC-dipole planes.
HORIZONTAL_EXCITATION = 2 * 0.042 / 180.0**0.5
VERTICAL_EXCITATION = 2 * 0.042 / 177.0**0.5


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def _create_loaded_mad_interface(sequence_file: Path) -> AcceleratorMadInterface:
    accelerator = LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800)
    return AcceleratorMadInterface(accelerator)


def _setup_xsuite_simulation(
    delta_p: float,
    magnets_to_perturb: str | list[str],
    magnet_seed: int,
    json_path: Path,
    sequence_file: Path,
    tmp_path: Path,
    test_id: str,
    rel_k1_std_dev=1e-4,
    flattop_turns=100,
    initial_tune_guess: dict[str, float] | None = None,
    track_delta_p: float | None = None,
):
    corrector_file = tmp_path / f"correctors_{test_id}.tfs"

    mad = cast(Any, _create_loaded_mad_interface(sequence_file))
    mad.mad["zero_twiss", "_"] = mad.mad.twiss(sequence="loaded_sequence")

    mad.observe()
    tws = mad.run_twiss()
    tws = tws.loc[tws.index.str.upper().str.contains("BPM")]
    mad.unobserve_elements(["BPM"])

    magnet_strengths = {}
    if magnets_to_perturb:
        magnet_strengths = mad.apply_magnet_perturbations(
            rel_error=rel_k1_std_dev,
            seed=magnet_seed,
            magnet_type=magnets_to_perturb,
        )
        if isinstance(magnet_strengths, tuple):
            magnet_strengths = magnet_strengths[0]
        assert magnet_strengths, "Expected magnet perturbations to update strengths"

    if initial_tune_guess is not None:
        mad.set_madx_variables(**initial_tune_guess)

    # Perform orbit correction
    matched_tunes = mad.perform_orbit_correction(
        machine_deltap=delta_p,
        target_qx=0.28,
        target_qy=0.31,
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

    qx = float(xsuite_tws.qx % 1)
    qy = float(xsuite_tws.qy % 1)
    assert np.isclose(qx, NAT_TUNES[0], atol=1e-3, rtol=0.0)
    assert np.isclose(qy, NAT_TUNES[1], atol=1e-3, rtol=0.0)

    ramp_turns = 1000

    # Use generalized tracking function
    track_delta_p = track_delta_p if track_delta_p is not None else delta_p

    monitored_line = run_ac_dipole_tracking(
        line=baseline_line,
        acd_marker=AC_DIPOLE_ELEMENT,
        sequence_name="lhcb1",
        tws=xsuite_tws,
        deltap=track_delta_p,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=[0.27, 0.322],
        bpm_pattern=r"(?i)bpm.*",
        horizontal_excitation=HORIZONTAL_EXCITATION,
        vertical_excitation=VERTICAL_EXCITATION,
    )

    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        add_variance_columns=False,
    )
    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0

    truth = tracking_df[["name", "turn", "x", "px", "y", "py"]].rename(
        columns={"px": "px_true", "py": "py_true", "x": "x_true", "y": "y_true"}
    )

    accelerator = LHC(beam=1, sequence_file=sequence_file, kinetic_energy=6800)
    model_details = ModelDetails(
        accelerator=accelerator,
        pt=accelerator.dp2pt(track_delta_p),
        magnet_strengths=magnet_strengths or None,
        corrector_knobs=corrector_file,
    )

    return tracking_df, truth, model_details, xsuite_tws


@pytest.mark.slow
def test_calculate_pz_recovers_true_momenta(seq_b1, tmp_path):
    """Test that calculate_pz reconstructs true momenta for on-momentum beam."""
    env = create_xsuite_environment(
        sequence_file=seq_b1,
        seq_name="lhcb1",
    )

    baseline_line: Line = env["lhcb1"].copy()
    ng = baseline_line.to_madng()
    tws = baseline_line.twiss(method="4d")

    qx = float(tws.qx % 1)
    qy = float(tws.qy % 1)
    assert np.isclose(qx, NAT_TUNES[0], atol=1e-6, rtol=1e-6)
    assert np.isclose(qy, NAT_TUNES[1], atol=1e-6, rtol=1e-6)
    qxd = DRV_TUNES[0]
    qyd = DRV_TUNES[1]
    acd_marker = AC_DIPOLE_ELEMENT.lower()
    betxac = tws.rows[acd_marker]["betx"][0]
    betyac = tws.rows[acd_marker]["bety"][0]
    ac_marker_place = "6.7065629327563011e+03"

    ng.send(f"""
    -- Install AC Kicker (AC Quad) elements
local hackicker, vackicker in MAD.element
!MAD.option.debug = 2;
local a = seq:replace({{
    hackicker "hackicker" {{
        at = {ac_marker_place},

        -- quad part
        nat_q = {qx},
        drv_q = {qxd},
        ac_bet = {betxac},
    }},
    vackicker "vackicker" {{
        at = {ac_marker_place},

        -- quad part
        nat_q = {qy},
        drv_q = {qyd},
        ac_bet = {betyac},
    }}
}}, "{acd_marker}");""")

    ramp_turns = 1000
    flattop_turns = 100

    monitored_line = run_ac_dipole_tracking(
        line=baseline_line,
        acd_marker=AC_DIPOLE_ELEMENT,
        sequence_name="lhcb1",
        tws=tws,
        deltap=0.0,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        driven_tunes=[qxd, qyd],
        bpm_pattern=r"(?i)bpm.*",
        horizontal_excitation=HORIZONTAL_EXCITATION,
        vertical_excitation=VERTICAL_EXCITATION,
    )

    tracking_df = process_tracking_data(
        monitored_line,
        ramp_turns=ramp_turns,
        flattop_turns=flattop_turns,
        add_variance_columns=False,
    )
    tracking_df["var_x"] = 1.0
    tracking_df["var_y"] = 1.0

    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    ng["tws", "flw"] = ng.twiss(sequence=ng.seq)
    tws: tfs.TfsDataFrame = (
        ng.tws.to_df()
        .set_index("name")
        .rename(index=str.upper)
        .loc[lambda df: df.index.str.contains("BPM")]
    )

    model_details = ModelDetails(
        accelerator=LHC(beam=1, sequence_file=seq_b1, kinetic_energy=6800),
        pt=0.0,
    )
    _verify_pz_reconstruction(
        tracking_df,
        truth,
        model_details,
        px_nonoise_max=3.5e-7,
        py_nonoise_max=2.5e-7,
        px_noisy_min=1e-6,
        px_noisy_max=2.8e-6,
        py_noisy_min=1e-6,
        py_noisy_max=2.6e-6,
        px_cleaned_max=7.5e-7,
        py_cleaned_max=6.5e-7,
        rng_seed=42,
    )


def _verify_pz_reconstruction(
    tracking_df,
    truth: pd.DataFrame,
    model_details: ModelDetails,
    px_nonoise_max: float,
    py_nonoise_max: float,
    px_noisy_min: float,
    px_noisy_max: float,
    py_noisy_min: float,
    py_noisy_max: float,
    px_cleaned_max: float,
    py_cleaned_max: float,
    rng_seed: int = 42,
):
    """Wrapper around the shared reconstruction assertions."""
    verify_pz_reconstruction(
        tracking_df,
        truth,
        model_details,
        calculate_pz,
        px_nonoise_max,
        py_nonoise_max,
        px_noisy_min,
        px_noisy_max,
        py_noisy_min,
        py_noisy_max,
        px_cleaned_max,
        py_cleaned_max,
        rng_seed,
    )


@pytest.mark.parametrize(
    "delta_p, do_apply_magnet_perturbations",
    [
        pytest.param(
            2e-4,
            False,
            id="orbit_correction_off_momentum",
        ),
        pytest.param(
            0.0,
            True,
            id="magnet_perturbations_on_momentum",
        ),
    ],
)
@pytest.mark.slow
def test_calculate_pz_with_corrections_and_perturbations(
    delta_p,
    do_apply_magnet_perturbations,
    seq_b1,
    tmp_path,
    xsuite_json_path,
):
    """Test momentum reconstruction with orbit correction and/or magnet perturbations.

    Covers two scenarios:
    - orbit_correction_off_momentum: Verify reconstruction with corrected orbits
    - magnet_perturbations_on_momentum: Verify robustness to random magnet errors
    """
    tolerance_values = {
        (2e-4, False): {
            "px_nonoise_max": 1.8e-7,
            "py_nonoise_max": 1.8e-7,
            "px_noisy_min": 2e-6,
            "px_noisy_max": 2.5e-6,
            "py_noisy_min": 2e-6,
            "py_noisy_max": 2.5e-6,
            "px_cleaned_max": 5.8e-7,
            "py_cleaned_max": 5.6e-7,
        },
        (0.0, True): {
            "px_nonoise_max": 1.8e-7,
            "py_nonoise_max": 1.8e-7,
            "px_noisy_min": 2e-6,
            "px_noisy_max": 2.5e-6,
            "py_noisy_min": 2e-6,
            "py_noisy_max": 2.5e-6,
            "px_cleaned_max": 5.8e-7,
            "py_cleaned_max": 5.6e-7,
        },
    }
    json_path = xsuite_json_path("lhcb1.seq")
    test_id = f"test_{delta_p}_{do_apply_magnet_perturbations}"

    tracking_df, truth, model_details, _ = _setup_xsuite_simulation(
        delta_p,
        "all" if do_apply_magnet_perturbations else "",
        12,
        json_path,
        seq_b1,
        tmp_path,
        test_id,
    )

    tol_dict = tolerance_values[(delta_p, do_apply_magnet_perturbations)]
    _verify_pz_reconstruction(
        tracking_df,
        truth,
        model_details,
        **tol_dict,
        rng_seed=42,
    )
