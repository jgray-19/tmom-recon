"""Shared utilities for momentum reconstruction integration tests."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tfs
from omc3.optics_measurements.constants import (
    ERR,
    EXT,
    ORBIT,
    ORBIT_NAME,
)
from omc3.scripts.fake_measurement_from_model import generate as generate_fake_measurement
from pymadng_utils.accelerators import LHC
from pymadng_utils.madx import convert_tfs_to_madx

from tests.reference_co import momentum_reference_from_twiss
from tmom_recon import ModelDetails, MomentumReference, calculate_pz, inject_noise_xy
from tmom_recon.model import resolve_model_details
from tmom_recon.svd import svd_clean_measurements

if TYPE_CHECKING:
    from collections.abc import Callable

    import xtrack as xt

LOGGER = logging.getLogger(__name__)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))


def model_details_for(accelerator, *, pt: float) -> ModelDetails:
    """Build :class:`ModelDetails` with tunes read from a tracking twiss.

    ``calculate_pz`` generates the MAD-NG optics itself; the test only supplies the
    accelerator, target tunes and momentum.
    """
    return ModelDetails(
        accelerator=accelerator,
        pt=float(pt),
    )


def lhc_model_details(seq_file: str, data_dir, tws, *, delta_p: float = 0.0) -> ModelDetails:
    """Build LHC :class:`ModelDetails` matching a tracking setup's optics."""
    accelerator = LHC(beam=1, sequence_file=data_dir / "sequences" / seq_file, kinetic_energy=6800)
    return model_details_for(accelerator, pt=accelerator.dp2pt(delta_p))


def momentum_reference_from_model(
    model_details: ModelDetails, df: pd.DataFrame
) -> MomentumReference:
    """The nominal-RF reference closed orbit, taken from the model's own twiss.

    Deliberately *not* the turn mean of the tracking data. That data is driven,
    so its per-BPM mean carries the AC dipole's forced offset rather than the
    closed orbit -- on ``lhcb1``, whose true orbit is zero, the mean is ~4e-5 m
    of pure contamination and lands in ``px`` as ~1.4e-6.

    A hardcoded zero is equally wrong the other way: the crossing optics run
    separation bumps of several mm, which the pt estimate would read as a
    momentum offset. The model orbit is right in both cases because these
    simulated machines carry no errors the model lacks, so model and machine
    closed orbits agree by construction. On a real machine this substitution is
    illegal -- an unknown dipole-error orbit is exactly degenerate with the
    dispersive orbit, so the reference must be measured.
    """
    names = pd.Index(pd.unique(df["name"]), name="name")
    # The reference has to cover every name the data carries, which on the PSB
    # includes the AC-dipole before/after markers. Those only exist in the twiss
    # when the markers are installed, so ask for them when the data needs them.
    wants_markers = any(str(name).endswith(("_BEFORE", "_AFTER")) for name in names)
    closed_orbit = resolve_model_details(
        model_details,
        observed_elements=[str(name) for name in names],
        install_ac_dipole_markers=wants_markers,
    ).closed_orbit_tws
    # MAD-NG emits the inserted markers as ``..._before``/``..._after`` while the
    # tracking data carries them uppercased, so match on case-folded names.
    by_upper = closed_orbit.rename(index=lambda name: str(name).upper())
    wanted = pd.Index([str(name).upper() for name in names], name="name")
    missing = wanted.difference(by_upper.index)
    if len(missing):
        raise ValueError(
            f"Model twiss is missing {len(missing)} name(s) present in the data: "
            f"{sorted(map(str, missing))[:10]}"
        )
    orbit = by_upper.loc[wanted].copy()
    orbit.index = names
    # closed_orbit_tws is always evaluated at deltap=0.0 (see
    # resolve_model_details), i.e. the nominal-RF orbit, regardless of
    # model_details.pt -- so the reference origin is 0.0, not model_details.pt.
    return momentum_reference_from_twiss(orbit, pt=0.0)


def transverse_calc(
    df: pd.DataFrame,
    model_details: ModelDetails,
    reference: MomentumReference,
    *,
    ac_dipole_config=None,
    **kwargs,
) -> pd.DataFrame:
    """Model-only reconstruction without dispersion (old transverse behaviour).

    *reference* is positional and required on purpose. It used to default to a
    zero orbit, which silently produced wrong answers twice: the crossing optics
    run mm-scale separation bumps that the pt estimate then reads as momentum.
    Build it with :func:`momentum_reference_from_model`.
    """
    result = calculate_pz(
        df,
        model_details,
        reference=reference,
        use_dispersion=False,
        acd=ac_dipole_config,
        **kwargs,
    )
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    return result


def dispersive_calc(
    df: pd.DataFrame,
    model_details: ModelDetails,
    reference: MomentumReference,
    *,
    ac_dipole_config=None,
    **kwargs,
) -> pd.DataFrame:
    """Model-only reconstruction with dispersion (old dispersive behaviour).

    *reference* is positional and required on purpose. It used to default to a
    zero orbit, which silently produced wrong answers twice: the crossing optics
    run mm-scale separation bumps that the pt estimate then reads as momentum.
    Build it with :func:`momentum_reference_from_model`.
    """
    result = calculate_pz(
        df,
        model_details,
        reference=reference,
        use_dispersion=True,
        acd=ac_dipole_config,
        **kwargs,
    )
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
    return result


def xsuite_to_ngtws(tbl: xt.Table) -> pd.DataFrame:
    """Convert xsuite twiss table to ngtws format DataFrame.

    Args:
        line: xsuite Line object containing the twiss table.

    Returns:
        DataFrame in ngtws format.
    """
    df = tbl.to_pandas()
    df["beta11"] = df["betx"]
    df["beta22"] = df["bety"]
    df["alfa11"] = df["alfx"]
    df["alfa22"] = df["alfy"]
    df["mu1"] = df["mux"]
    df["mu2"] = df["muy"]
    df = tfs.TfsDataFrame(
        df,
        headers={"q1": tbl.qx, "q2": tbl.qy},
    )
    # remove
    df["name"] = df["name"].str.upper()  # ty:ignore[unresolved-attribute]
    df = df.set_index("name")
    bpm_names = df[df.index.str.match(r"^BPM.*\.B1$")].index.tolist()
    return df[df.index.isin(bpm_names)]


def get_truth(tracking_df: pd.DataFrame, tws: pd.DataFrame) -> pd.DataFrame:
    """Extract truth momenta and prepare twiss from baseline line.

    Parameters
    ----------
    baseline_line : xtrack.Line
        The baseline accelerator line.
    tracking_df : pd.DataFrame
        The tracking DataFrame containing actual (true) momenta.

    Returns
    -------
    truth : pd.DataFrame
        DataFrame with true momenta (px_true, py_true).
    """
    df = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    # Ensure only BPMs present in twiss are included
    return df[df["name"].isin(tws.index)]


def verify_pz_reconstruction(
    tracking_df,
    truth: pd.DataFrame,
    model_details: ModelDetails,
    calculate_pz_func: Callable[..., pd.DataFrame],  # Assuming return type is Any; adjust if needed
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
    """Verify momentum reconstruction with noise and SVD cleaning.

    Tests three scenarios: clean data, noisy data, and SVD-cleaned data.
    Verifies that: (1) clean reconstruction meets accuracy thresholds,
    (2) noisy reconstruction degrades in expected range, and
    (3) SVD cleaning significantly improves reconstruction.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        The tracking data containing measurements.
    truth : pd.DataFrame
        The true momentum values (px_true, py_true).
    tws : tfs.TfsDataFrame
        Twiss parameters.
    calculate_pz_func : callable
        Function to calculate momentum (e.g., calculate_pz or calculate_transverse_pz).
    px_nonoise_max : float
        Maximum acceptable RMSE for nonoise px reconstruction.
    py_nonoise_max : float
        Maximum acceptable RMSE for nonoise py reconstruction.
    px_noisy_min : float
        Minimum expected RMSE for noisy px.
    px_noisy_max : float
        Maximum acceptable RMSE for noisy px.
    py_noisy_min : float
        Minimum expected RMSE for noisy py.
    py_noisy_max : float
        Maximum acceptable RMSE for noisy py.
    px_cleaned_max : float
        Maximum acceptable RMSE for SVD-cleaned px.
    py_cleaned_max : float
        Maximum acceptable RMSE for SVD-cleaned py.
    py_divisor : float
        Divisor to verify SVD improvement for py.
    rng_seed : int
        Random seed for noise generation.
    """
    # The nominal-RF reference must come from the model, not a hardcoded zero:
    # these setups perturb magnets and run orbit correction, and the model
    # carries both, so the machine's closed orbit is non-zero by construction.
    reference = momentum_reference_from_model(model_details, tracking_df)

    no_noise_result = calculate_pz_func(
        tracking_df.copy(deep=True),
        model_details,
        reference=reference,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    rng = np.random.default_rng(rng_seed)
    noisy_df = tracking_df.copy(deep=True)
    noisy_df = inject_noise_xy(noisy_df, rng, noise_std=1e-4)
    noisy_result = calculate_pz_func(
        noisy_df,
        model_details,
        reference=reference,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    # Apply SVD cleaning to noisy data
    cleaned_df = svd_clean_measurements(noisy_df)
    cleaned_noise_result = calculate_pz_func(
        cleaned_df,
        model_details,
        reference=reference,
        info=True,
    ).rename(columns={"px": "px_calc", "py": "py_calc"})

    merged_no_noise = truth.merge(
        no_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )
    merged_noisy = truth.merge(
        noisy_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    merged_cleaned = truth.merge(
        cleaned_noise_result[["name", "turn", "px_calc", "py_calc"]],
        on=["name", "turn"],
    )

    assert len(merged_no_noise) == len(truth)
    assert len(merged_noisy) == len(truth)
    assert len(merged_cleaned) == len(truth)

    px_rmse_nonoise = rmse(
        merged_no_noise["px_true"].to_numpy(),
        merged_no_noise["px_calc"].to_numpy(),
    )
    py_rmse_nonoise = rmse(
        merged_no_noise["py_true"].to_numpy(),
        merged_no_noise["py_calc"].to_numpy(),
    )
    px_rmse_noisy = rmse(
        merged_noisy["px_true"].to_numpy(),
        merged_noisy["px_calc"].to_numpy(),
    )
    py_rmse_noisy = rmse(
        merged_noisy["py_true"].to_numpy(),
        merged_noisy["py_calc"].to_numpy(),
    )
    px_rmse_cleaned = rmse(
        merged_cleaned["px_true"].to_numpy(),
        merged_cleaned["px_calc"].to_numpy(),
    )
    py_rmse_cleaned = rmse(
        merged_cleaned["py_true"].to_numpy(),
        merged_cleaned["py_calc"].to_numpy(),
    )

    LOGGER.info(
        f"PX RMSE no noise: {px_rmse_nonoise:.2e}, noisy: {px_rmse_noisy:.2e}, cleaned: {px_rmse_cleaned:.2e}"
    )
    LOGGER.info(
        f"PY RMSE no noise: {py_rmse_nonoise:.2e}, noisy: {py_rmse_noisy:.2e}, cleaned: {py_rmse_cleaned:.2e}"
    )

    assert px_rmse_nonoise < px_nonoise_max, (
        f"PX no-noise RMSE {px_rmse_nonoise:.2e} should be < {px_nonoise_max:.2e}"
    )
    assert py_rmse_nonoise < py_nonoise_max, (
        f"PY no-noise RMSE {py_rmse_nonoise:.2e} should be < {py_nonoise_max:.2e}"
    )
    assert px_noisy_min < px_rmse_noisy < px_noisy_max, (
        f"PX noisy RMSE {px_rmse_noisy:.2e} should be in ({px_noisy_min:.2e}, {px_noisy_max:.2e})"
    )
    assert py_noisy_min < py_rmse_noisy < py_noisy_max, (
        f"PY noisy RMSE {py_rmse_noisy:.2e} should be in ({py_noisy_min:.2e}, {py_noisy_max:.2e})"
    )
    # Check cleaned is better than noisy and meets absolute threshold
    assert px_rmse_cleaned < px_rmse_noisy, (
        f"PX cleaned {px_rmse_cleaned:.2e} should be < noisy {px_rmse_noisy:.2e}"
    )
    assert py_rmse_cleaned < py_rmse_noisy, (
        f"PY cleaned {py_rmse_cleaned:.2e} should be < noisy {py_rmse_noisy:.2e}"
    )
    assert px_rmse_cleaned < px_cleaned_max, (
        f"PX cleaned RMSE {px_rmse_cleaned:.2e} should be < {px_cleaned_max:.2e}"
    )
    assert py_rmse_cleaned < py_cleaned_max, (
        f"PY cleaned RMSE {py_rmse_cleaned:.2e} should be < {py_cleaned_max:.2e}"
    )


def add_error_to_orbit_measurement(fldr):
    for plane in ["x", "y"]:
        meas_file = fldr / f"{ORBIT_NAME}{plane}{EXT}"
        df = tfs.read(meas_file)
        df[f"{ERR}{ORBIT}{plane.upper()}"] = 1e-6
        tfs.write(meas_file, df)


def assert_dispersive_measurement_recovers_pt(
    tracking_df: pd.DataFrame,
    ng_tws: pd.DataFrame,
    truth: pd.DataFrame,
    meas_dir,
    expected_pt: float,
    model_details: ModelDetails,
    *,
    px_rmse_max: float,
    py_rmse_max: float,
    reverse_meas_tws: bool = False,
):
    """Reconstruct momenta from fake measurements and check pt and RMSE vs truth.

    Generates an omc3 fake measurement from ``ng_tws`` (MAD-NG format), injects a
    constant orbit error, runs :func:`calculate_pz` against ``measurement_dir`` and
    asserts that the estimated pt matches the caller-provided ``expected_pt`` and
    that the reconstructed ``px``/``py`` match ``truth`` within the given RMSE
    thresholds.

    Returns the reconstruction result so callers can make extra assertions.
    """
    madx_tws = convert_tfs_to_madx(ng_tws, remove_drifts=False)

    generate_fake_measurement(
        twiss=madx_tws,
        outputdir=meas_dir,
        parameters=["BETX", "BETY", "DX", "DY", "PHASEX", "PHASEY", "X", "Y"],
    )

    # Add a nonzero orbit error
    add_error_to_orbit_measurement(meas_dir)

    # The function handles closed orbit removal and px/py restoration internally.
    result = calculate_pz(
        tracking_df.copy(deep=True),
        model_details,
        reference=momentum_reference_from_model(model_details, tracking_df),
        measurement_dir=str(meas_dir),
        reverse_meas_tws=reverse_meas_tws,
        info=False,
    )
    assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"

    pt_est = result.attrs["PT_EST"]
    assert abs(pt_est - expected_pt) < 1e-5, (
        f"PT_EST {pt_est:.2e} not close to true {expected_pt:.2e}"
    )

    expected_cols = ["name", "turn", "x", "y", "px", "py"]
    assert all(col in result.columns for col in expected_cols)

    merged = truth.merge(
        result[["name", "turn", "px", "py"]],
        on=["name", "turn"],
    )

    px_rmse = rmse(merged["px_true"].to_numpy(), merged["px"].to_numpy())
    py_rmse = rmse(merged["py_true"].to_numpy(), merged["py"].to_numpy())

    print(f"Dispersive measurement px RMSE: {px_rmse:.2e}, py RMSE: {py_rmse:.2e}")

    assert px_rmse < px_rmse_max, f"px RMSE {px_rmse:.2e} > {px_rmse_max:.2e}"
    assert py_rmse < py_rmse_max, f"py RMSE {py_rmse:.2e} > {py_rmse_max:.2e}"
    return result
