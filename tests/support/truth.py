"""Truth and optics-conversion helpers for integration tests."""

from __future__ import annotations

import pandas as pd
import tfs

from tests.reference_co import position_only_reference_from_twiss
from tmom_recon import ModelDetails, MomentumReference
from tmom_recon.model import resolve_model_details


def xsuite_to_ngtws(tbl, bpm_names: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    """Convert an Xsuite Twiss table to the MAD-NG-compatible BPM table.

    When ``bpm_names`` is supplied it is treated as the authoritative set of
    observed BPMs. This keeps PSB/LHC conversion independent of accelerator naming
    conventions such as ``.B1``-specific suffixes.
    """
    df = tbl.to_pandas()
    df["beta11"] = df["betx"]
    df["beta22"] = df["bety"]
    df["alfa11"] = df["alfx"]
    df["alfa22"] = df["alfy"]
    df["mu1"] = df["mux"]
    df["mu2"] = df["muy"]
    df = tfs.TfsDataFrame(df, headers={"q1": tbl.qx, "q2": tbl.qy})
    df["name"] = df["name"].str.upper()
    df = df.set_index("name")

    if bpm_names is not None:
        requested = {str(name).upper() for name in bpm_names}
        return df[df.index.isin(requested)]

    bpm_names = df[df.index.str.contains("BPM", case=False, regex=False)].index.tolist()
    return df[df.index.isin(bpm_names)]


def get_truth(tracking_df: pd.DataFrame, tws: pd.DataFrame) -> pd.DataFrame:
    """Extract true transverse momenta for names represented by ``tws``."""
    truth = tracking_df[["name", "turn", "px", "py"]].rename(
        columns={"px": "px_true", "py": "py_true"}
    )
    return truth[truth["name"].isin(tws.index)]


def simulated_nominal_reference_from_model(
    model_details: ModelDetails, df: pd.DataFrame
) -> MomentumReference:
    """Build the nominal-RF reference orbit for a model-backed simulation."""
    if model_details.pt != 0.0:
        raise ValueError("simulated_nominal_reference_from_model requires pt=0.0")
    names = pd.Index(pd.unique(df["name"]), name="name")
    wants_markers = any(str(name).endswith(("_BEFORE", "_AFTER")) for name in names)
    closed_orbit = resolve_model_details(
        model_details,
        observed_elements=[str(name) for name in names],
        install_ac_dipole_markers=wants_markers,
    ).closed_orbit_tws
    by_upper = closed_orbit.rename(index=lambda name: str(name).upper())
    wanted = pd.Index([str(name).upper() for name in names], name="name")
    missing = wanted.difference(by_upper.index)
    if len(missing):
        raise ValueError(f"Model twiss is missing names present in data: {list(missing)[:10]}")
    orbit = by_upper.loc[wanted].copy()
    orbit.index = names
    return position_only_reference_from_twiss(orbit, pt=model_details.pt)


__all__ = ["get_truth", "simulated_nominal_reference_from_model", "xsuite_to_ngtws"]
