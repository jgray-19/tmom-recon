from __future__ import annotations

from pathlib import Path

import pandas as pd
from pymadng_utils.accelerators import LHC

from tmom_recon.acd.madng_driver import ACDipoleMadDriver

AC_DIPOLE_ELEMENT = "MKQA.6L4.B1"


def _get_driver(
    seq: Path,
    deltap: float = 0.0,
    *,
    debug: bool = False,
    mad_logfile: Path | None = None,
) -> ACDipoleMadDriver:
    return ACDipoleMadDriver(
        accelerator=LHC(
            beam=1,
            sequence_file=seq,
            kinetic_energy=6800,
        ),
        deltap=deltap,
        observed_elements=AC_DIPOLE_ELEMENT,
        debug=debug,
        mad_logfile=mad_logfile,
    )


def _ac_dipole_segment_around_element(
    twiss_elements,
    available_bpms,
    *,
    element_name: str = AC_DIPOLE_ELEMENT,
) -> tuple[str, str]:
    if hasattr(twiss_elements, "to_pandas"):
        tws_df = twiss_elements.to_pandas()
    else:
        tws_df = pd.DataFrame(twiss_elements).reset_index()

    if "name" not in tws_df.columns:
        first_col = str(tws_df.columns[0])
        tws_df = tws_df.rename(columns={first_col: "name"})

    tws_df = tws_df.assign(name=tws_df["name"].astype(str).str.upper())

    target = str(element_name).upper()
    target_rows = tws_df[tws_df["name"] == target]
    if target_rows.empty:
        raise ValueError(f"Element {target} not found in full-element twiss")
    target_s = float(target_rows.iloc[0]["s"])

    bpm_df = tws_df[tws_df["name"].str.match(r"^BPM.*\.B1$")][["name", "s"]].drop_duplicates(
        subset="name",
        keep="first",
    )
    available_set = {str(name).upper() for name in available_bpms}
    bpm_df = bpm_df[bpm_df["name"].isin(available_set)].sort_values("s").reset_index(drop=True)

    upstream_rows = bpm_df[bpm_df["s"] <= target_s]
    downstream_rows = bpm_df[bpm_df["s"] > target_s]
    upstream = (
        str(upstream_rows.iloc[-1]["name"])
        if not upstream_rows.empty
        else str(bpm_df.iloc[-1]["name"])
    )
    downstream = (
        str(downstream_rows.iloc[0]["name"])
        if not downstream_rows.empty
        else str(bpm_df.iloc[0]["name"])
    )
    if upstream == downstream:
        raise ValueError("Could not determine distinct upstream/downstream BPMs for AC dipole")
    return upstream, downstream
