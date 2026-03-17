from __future__ import annotations

from pathlib import Path

import tfs

from tmom_recon.acd.madng_driver import ACDipoleMadDriver

AC_DIPOLE_ELEMENT = "MKQA.6L4.B1"


def _full_xsuite_to_ngtws(tbl) -> tfs.TfsDataFrame:
    df = tbl.to_pandas()
    df["beta11"] = df["betx"]
    df["beta22"] = df["bety"]
    df["alfa11"] = df["alfx"]
    df["alfa22"] = df["alfy"]
    df["mu1"] = df["mux"]
    df["mu2"] = df["muy"]
    df["name"] = df["name"].astype(str).str.upper()
    return tfs.TfsDataFrame(
        df.set_index("name"),
        headers={"q1": tbl.qx, "q2": tbl.qy},
    )


def _get_driver(
    seq: Path,
    first_bpm: str,
    deltap: float = 0.0,
    *,
    debug: bool = False,
    mad_logfile: Path | None = None,
) -> ACDipoleMadDriver:
    return ACDipoleMadDriver(
        beam=1,
        beam_energy=6800,
        deltap=deltap,
        sequence_file=seq,
        start_bpm=first_bpm,
        observed_elements=AC_DIPOLE_ELEMENT,
        debug=debug,
        mad_logfile=mad_logfile,
    )


def _ac_dipole_segment_around_element(
    full_tws,
    available_bpms,
    *,
    element_name: str = AC_DIPOLE_ELEMENT,
) -> tuple[str, str]:
    tws_df = full_tws.to_pandas()
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
