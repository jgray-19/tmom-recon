from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pymadng_utils.accelerators import LHC

from tmom_recon.acd.madng_driver import ACDipoleMadDriver

AC_DIPOLE_ELEMENT = "MKQA.6L4.B1"


def _get_driver(
    seq: Path,
    pt: float = 0.0,
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
        pt=pt,
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


def r_squared(true, pred) -> float:
    """Coefficient of determination of ``pred`` against ``true``."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _truth_at(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Per-turn ``x``/``px``/``y``/``py`` truth at element ``name`` from tracking data."""
    rows = df.loc[df["name"] == name, ["turn", "x", "px", "y", "py"]].sort_values("turn")
    return rows.rename(
        columns={"x": "x_true", "px": "px_true", "y": "y_true", "py": "py_true"}
    ).reset_index(drop=True)


def acd_state_marker_names(model: ACDipoleMadDriver) -> tuple[str, str]:
    """Return the upper-cased ``(before, after)`` AC-dipole state-marker names."""
    return (
        model.accelerator.acd_marker_name("before").upper(),
        model.accelerator.acd_marker_name("after").upper(),
    )


def assert_acd_momenta_match_truth(
    result: pd.DataFrame,
    tracking_df: pd.DataFrame,
    model: ACDipoleMadDriver,
    *,
    clean: bool,
    kick_r2_min: float,
    bpm_r2_min: float,
    marker_r2_min: float,
    marker_pos_r2_min: float = 0.999,
) -> None:
    """Assert an AC-dipole reconstruction matches the tracked truth.

    Checks, against the directly tracked truth, that:

    - the harmonic kick fit reconstructs the true kick (after - before markers),
    - with no noise the internal observed-vs-fit R^2 is essentially perfect,
    - the BPM momenta agree with truth (raw when ``clean``, cleaned under noise),
    - the momenta at the ``<acd>_before`` / ``<acd>_after`` markers agree with truth,
    - the ``x``/``y`` positions at those markers (obtained by tracking the BPM
      states to the marker) agree with truth.

    Args:
        result: The full reconstruction result (state rows; summary in ``attrs``).
        tracking_df: Tracking data, including the ``<acd>_before`` / ``<acd>_after``
            marker rows used as truth.
        model: MAD-NG driver, used for the accelerator's marker names.
        clean: Whether the reconstruction ran on noiseless data.
        kick_r2_min: Minimum R^2 for the kick fit vs the true kick.
        bpm_r2_min: Minimum R^2 for the BPM momenta vs truth.
        marker_r2_min: Minimum R^2 for the marker momenta vs truth.
        marker_pos_r2_min: Minimum R^2 for the marker ``x``/``y`` positions vs truth.
    """
    summary = result.attrs["summary"]
    bpm_upstream = result.attrs["bpm_upstream"]
    bpm_downstream = result.attrs["bpm_downstream"]
    before_marker, after_marker = acd_state_marker_names(model)
    before_truth = _truth_at(tracking_df, before_marker)
    after_truth = _truth_at(tracking_df, after_marker)

    # The harmonic kick fit reconstructs the true kick (after - before markers).
    true_kick = after_truth.merge(before_truth, on="turn", suffixes=("_a", "_b"))
    true_kick["dpx_true"] = true_kick["px_true_a"] - true_kick["px_true_b"]
    true_kick["dpy_true"] = true_kick["py_true_a"] - true_kick["py_true_b"]
    kick = summary.merge(true_kick[["turn", "dpx_true", "dpy_true"]], on="turn", how="inner")
    dpx_r2 = r_squared(kick["dpx_true"], kick["dpx_fit_rad"])
    dpy_r2 = r_squared(kick["dpy_true"], kick["dpy_fit_rad"])
    assert dpx_r2 > kick_r2_min, f"Kick dpx R^2={dpx_r2} below threshold {kick_r2_min}"
    assert dpy_r2 > kick_r2_min, f"Kick dpy R^2={dpy_r2} below threshold {kick_r2_min}"
    if clean:
        assert result.attrs["dpx_r2"] > 0.999
        assert result.attrs["dpy_r2"] > 0.999

    # BPM momenta agree with truth (cleaned outputs are noise-robust).
    bpm_col = "{plane}_bpm_{side}" if clean else "{plane}_bpm_{side}_cleaned"
    for side, bpm in (("upstream", bpm_upstream), ("downstream", bpm_downstream)):
        merged = summary.merge(_truth_at(tracking_df, bpm), on="turn", how="inner")
        for plane in ("px", "py"):
            r2 = r_squared(
                merged[f"{plane}_true"].to_numpy(),
                merged[bpm_col.format(plane=plane, side=side)].to_numpy(),
            )
            assert r2 > bpm_r2_min, f"{plane}_bpm_{side} R^2={r2}"

    # Momenta and positions at the AC dipole `before` / `after` markers agree with
    # truth. The marker x/y positions are produced purely by tracking the BPM
    # states to the marker (no kick fit), so verifying them confirms the tracked
    # position is carried through to the output unmodified.
    state_rows = result.assign(name=result["name"].astype(str).str.upper())
    for marker_name, marker_truth in ((before_marker, before_truth), (after_marker, after_truth)):
        rows = state_rows.loc[state_rows["name"] == marker_name].merge(
            marker_truth, on="turn", how="inner"
        )
        assert len(rows) == len(marker_truth)
        for plane in ("px", "py"):
            r2 = r_squared(rows[f"{plane}_true"].to_numpy(), rows[plane].to_numpy())
            assert r2 > marker_r2_min, f"{marker_name} {plane} R^2={r2}"
        for coord in ("x", "y"):
            assert rows[coord].notna().all(), f"{marker_name} {coord} has NaNs"
            r2 = r_squared(rows[f"{coord}_true"].to_numpy(), rows[coord].to_numpy())
            assert r2 > marker_pos_r2_min, f"{marker_name} {coord} position R^2={r2}"
