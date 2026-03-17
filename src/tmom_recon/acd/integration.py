from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .madng_driver import ACDipoleMadDriver
from .reconstruction import calculate_ac_dipole_momentum


@dataclass(frozen=True)
class ACDipoleConfig:
    ac_dipole_marker: str
    model: ACDipoleMadDriver
    bpm_upstream: str | None = None
    bpm_downstream: str | None = None
    n_bpms_each_side: int = 1
    smooth_lambda: float = 1.0


def apply_ac_dipole_bpm_overrides_inplace(
    result: pd.DataFrame,
    data: pd.DataFrame,
    tws: pd.DataFrame,
    config: ACDipoleConfig,
) -> None:
    """Patch px/py for the BPMs around the AC dipole using cleaned ACD estimates.

    The function runs ACD reconstruction once and replaces ``px``/``py`` values
    in ``result`` at the selected upstream/downstream BPMs by matching on
    ``(name, turn)``.
    """

    data_for_acd = data.copy(deep=True)
    if "var_x" not in data_for_acd.columns:
        data_for_acd["var_x"] = 1.0
    if "var_y" not in data_for_acd.columns:
        data_for_acd["var_y"] = 1.0

    acd_result = calculate_ac_dipole_momentum(
        data_for_acd,
        tws,
        ac_dipole_marker=config.ac_dipole_marker,
        model=config.model,
        bpm_upstream=config.bpm_upstream,
        bpm_downstream=config.bpm_downstream,
        n_bpms_each_side=config.n_bpms_each_side,
        smooth_lambda=config.smooth_lambda,
        inject_noise=False,
    )

    bpm_upstream = str(acd_result.attrs.get("bpm_upstream", acd_result["bpm_upstream"].iloc[0]))
    bpm_downstream = str(
        acd_result.attrs.get("bpm_downstream", acd_result["bpm_downstream"].iloc[0])
    )

    side_specs = [
        (
            bpm_upstream,
            "px_bpm_upstream_cleaned",
            "py_bpm_upstream_cleaned",
        ),
        (
            bpm_downstream,
            "px_bpm_downstream_cleaned",
            "py_bpm_downstream_cleaned",
        ),
    ]

    for bpm_name, px_col, py_col in side_specs:
        side = acd_result[["turn", px_col, py_col]].rename(columns={px_col: "px", py_col: "py"})
        side = side.set_index("turn")
        mask = result["name"].astype(str) == bpm_name
        if not mask.any():
            continue
        turns = result.loc[mask, "turn"].to_numpy()
        result.loc[mask, "px"] = side.reindex(turns)["px"].to_numpy(dtype=float)
        result.loc[mask, "py"] = side.reindex(turns)["py"].to_numpy(dtype=float)
