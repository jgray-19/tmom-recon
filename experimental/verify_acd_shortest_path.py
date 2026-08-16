from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.acd.acd_test_helpers import (  # noqa: E402
    AC_DIPOLE_ELEMENT,
    _ac_dipole_segment_around_element,
    _get_driver,
)
from tests.momentum.momentum_test_utils import rmse  # noqa: E402


def _raw_mad_track(
    model,
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
    print(range_name, direction, track_df)
    return (
        track_df.reset_index(drop=True)
        .groupby("id", sort=False, as_index=False)
        .tail(1)
        .sort_values("id", kind="stable")
        .reset_index(drop=True)
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sequence_file = repo_root / "tests" / "data" / "sequences" / "lhcb1.seq"
    model = _get_driver(sequence_file, debug=False)

    bpm_upstream, bpm_downstream = _ac_dipole_segment_around_element(
        model.twiss_elements,
        available_bpms=[
            name
            for name in model.twiss_elements.index.astype(str)
            if str(name).upper().startswith("BPM")
        ],
        element_name=AC_DIPOLE_ELEMENT,
    )
    up_states = np.array(
        [
            [1.0e-4, 2.0e-6, -1.5e-4, -3.0e-6],
            [-0.8e-4, 1.2e-6, 0.9e-4, 2.5e-6],
            [0.5e-4, -2.2e-6, -0.4e-4, 1.8e-6],
        ],
        dtype=float,
    )
    down_states = np.array(
        [
            [0.7e-4, -1.5e-6, 1.1e-4, -2.1e-6],
            [-1.1e-4, 2.4e-6, -0.6e-4, 1.7e-6],
            [0.3e-4, 0.8e-6, 0.2e-4, -1.2e-6],
        ],
        dtype=float,
    )

    model_up_to_acd = model.track_particles(bpm_upstream, AC_DIPOLE_ELEMENT, up_states, direction=1)
    model_down_to_acd = model.track_particles(
        bpm_downstream, AC_DIPOLE_ELEMENT, down_states, direction=-1
    )

    raw_up_to_acd = _raw_mad_track(
        model,
        range_name=f"{bpm_upstream}/{AC_DIPOLE_ELEMENT}",
        direction=1,
        states=up_states,
    )[["x", "px", "y", "py"]].to_numpy(dtype=float)
    raw_down_to_acd = _raw_mad_track(
        model,
        range_name=f"{bpm_downstream}/{AC_DIPOLE_ELEMENT}",
        direction=-1,
        states=down_states,
    )[["x", "px", "y", "py"]].to_numpy(dtype=float)
    raw_down_wrong = _raw_mad_track(
        model,
        range_name=f"{bpm_downstream}/{AC_DIPOLE_ELEMENT}",
        direction=1,
        states=down_states,
    )[["x", "px", "y", "py"]].to_numpy(dtype=float)

    print(f"Upstream BPM:   {bpm_upstream}")
    print(f"Downstream BPM: {bpm_downstream}")
    print(f"ACD marker:     {AC_DIPOLE_ELEMENT}")
    print()
    print("RMSE(wrapper vs raw observe=0 MAD)")
    print(
        f"  upstream -> ACD, dir=+1: {rmse(model_up_to_acd.reshape(-1), raw_up_to_acd.reshape(-1)):.3e}"
    )
    print(
        f"  downstream -> ACD, dir=-1: {rmse(model_down_to_acd.reshape(-1), raw_down_to_acd.reshape(-1)):.3e}"
    )
    print()
    print("RMSE(correct shortest path vs wrong forward choice from downstream)")
    print(
        f"  correct dir=-1 vs wrapper: {rmse(model_down_to_acd.reshape(-1), raw_down_to_acd.reshape(-1)):.3e}"
    )
    print(
        f"  wrong   dir=+1 vs wrapper: {rmse(model_down_to_acd.reshape(-1), raw_down_wrong.reshape(-1)):.3e}"
    )


if __name__ == "__main__":
    main()
