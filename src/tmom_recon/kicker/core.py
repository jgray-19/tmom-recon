from __future__ import annotations

import pandas as pd

from tmom_recon.lattice.transport import solve_kick_4d_least_squares, transport_matrix_4d_from_twiss


def check_data_index(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure the table is indexed by element name."""
    data = data.copy()
    if data.index.name is None:
        if "name" in data.columns:
            data = data.set_index("name")
        elif "NAME" in data.columns:
            data = data.set_index("NAME")
        else:
            raise ValueError("Data must be indexed by BPM name or have a 'name' or 'NAME' column.")
    elif not isinstance(data.index.name, str) or data.index.name.lower() != "name":
        raise ValueError("Data index must be named 'name' or 'NAME'.")
    return data


def subtract_closed_orbit(
    data: pd.DataFrame,
    n_turns_free: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove the closed orbit from the measured positions."""
    data = check_data_index(data)

    closed_orbit = data.groupby(data.index).apply(
        lambda x: x[x["turn"] < n_turns_free][["x", "y"]].mean()
    )
    closed_orbit_uncertainty = data.groupby(data.index).apply(
        lambda x: x[x["turn"] < n_turns_free][["x", "y"]].std()
    )

    data["x"] = data["x"] - data.index.map(closed_orbit["x"])
    data["y"] = data["y"] - data.index.map(closed_orbit["y"])
    return data, closed_orbit, closed_orbit_uncertainty


def find_kick(
    data: pd.DataFrame,
    n_turns_free: int = 1000,
) -> tuple[str, int]:
    """Find the first turn with a clear post-kick displacement."""
    data, _, _ = subtract_closed_orbit(data, n_turns_free)
    per_turn_max = data.groupby("turn")[["x", "y"]].apply(lambda d: float(d.abs().max().max()))

    pre_kick_median = float(per_turn_max[per_turn_max.index < n_turns_free].median())
    threshold = 100.0 * pre_kick_median if pre_kick_median > 0 else 1e-10

    post_kick = per_turn_max[(per_turn_max.index >= n_turns_free) & (per_turn_max > threshold)]
    if post_kick.empty:
        raise ValueError("No kicks found above the specified threshold.")
    kick_turn = int(post_kick.index.min())

    kick_turn_data = data[data["turn"] == kick_turn]
    kick_bpm = str(kick_turn_data[["x", "y"]].abs().max(axis=1).idxmax())
    return kick_bpm, kick_turn


def reconstruct_momentum_kick(
    data: pd.DataFrame,
    twiss: pd.DataFrame,
    n_turns_free: int = 1000,
    n_turns_after_kick: int = 3,
) -> pd.DataFrame:
    r"""Reconstruct the instantaneous kick from the first downstream BPM response.

    Workflow:

    1. Remove the closed orbit so that the kick is reconstructed from the
       oscillatory displacement only.
    2. Detect the first turn with a clear post-kick excursion and the BPM that
       shows the largest displacement on that turn (*kick_bpm*).
    3. Build the uncoupled block-diagonal 4×4 transport matrix from the kicker
       to *kick_bpm* using the Courant-Snyder Twiss parameterisation (see
       :func:`tmom_recon.lattice.transport.transport_matrix_4d_from_twiss`).
    4. Solve for ``(px_kick, py_kick)`` via
       :func:`tmom_recon.lattice.transport.solve_kick_4d_least_squares` using
       the measured ``(x, y)`` displacement at *kick_bpm* on the kick turn.
    5. Return a trimmed data frame covering *n_turns_after_kick* turns starting
       from the kick turn, with upstream BPMs on the kick turn removed and
       ``px``/``py`` columns set (non-zero only at the kick BPM on the kick
       turn).

    Args:
        data: Turn-by-turn BPM tracking data.  Must be indexed by element name
            or have a ``"name"`` / ``"NAME"`` column; must contain ``"turn"``,
            ``"x"``, and ``"y"`` columns.
        twiss: Twiss table indexed by element name, including a ``"kicker"``
            row.  Required columns: ``beta11``, ``alfa11``, ``mu1``,
            ``beta22``, ``alfa22``, ``mu2``, ``s``.
        n_turns_free: Number of pre-kick turns used to estimate the closed
            orbit.  The kick is searched for in turns ≥ *n_turns_free*.
        n_turns_after_kick: Number of post-kick turns to retain in the output,
            starting from the kick turn (inclusive).

    Returns:
        DataFrame with the same columns as *data* plus ``"px"`` and ``"py"``.
        Rows cover turns ``[kick_turn, kick_turn + n_turns_after_kick)``.
        BPMs whose *s* position is upstream of *kick_bpm* are dropped from the
        kick turn.  ``"px"`` and ``"py"`` are non-zero only at *kick_bpm* on
        the kick turn, where they hold the reconstructed kick momenta.
    """
    twiss = check_data_index(twiss)

    kick_bpm, kick_turn = find_kick(data, n_turns_free)
    data, _, _ = subtract_closed_orbit(data, n_turns_free)

    row = data[(data["turn"] == kick_turn) & (data.index == kick_bpm)].iloc[0]
    mat = transport_matrix_4d_from_twiss(twiss, source="kicker", target=kick_bpm)
    delta_px, delta_py = solve_kick_4d_least_squares(
        {kick_bpm: mat},
        {kick_bpm: float(row["x"])},
        {kick_bpm: float(row["y"])},
    )

    data = data[
        (data["turn"] >= kick_turn) & (data["turn"] < kick_turn + n_turns_after_kick)
    ].copy()

    bpm_order = twiss.loc[data.index, "s"]
    kick_bpm_s = float(twiss.loc[kick_bpm, "s"])
    data = data[~((data["turn"] == kick_turn) & (bpm_order < kick_bpm_s))]
    data["px"] = 0.0
    data["py"] = 0.0
    data.loc[(data["turn"] == kick_turn) & (data.index == kick_bpm), "px"] = delta_px
    data.loc[(data["turn"] == kick_turn) & (data.index == kick_bpm), "py"] = delta_py

    return data


def extract_n_turns_after_kick(
    data: pd.DataFrame,
    n_turns_free: int = 1000,
) -> pd.DataFrame:
    """Extract turns immediately following the identified kick."""
    data, _, _ = subtract_closed_orbit(data, n_turns_free)
    _kick_bpm, kick_turn = find_kick(data, n_turns_free)
    return data[(data["turn"] >= kick_turn) & (data["turn"] < kick_turn + n_turns_free)]
