r"""Build a twiss-shaped frame from an omc3 optics measurement directory.

Phase convention: every phase in this module -- the ``MUX``/``MUY`` columns, the
``Q1``/``Q2`` headers and the accumulated :class:`PhaseData` -- is a **cumulative
betatron phase in turns**, the same quantity as the model twiss ``mu1``/``mu2``.
It is *not* the ``delta`` of :mod:`tmom_recon.physics.bpm_phases`, which is the
deviation of a selected neighbour advance from a quarter turn
(:math:`\phi_{\mathrm{code}} = \phi_x - \pi/2`). The conversion between them
happens only in the neighbour finders.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
from omc3.optics_measurements.constants import (
    ALPHA,
    AMP_BETA_NAME,
    BETA,
    BETA_NAME,
    DISPERSION,
    DISPERSION_NAME,
    ERR,
    EXT,
    MOMENTUM_DISPERSION,
    NAME,
    ORBIT,
    ORBIT_NAME,
    PHASE,
    PHASE_ADV,
    PHASE_NAME,
    S,
)

LOGGER = logging.getLogger(__name__)


def build_twiss_from_measurements(
    measurement_dir: Path,
    include_errors: bool = False,
    use_amplitude_beta: bool = True,
    reverse_bpm_order: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Builds a twiss dataframe from optics measurement files.

    Reads beta, dispersion, phase, and orbit measurement files, computes cumulative
    phase advances, and combines all measurements into a single twiss dataframe.
    Only BPMs with complete measurements across all planes are included.

    Args:
        measurement_dir: Path to directory containing measurement .tfs files.
        include_errors: If True, include error columns in the output.
        use_amplitude_beta: If True, use beta from amplitude measurements (beta_amplitude_x/y.tfs).
                   If False, use beta from phase measurements (beta_phase_x/y.tfs).
        reverse_bpm_order: If True, reverse BPM ordering and accumulate phase from the last BPM.

    Returns:
        TfsDataFrame with twiss parameters, sorted by increasing phase advance.
        Boolean indicating if dispersion data was included.
    """
    # Determine which beta measurement files to use
    beta_file_name = AMP_BETA_NAME if use_amplitude_beta else BETA_NAME

    # Read all measurement files
    measurements = {
        "beta_x": _read_measurement_file(measurement_dir, beta_file_name, "x"),
        "beta_y": _read_measurement_file(measurement_dir, beta_file_name, "y"),
        "phase_x": _read_measurement_file(measurement_dir, PHASE_NAME, "x"),
        "phase_y": _read_measurement_file(measurement_dir, PHASE_NAME, "y"),
        "orbit_x": _read_measurement_file(measurement_dir, ORBIT_NAME, "x"),
        "orbit_y": _read_measurement_file(measurement_dir, ORBIT_NAME, "y"),
    }

    # Alpha is sometimes missing in amplitude beta files; fall back to phase beta files when needed
    measurements["alpha_x"], measurements["alpha_y"] = _select_alpha_measurements(
        measurements["beta_x"],
        measurements["beta_y"],
        measurement_dir,
        use_amplitude_beta,
    )
    try:
        measurements["disp_x"] = _read_measurement_file(measurement_dir, DISPERSION_NAME, "x")
        measurements["disp_y"] = _read_measurement_file(measurement_dir, DISPERSION_NAME, "y")
        dispersion_found = True
    except FileNotFoundError:
        LOGGER.warning(
            "Dispersion measurement files not found, proceeding without dispersion data."
        )
        dispersion_found = False

    # Cumulative phase accumulated from the measured (mod-1) adjacent advances.
    phase_data_x = _compute_cumulative_phase(
        measurements["phase_x"], f"{PHASE}X", reverse=reverse_bpm_order
    )
    phase_data_y = _compute_cumulative_phase(
        measurements["phase_y"], f"{PHASE}Y", reverse=reverse_bpm_order
    )

    # Log how many BPMs were found in each measurement
    for key, df in measurements.items():
        LOGGER.debug(f"Measurement '{key}' has {len(df)} BPMs.")

    # Find BPMs with complete measurements and sort by phase
    bpm_index = _get_valid_bpm_index(measurements, phase_data_x, phase_data_y)
    LOGGER.info(f"Total BPMs with complete measurements: {len(bpm_index)}")

    # Build twiss dataframe
    twiss_df = tfs.TfsDataFrame(index=bpm_index)
    twiss_df.index.name = NAME

    # Add optics measurements
    twiss_df = _add_optics_columns(twiss_df, measurements, bpm_index)
    if dispersion_found:
        twiss_df = _add_dispersion_columns(
            twiss_df, measurements, bpm_index, negate=reverse_bpm_order
        )

    # Extract phase data for valid BPMs
    twiss_df[f"{PHASE_ADV}X"] = np.array([phase_data_x.mu[bpm] for bpm in bpm_index])
    twiss_df[f"{PHASE_ADV}Y"] = np.array([phase_data_y.mu[bpm] for bpm in bpm_index])

    var_mu_x = np.array([phase_data_x.var[bpm] for bpm in bpm_index])
    var_mu_y = np.array([phase_data_y.var[bpm] for bpm in bpm_index])
    twiss_df["mu1_var"] = var_mu_x
    twiss_df["mu2_var"] = var_mu_y

    # Add error columns
    if include_errors:
        twiss_df = _add_error_columns(twiss_df, measurements, bpm_index, var_mu_x, var_mu_y)
    if dispersion_found and include_errors:
        twiss_df = _add_dispersion_error_columns(twiss_df, measurements, bpm_index)

    # Add the S column
    twiss_df[S] = measurements["beta_x"].loc[bpm_index, S].values

    # Set headers. Q1/Q2 must be the *full* betatron phase around the ring (integer +
    # fractional), because that is what the downstream phase-advance matrix uses as its
    # modulo total against the full-tune cumulative phase. The measured phase chain is open
    # (no last->first edge) and the OMC3 Q1 header is only the fractional tune, so the ring
    # total is reconstructed from both: the closing edge back to the first BPM is the tune
    # minus the summed phases, ``(Q_frac - mu_last) mod 1``, and adding it lifts the
    # cumulative (whose integer part is already accumulated in ``mu_last``) to the full tune.
    twiss_df.headers = {
        "MU1_TOTAL_VAR": phase_data_x.total_var,
        "MU2_TOTAL_VAR": phase_data_y.total_var,
        "Q1": _reconstruct_ring_tune(phase_data_x, measurements["beta_x"].headers.get("Q1")),
        "Q2": _reconstruct_ring_tune(phase_data_y, measurements["beta_x"].headers.get("Q2")),
    }

    return twiss_df, dispersion_found


def _read_measurement_file(measurement_dir: Path, base_name: str, plane: str) -> pd.DataFrame:
    """Read a single measurement file."""
    file_path = measurement_dir / f"{base_name}{plane}{EXT}"
    if not file_path.exists():
        raise FileNotFoundError(f"Measurement file not found: {file_path}")
    return tfs.read(file_path, index=NAME)


def _get_valid_bpm_index(
    measurements: dict[str, pd.DataFrame],
    phase_data_x: "PhaseData",
    phase_data_y: "PhaseData",
) -> pd.Index:
    """
    Find BPMs present in all measurements and sort by phase advance.

    Returns:
        pd.Index of valid BPM names, sorted by increasing horizontal phase advance.
    """
    # Start with intersection of all measurement indices
    valid_index = phase_data_x.index.intersection(phase_data_y.index)
    for key, df in measurements.items():
        if not key.startswith("phase_"):  # Phase data already considered
            valid_index = valid_index.intersection(df.index)

    # Sort by horizontal phase advance
    phase_values = np.array([phase_data_x.mu[bpm] for bpm in valid_index])
    return valid_index[np.argsort(phase_values)]


def _add_columns(
    twiss_df: tfs.TfsDataFrame,
    mapping: dict[str, tuple[pd.DataFrame, str]],
    bpm_index: pd.Index,
    negate: bool = False,
) -> tfs.TfsDataFrame:
    """Generic function to add columns to twiss dataframe based on a mapping."""
    out = twiss_df.copy(deep=True)
    factor = -1 if negate else 1
    for col_name, (source_df, source_col) in mapping.items():
        out[col_name] = source_df.loc[bpm_index, source_col].values * factor
    return out


def _add_optics_columns(
    twiss_df: tfs.TfsDataFrame,
    measurements: dict[str, pd.DataFrame],
    bpm_index: pd.Index,
) -> tfs.TfsDataFrame:
    """Add beta, alpha, dispersion, and orbit columns to twiss dataframe."""
    # Map source columns to twiss columns
    optics_map = {
        f"{BETA}X": (measurements["beta_x"], f"{BETA}X"),
        f"{BETA}Y": (measurements["beta_y"], f"{BETA}Y"),
        f"{ALPHA}X": (measurements["alpha_x"], f"{ALPHA}X"),
        f"{ALPHA}Y": (measurements["alpha_y"], f"{ALPHA}Y"),
        f"{ORBIT}X": (measurements["orbit_x"], f"{ORBIT}X"),
        f"{ORBIT}Y": (measurements["orbit_y"], f"{ORBIT}Y"),
    }

    return _add_columns(twiss_df, optics_map, bpm_index)


def _add_dispersion_columns(
    twiss_df: tfs.TfsDataFrame,
    measurements: dict[str, pd.DataFrame],
    bpm_index: pd.Index,
    negate: bool = False,
) -> tfs.TfsDataFrame:
    """Add dispersion columns to twiss dataframe."""
    dispersion_map_x = {
        f"{DISPERSION}X": (measurements["disp_x"], f"{DISPERSION}X"),
        f"{MOMENTUM_DISPERSION}X": (measurements["disp_x"], f"{MOMENTUM_DISPERSION}X"),
    }
    dispersion_map_y = {
        f"{DISPERSION}Y": (measurements["disp_y"], f"{DISPERSION}Y"),
        f"{MOMENTUM_DISPERSION}Y": (measurements["disp_y"], f"{MOMENTUM_DISPERSION}Y"),
    }
    out = _add_columns(twiss_df, dispersion_map_x, bpm_index, negate=negate)
    return _add_columns(out, dispersion_map_y, bpm_index)


def _add_error_columns(
    twiss_df: tfs.TfsDataFrame,
    measurements: dict[str, pd.DataFrame],
    bpm_index: pd.Index,
    var_mu_x: np.ndarray,
    var_mu_y: np.ndarray,
) -> tfs.TfsDataFrame:
    """Add error columns to twiss dataframe if requested.

    Args:
        twiss_df: Twiss dataframe to populate.
        measurements: Dict of measurement dataframes.
        bpm_index: Index of valid BPMs.
        var_mu_x: Cumulative phase variance for x plane.
        var_mu_y: Cumulative phase variance for y plane.
    """
    out = twiss_df.copy(deep=True)
    error_map = {
        f"{ERR}{BETA}X": (measurements["beta_x"], f"{ERR}{BETA}X"),
        f"{ERR}{BETA}Y": (measurements["beta_y"], f"{ERR}{BETA}Y"),
        f"{ERR}{ALPHA}X": (measurements["alpha_x"], f"{ERR}{ALPHA}X"),
        f"{ERR}{ALPHA}Y": (measurements["alpha_y"], f"{ERR}{ALPHA}Y"),
        f"{ERR}{ORBIT}X": (measurements["orbit_x"], f"{ERR}{ORBIT}X"),
        f"{ERR}{ORBIT}Y": (measurements["orbit_y"], f"{ERR}{ORBIT}Y"),
    }

    missing_error_cols = []
    for col_name, (source_df, source_col) in error_map.items():
        try:
            out[col_name] = source_df.loc[bpm_index, source_col].values
        except KeyError:
            missing_error_cols.append(source_col)

    if missing_error_cols:
        LOGGER.warning(
            "Missing error columns in measurement files, proceeding without them: %s",
            missing_error_cols,
        )

    out[f"{ERR}{PHASE_ADV}X"] = np.sqrt(var_mu_x)
    out[f"{ERR}{PHASE_ADV}Y"] = np.sqrt(var_mu_y)
    return out


def _add_dispersion_error_columns(
    twiss_df: tfs.TfsDataFrame,
    measurements: dict[str, pd.DataFrame],
    bpm_index: pd.Index,
) -> tfs.TfsDataFrame:
    """Add dispersion error columns to twiss dataframe.

    Only called when dispersion data is available.
    """
    dispersion_error_map = {
        f"{ERR}{DISPERSION}X": (measurements["disp_x"], f"{ERR}{DISPERSION}X"),
        f"{ERR}{DISPERSION}Y": (measurements["disp_y"], f"{ERR}{DISPERSION}Y"),
        f"{ERR}{MOMENTUM_DISPERSION}X": (measurements["disp_x"], f"{ERR}{MOMENTUM_DISPERSION}X"),
        f"{ERR}{MOMENTUM_DISPERSION}Y": (measurements["disp_y"], f"{ERR}{MOMENTUM_DISPERSION}Y"),
    }
    try:
        return _add_columns(twiss_df, dispersion_error_map, bpm_index)
    except KeyError:
        missing_cols = [
            col
            for (col, (df, src_col)) in dispersion_error_map.items()
            if src_col not in df.columns
        ]
        LOGGER.warning(
            f"Dispersion error columns missing: {missing_cols}. Proceeding without dispersion errors."
        )
        return twiss_df.copy(deep=True)


class PhaseData:
    """Container for cumulative phase measurement data."""

    def __init__(self, mu: dict, var: dict, total_var: float, index: pd.Index):
        self.mu = mu  # Cumulative phase at each BPM (turns)
        self.var = var  # Cumulative variance at each BPM (turns^2)
        self.total_var = total_var  # Total variance around ring (turns^2)
        self.index = index  # BPM names in phase chain


def _ordered_phase_edges(
    phase_df: pd.DataFrame, phase_col: str, *, reverse: bool
) -> tuple[list, np.ndarray, np.ndarray, float]:
    """Read per-edge phase advances by following the NAME -> NAME2 chain.

    Returns measured phase advances between adjacent BPMs in accumulation order.

    Args:
        phase_df: TFS dataframe with phase measurements (index=NAME, contains NAME2 column).
        phase_col: Column name for phase values (e.g., "PHASEX").
        reverse: If True, order the chain from the last BPM stepping backwards.

    Returns:
        ``(order, diffs, variances, total_var)`` where ``order`` is the BPM-name sequence,
        ``diffs[i]`` / ``variances[i]`` are the measured advance and variance between
        ``order[i]`` and ``order[i + 1]``, and ``total_var`` is the summed edge variance
        around the ring.

    Raises:
        ValueError: If phase_df is empty or contains NaN values.
    """
    if len(phase_df) == 0:
        raise ValueError("Phase DataFrame is empty, cannot compute cumulative phase.")

    # Follow the forward NAME -> NAME2 chain once, recording each measured edge.
    current_bpm = phase_df.index[0]
    forward_order = []
    edge_phase = {}
    edge_var = {}
    step_variances = []

    for _ in range(len(phase_df)):
        forward_order.append(current_bpm)
        next_bpm = phase_df.loc[current_bpm, "NAME2"]
        phase_advance = phase_df.loc[current_bpm, phase_col]
        phase_error = phase_df.loc[current_bpm, f"{ERR}{phase_col}"]

        if pd.isna(phase_advance) or pd.isna(phase_error):
            raise ValueError(f"NaN phase advance or error for {current_bpm} -> {next_bpm}")

        edge_phase[(current_bpm, next_bpm)] = phase_advance
        edge_var[(current_bpm, next_bpm)] = phase_error**2
        step_variances.append(phase_error**2)
        current_bpm = next_bpm

    # The loop appended every source BPM. A real OMC3 phase file is an *open* chain: it
    # gives the advance to each next BPM but has no row closing the last BPM back to the
    # first, so the final NAME2 target was visited only as a destination. Append it so its
    # cumulative phase (the sum of every measured edge) is kept rather than dropped. A
    # *closed* chain instead lands back on the first BPM, which is already in the order.
    if current_bpm != forward_order[0]:
        forward_order.append(current_bpm)

    order = list(reversed(forward_order)) if reverse else forward_order

    diffs = np.empty(len(order) - 1, dtype=float)
    variances = np.empty(len(order) - 1, dtype=float)
    for i in range(len(order) - 1):
        # In reverse the measured edge between order[i] and order[i+1] was stored in the
        # forward direction (order[i+1] -> order[i]); the advance magnitude is the same.
        key = (order[i + 1], order[i]) if reverse else (order[i], order[i + 1])
        diffs[i] = edge_phase[key]
        variances[i] = edge_var[key]

    return order, diffs, variances, float(sum(step_variances))


def _compute_cumulative_phase(
    phase_df: pd.DataFrame, phase_col: str, *, reverse: bool = False
) -> PhaseData:
    """
    Compute cumulative phase by accumulating the measured (mod-1) edge advances.

    Accumulates the modulo-one edge advances and their variances from the first
    BPM in the chain.

    Args:
        phase_df: TFS dataframe with phase measurements (index=NAME, contains NAME2 column).
        phase_col: Column name for phase values (e.g., "PHASEX").
        reverse: If True, start accumulation at the last BPM and step backwards.

    Returns:
        PhaseData object with cumulative phase information.

    Raises:
        ValueError: If phase_df is empty or contains NaN values.
    """
    order, diffs, variances, total_var = _ordered_phase_edges(phase_df, phase_col, reverse=reverse)

    mu_values = np.concatenate(([0.0], np.cumsum(diffs)))
    var_values = np.concatenate(([0.0], np.cumsum(variances)))

    return PhaseData(
        mu=dict(zip(order, mu_values)),
        var=dict(zip(order, var_values)),
        total_var=total_var,
        index=pd.Index(order),
    )


def _reconstruct_ring_tune(phase_data: "PhaseData", q_header: float | None) -> float:
    """Reconstruct the full ring tune (integer + fractional) from an open phase chain.

    Reconstructs the closing edge from the fractional tune in the ``Q`` header.
    Without that header, returns the cumulative phase before the closing edge.
    """
    mu_last = phase_data.mu[phase_data.index[-1]]
    if q_header is None:
        LOGGER.warning(
            "No Q header available to reconstruct the ring tune; using cumulative phase "
            "(%.6f) without the unmeasured closing edge.",
            mu_last,
        )
        return float(mu_last)
    q_frac = float(q_header) % 1.0
    closing_edge = (q_frac - mu_last) % 1.0
    return float(mu_last + closing_edge)


def _select_alpha_measurements(
    beta_x_df: pd.DataFrame,
    beta_y_df: pd.DataFrame,
    measurement_dir: Path,
    use_amplitude_beta: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose data source for alpha columns.

    Amplitude beta files can lack ALPHA columns; fall back to beta-from-phase files
    when needed to provide ALFA measurements.
    """

    alpha_col_x = f"{ALPHA}X"
    alpha_col_y = f"{ALPHA}Y"

    if alpha_col_x in beta_x_df.columns and alpha_col_y in beta_y_df.columns:
        return beta_x_df, beta_y_df

    if use_amplitude_beta:
        LOGGER.info(
            "Alpha columns missing from amplitude beta files; loading alpha from phase beta files."
        )
        phase_beta_x = _read_measurement_file(measurement_dir, BETA_NAME, "x")
        phase_beta_y = _read_measurement_file(measurement_dir, BETA_NAME, "y")
        return phase_beta_x, phase_beta_y

    missing_cols = []
    if alpha_col_x not in beta_x_df.columns:
        missing_cols.append(alpha_col_x)
    if alpha_col_y not in beta_y_df.columns:
        missing_cols.append(alpha_col_y)
    raise KeyError(f"Alpha columns missing from beta measurements: {missing_cols}")
