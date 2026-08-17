from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _gd_optimal_threshold(singular_values: np.ndarray, rows: int, cols: int) -> int:
    """Return the Gavish-Donoho hard-threshold rank estimate.

    Args:
        singular_values: Singular values in descending order.
        rows: Number of matrix rows.
        cols: Number of matrix columns.

    Returns:
        Estimated rank, clipped to at least one mode.

    Note:
        This uses the common polynomial approximation for the universal
        singular value threshold from Gavish and Donoho (2014):
        ``tau = median(s) * (0.56*beta**3 - 0.95*beta**2 + 1.82*beta + 1.43)``,
        where ``beta`` is the matrix aspect ratio in ``(0, 1]``.
    """
    beta = rows / cols if rows <= cols else cols / rows
    coefficient = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    threshold = np.median(singular_values) * coefficient
    chosen_rank = int(np.sum(singular_values > threshold))
    logger.debug("GD optimal threshold retained %d modes", max(chosen_rank, 1))
    return max(chosen_rank, 1)


def _known_noise_threshold_rank(
    singular_values: np.ndarray,
    rows: int,
    cols: int,
    threshold_scale: float,
) -> tuple[int, float]:
    """Return the rank above the known-noise singular-value edge.

    The caller should pass singular values from a whitened matrix whose noise
    entries have unit variance. For an independent Gaussian noise matrix, the
    largest noise singular value concentrates near ``sqrt(rows) + sqrt(cols)``.
    """
    threshold = float(threshold_scale) * (np.sqrt(rows) + np.sqrt(cols))
    chosen_rank = int(np.sum(singular_values > threshold))
    logger.debug("Known-noise SVD threshold %.6g retained %d modes", threshold, chosen_rank)
    return chosen_rank, threshold


def _fill_small_nans(matrix: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Interpolate short NaN gaps along each BPM column.

    Args:
        matrix: Input matrix with turns on rows and BPMs on columns.
        max_gap: Largest contiguous NaN span to interpolate. Longer spans are
            left as missing values.

    Returns:
        Matrix with short gaps interpolated and long gaps preserved.
    """
    matrix_frame = pd.DataFrame(matrix)
    filled_frame = matrix_frame.copy()
    for column in matrix_frame.columns:
        series = matrix_frame[column]
        na_mask = series.isna().to_numpy()
        filled_series = series.interpolate(limit_direction="both")
        if na_mask.any():
            long_gaps: list[tuple[int, int]] = []
            gap_start: int | None = None
            for index, is_missing in enumerate(na_mask):
                if is_missing and gap_start is None:
                    gap_start = index
                at_gap_end = (not is_missing) or index == len(na_mask) - 1
                if at_gap_end and gap_start is not None:
                    gap_stop = index if is_missing else index - 1
                    if (gap_stop - gap_start + 1) > max_gap:
                        long_gaps.append((gap_start, gap_stop))
                    gap_start = None
            for gap_start, gap_stop in long_gaps:
                filled_series.iloc[gap_start : gap_stop + 1] = np.nan
        filled_frame[column] = filled_series
    logger.debug("Interpolated short NaN gaps in %d BPM columns", len(matrix_frame.columns))
    return filled_frame.to_numpy(dtype=float)


def _pivot_to_matrix(
    meas_df: pd.DataFrame, component: str, turn_range: np.ndarray, bpm_list: list[str]
) -> np.ndarray:
    """Pivot one long-format component into a turn-by-BPM matrix.

    Args:
        meas_df: Long-format measurement table.
        component: Column name to pivot.
        turn_range: Consecutive turn range for reindexing.
        bpm_list: BPM ordering for the matrix columns.

    Returns:
        Pivoted matrix.
    """
    pivoted = (
        meas_df.pivot(index="turn", columns="name", values=component)
        .reindex(index=turn_range)
        .reindex(columns=bpm_list)
    )
    return pivoted.to_numpy(dtype=float)


def _compute_centre(
    matrix: np.ndarray, centre: str | None, weights: np.ndarray | None = None
) -> np.ndarray | float:
    """Compute the centring offset for an SVD cleaning pass.

    Args:
        matrix: Matrix to centre.
        centre: One of ``"bpm"``, ``"global"``, or ``None``.
        weights: Optional non-negative weights with the same shape as
            ``matrix``.

    Returns:
        Per-column means, a scalar global mean, or ``0.0`` when no centring is
        requested.
    """
    if centre is None:
        return 0.0
    if weights is None:
        if centre == "bpm":
            return np.nanmean(matrix, axis=0, keepdims=True)
        if centre == "global":
            return np.nanmean(matrix, keepdims=True)
        return 0.0

    safe_weights = np.where(np.isfinite(matrix), weights, 0.0)
    weighted_values = np.where(np.isfinite(matrix), matrix * safe_weights, 0.0)
    if centre == "bpm":
        numerator = np.sum(weighted_values, axis=0, keepdims=True)
        denominator = np.sum(safe_weights, axis=0, keepdims=True)
    elif centre == "global":
        numerator = np.array([[np.sum(weighted_values)]], dtype=float)
        denominator = np.array([[np.sum(safe_weights)]], dtype=float)
    else:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
        )


def _select_rank(
    shape: tuple[int, ...],
    singular_values: np.ndarray,
    rank: int | str,
    *,
    known_noise_threshold_scale: float | None = None,
) -> tuple[int, float | None]:
    """Resolve the requested SVD rank.

    Args:
        shape: Shape of the decomposed matrix.
        singular_values: Singular values from the decomposition.
        rank: Integer rank or ``"auto"``.
        known_noise_threshold_scale: Optional multiplier for the whitened
            known-noise singular-value edge. When provided, ``rank`` must be
            ``"known_noise"``.

    Returns:
        Chosen rank and the known-noise threshold when applicable.
    """
    rows, cols = shape
    logger.debug(
        "SVD completed for %dx%d matrix with %d singular values", rows, cols, len(singular_values)
    )
    if rank == "known_noise":
        if known_noise_threshold_scale is None:
            raise ValueError("known-noise rank selection requires a threshold scale")
        chosen_rank, threshold = _known_noise_threshold_rank(
            singular_values, rows, cols, known_noise_threshold_scale
        )
        return chosen_rank, threshold

    chosen_rank = (
        _gd_optimal_threshold(singular_values, rows, cols) if rank == "auto" else int(rank)
    )
    logger.debug("Chosen rank: %d", chosen_rank)
    return chosen_rank, None


def _truncated_svd(
    matrix: np.ndarray,
    rank: int | str,
    *,
    known_noise_threshold_scale: float | None = None,
) -> tuple[np.ndarray, int, np.ndarray, float | None]:
    """Reconstruct a matrix from its leading singular modes.

    Args:
        matrix: Input matrix to decompose.
        rank: Integer rank or ``"auto"``.
        known_noise_threshold_scale: Optional multiplier for the whitened
            known-noise singular-value edge.

    Returns:
        Reconstructed matrix, chosen rank, full singular value spectrum, and
        known-noise threshold when applicable.
    """
    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    chosen_rank, threshold = _select_rank(
        matrix.shape,
        singular_values,
        rank,
        known_noise_threshold_scale=known_noise_threshold_scale,
    )
    reconstructed = (
        u_matrix[:, :chosen_rank]
        @ np.diag(singular_values[:chosen_rank])
        @ vt_matrix[:chosen_rank, :]
    )
    return reconstructed, chosen_rank, singular_values, threshold


def _prepare_column_scales(
    variance_matrix: np.ndarray,
    observed_mask: np.ndarray,
    component: str,
) -> np.ndarray:
    """Validate variances and derive one standard deviation per device column.

    Args:
        variance_matrix: Variance matrix aligned with the measurement matrix.
        observed_mask: Boolean mask for entries with observed measurements.
        component: Name of the cleaned plane for error messages.

    Returns:
        Row vector of per-device standard deviations.

    Raises:
        ValueError: If an observed entry lacks a finite positive variance.
    """
    invalid_mask = observed_mask & (~np.isfinite(variance_matrix) | (variance_matrix <= 0.0))
    if np.any(invalid_mask):
        n_invalid_bpms = int(np.any(invalid_mask, axis=0).sum())
        logger.warning(
            "Ignoring %d BPM(s) with non-finite or non-positive %s variances in weighted SVD",
            n_invalid_bpms,
            component,
        )
        observed_mask = observed_mask & ~invalid_mask

    column_scales = np.full((1, variance_matrix.shape[1]), np.nan, dtype=float)
    for column_index in range(variance_matrix.shape[1]):
        column_variances = variance_matrix[:, column_index][observed_mask[:, column_index]]
        if column_variances.size > 0:
            column_scales[0, column_index] = np.sqrt(float(np.median(column_variances)))
    return column_scales


def _svd_clean_matrix(
    matrix: np.ndarray,
    *,
    centre: str | None,
    rank: int | str,
    max_nan_gap: int,
    variance_matrix: np.ndarray | None = None,
    variance_name: str = "measurement",
    fix_zero_columns: bool = False,
    known_noise_threshold_scale: float | None = None,
) -> tuple[np.ndarray, int, np.ndarray, float | None]:
    """Clean one measurement plane with truncated SVD.

    Args:
        matrix: Turn-by-BPM measurement matrix.
        centre: One of ``"bpm"``, ``"global"``, or ``None``.
        rank: Integer rank or ``"auto"``.
        max_nan_gap: Largest measurement gap to interpolate before
            decomposition.
        variance_matrix: Optional variance matrix. When provided, the cleaner
            derives one standard deviation per device column, whitens each
            column before applying SVD, and rescales afterwards.
        variance_name: Plane name used in validation messages.
        known_noise_threshold_scale: Optional multiplier for the whitened
            known-noise singular-value edge. Requires ``variance_matrix``.

    Returns:
        Cleaned matrix, chosen rank, full singular value spectrum, and
        known-noise threshold when applicable.
    """
    if known_noise_threshold_scale is not None and variance_matrix is None:
        raise ValueError("known-noise SVD cleaning requires measurement variances")

    if not fix_zero_columns:
        # Exact zeros are an OP dead-reading workaround signalling a faulty BPM, not
        # real measurements. Match harpy: drop the whole BPM (column) if any of its
        # turns is exactly zero, so it propagates out as NaN for harpy to flag too.
        matrix = matrix.copy()
        matrix[:, np.any(matrix == 0.0, axis=0)] = np.nan
    missing_mask = np.isnan(matrix)
    filled_matrix = _fill_small_nans(matrix, max_gap=max_nan_gap)

    if variance_matrix is None:
        centre_offset = _compute_centre(filled_matrix, centre)
        centred_matrix = np.nan_to_num(filled_matrix - centre_offset, copy=False)
        reconstructed, chosen_rank, singular_values, threshold = _truncated_svd(
            centred_matrix, rank
        )
        cleaned_matrix = reconstructed + centre_offset
    else:
        column_scales = _prepare_column_scales(
            variance_matrix, ~missing_mask, component=variance_name
        )
        invalid_columns = np.isnan(column_scales[0])
        if np.any(invalid_columns):
            missing_mask[:, invalid_columns] = True
            filled_matrix[:, invalid_columns] = np.nan
        weights = np.broadcast_to(1.0 / (column_scales**2), filled_matrix.shape)
        centre_offset = _compute_centre(filled_matrix, centre, weights=weights)
        centred_matrix = filled_matrix - centre_offset
        scaled_matrix = np.nan_to_num(centred_matrix / column_scales, copy=False)
        svd_rank = "known_noise" if known_noise_threshold_scale is not None else rank
        reconstructed, chosen_rank, singular_values, threshold = _truncated_svd(
            scaled_matrix,
            svd_rank,
            known_noise_threshold_scale=known_noise_threshold_scale,
        )
        cleaned_matrix = reconstructed * column_scales + centre_offset

    cleaned_matrix[missing_mask] = np.nan
    return cleaned_matrix, chosen_rank, singular_values, threshold


def _finalise_cleaned_frame(
    meas_df: pd.DataFrame,
    turn_range: np.ndarray,
    bpm_list: list[str],
    x_cleaned: np.ndarray,
    y_cleaned: np.ndarray,
    *,
    rank_x: int,
    rank_y: int,
    singular_values_x: np.ndarray,
    singular_values_y: np.ndarray,
    centre: str | None,
    weighted: bool,
    known_noise_threshold_x: float | None = None,
    known_noise_threshold_y: float | None = None,
) -> pd.DataFrame:
    """Merge cleaned matrices back into the original measurement frame.

    Args:
        meas_df: Original long-format measurement frame.
        turn_range: Consecutive turn range used for pivoting.
        bpm_list: BPM ordering used for pivoting.
        x_cleaned: Cleaned horizontal matrix.
        y_cleaned: Cleaned vertical matrix.
        rank_x: Chosen rank for the horizontal plane.
        rank_y: Chosen rank for the vertical plane.
        singular_values_x: Full horizontal singular spectrum.
        singular_values_y: Full vertical singular spectrum.
        centre: Centring mode used for cleaning.
        weighted: Whether weighted SVD cleaning was used.
        known_noise_threshold_x: Horizontal known-noise SVD threshold, if used.
        known_noise_threshold_y: Vertical known-noise SVD threshold, if used.

    Returns:
        Original data with cleaned ``x`` and ``y`` columns and SVD metadata in
        ``attrs``.
    """
    cleaned_df = pd.DataFrame(
        {
            "turn": np.repeat(turn_range, len(bpm_list)),
            "name": np.tile(bpm_list, len(turn_range)),
            "x": x_cleaned.reshape(-1),
            "y": y_cleaned.reshape(-1),
        }
    )

    result = meas_df.set_index(["turn", "name"])
    cleaned_indexed = cleaned_df.set_index(["turn", "name"])
    result["x"] = cleaned_indexed["x"]
    result["y"] = cleaned_indexed["y"]
    result.attrs["svd_rank_x"] = rank_x
    result.attrs["svd_rank_y"] = rank_y
    result.attrs["svd_singular_values_x"] = tuple(float(value) for value in singular_values_x)
    result.attrs["svd_singular_values_y"] = tuple(float(value) for value in singular_values_y)
    result.attrs["centre"] = centre
    result.attrs["center"] = centre
    result.attrs["svd_weighted"] = weighted
    if known_noise_threshold_x is not None:
        result.attrs["svd_known_noise_threshold_x"] = known_noise_threshold_x
    if known_noise_threshold_y is not None:
        result.attrs["svd_known_noise_threshold_y"] = known_noise_threshold_y
    return result.reset_index()


def svd_clean_measurements(
    meas_df: pd.DataFrame,
    bpm_list: list[str] | None = None,
    center: str | None = "bpm",
    rank: int | str = "auto",
    max_nan_gap: int = 5,
    fix_zero_columns: bool = False,
) -> pd.DataFrame:
    """Clean BPM measurements with truncated SVD.

    Args:
        meas_df: Long-format measurement table containing at least ``turn``,
            ``name``, ``x``, and ``y`` columns.
        bpm_list: Optional BPM ordering. When omitted, the order from
            ``meas_df`` is used.
        center: Centring mode passed to the cleaner. Accepted values are
            ``"bpm"``, ``"global"``, or ``None``.
        rank: Integer rank or ``"auto"`` to use the Gavish-Donoho threshold.
        max_nan_gap: Largest contiguous missing span to interpolate before
            decomposition.

    Returns:
        Copy of the input measurements with cleaned ``x`` and ``y`` values.
    """
    logger.info("Starting SVD cleaning of measurements")
    if bpm_list is None:
        bpm_list = meas_df["name"].unique().tolist()

    turn_range = np.arange(int(meas_df["turn"].min()), int(meas_df["turn"].max()) + 1)
    logger.debug("Processing %d BPMs over %d turns", len(bpm_list), len(turn_range))

    x_matrix = _pivot_to_matrix(meas_df, "x", turn_range, bpm_list)
    y_matrix = _pivot_to_matrix(meas_df, "y", turn_range, bpm_list)
    x_cleaned, rank_x, singular_values_x, _ = _svd_clean_matrix(
        x_matrix,
        centre=center,
        rank=rank,
        max_nan_gap=max_nan_gap,
        fix_zero_columns=fix_zero_columns,
    )
    y_cleaned, rank_y, singular_values_y, _ = _svd_clean_matrix(
        y_matrix,
        centre=center,
        rank=rank,
        max_nan_gap=max_nan_gap,
        fix_zero_columns=fix_zero_columns,
    )

    logger.info("SVD cleaning completed successfully")
    return _finalise_cleaned_frame(
        meas_df,
        turn_range,
        bpm_list,
        x_cleaned,
        y_cleaned,
        rank_x=rank_x,
        rank_y=rank_y,
        singular_values_x=singular_values_x,
        singular_values_y=singular_values_y,
        centre=center,
        weighted=False,
    )


def weighted_svd_clean_measurements(
    meas_df: pd.DataFrame,
    bpm_list: list[str] | None = None,
    center: str | None = "bpm",
    rank: int | str = "auto",
    max_nan_gap: int = 5,
    fix_zero_columns: bool = False,
) -> pd.DataFrame:
    """Clean BPM measurements with variance-weighted truncated SVD.

    Args:
        meas_df: Long-format measurement table containing at least ``turn``,
            ``name``, ``x``, ``y``, ``var_x``, and ``var_y`` columns.
        bpm_list: Optional BPM ordering. When omitted, the order from
            ``meas_df`` is used.
        center: Centring mode passed to the cleaner. Accepted values are
            ``"bpm"``, ``"global"``, or ``None``.
        rank: Integer rank or ``"auto"`` to use the Gavish-Donoho threshold.
        max_nan_gap: Largest contiguous missing span to interpolate before
            decomposition.

    Returns:
        Copy of the input measurements with cleaned ``x`` and ``y`` values.

    Raises:
        ValueError: If ``var_x`` or ``var_y`` are missing, or if observed
            measurements do not have finite positive variances.
    """
    logger.info("Starting weighted SVD cleaning of measurements")
    required_columns = {"x", "y", "turn", "var_x", "var_y"}
    missing_columns = required_columns.difference(meas_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Weighted SVD cleaning requires columns: {missing}")

    if bpm_list is None:
        bpm_list = meas_df["name"].unique().tolist()

    turn_range = np.arange(int(meas_df["turn"].min()), int(meas_df["turn"].max()) + 1)
    logger.debug("Processing %d BPMs over %d turns", len(bpm_list), len(turn_range))

    x_matrix = _pivot_to_matrix(meas_df, "x", turn_range, bpm_list)
    y_matrix = _pivot_to_matrix(meas_df, "y", turn_range, bpm_list)
    var_x_matrix = _pivot_to_matrix(meas_df, "var_x", turn_range, bpm_list)
    var_y_matrix = _pivot_to_matrix(meas_df, "var_y", turn_range, bpm_list)

    x_cleaned, rank_x, singular_values_x, _ = _svd_clean_matrix(
        x_matrix,
        centre=center,
        rank=rank,
        max_nan_gap=max_nan_gap,
        variance_matrix=var_x_matrix,
        variance_name="horizontal measurement",
        fix_zero_columns=fix_zero_columns,
    )
    y_cleaned, rank_y, singular_values_y, _ = _svd_clean_matrix(
        y_matrix,
        centre=center,
        rank=rank,
        max_nan_gap=max_nan_gap,
        variance_matrix=var_y_matrix,
        variance_name="vertical measurement",
        fix_zero_columns=fix_zero_columns,
    )

    logger.info("Weighted SVD cleaning completed successfully")
    return _finalise_cleaned_frame(
        meas_df,
        turn_range,
        bpm_list,
        x_cleaned,
        y_cleaned,
        rank_x=rank_x,
        rank_y=rank_y,
        singular_values_x=singular_values_x,
        singular_values_y=singular_values_y,
        centre=center,
        weighted=True,
    )


def known_noise_svd_clean_measurements(
    meas_df: pd.DataFrame,
    bpm_list: list[str] | None = None,
    center: str | None = "bpm",
    threshold_scale: float = 1.0,
    max_nan_gap: int = 5,
    fix_zero_columns: bool = False,
) -> pd.DataFrame:
    """Clean BPM measurements with a variance-informed SVD noise cut.

    This is the known-variance counterpart to :func:`weighted_svd_clean_measurements`.
    It whitens each BPM column by the supplied ``var_x``/``var_y`` values, keeps
    singular modes above ``threshold_scale * (sqrt(n_turns) + sqrt(n_bpms))``,
    then rescales back to physical units.

    Args:
        meas_df: Long-format measurement table containing at least ``turn``,
            ``name``, ``x``, ``y``, ``var_x``, and ``var_y`` columns.
        bpm_list: Optional BPM ordering. When omitted, the order from
            ``meas_df`` is used.
        center: Centring mode passed to the cleaner. Accepted values are
            ``"bpm"``, ``"global"``, or ``None``.
        threshold_scale: Multiplier on the whitened random-matrix noise edge.
            Values above one are more conservative and cut more modes.
        max_nan_gap: Largest contiguous missing span to interpolate before
            decomposition.

    Returns:
        Copy of the input measurements with cleaned ``x`` and ``y`` values.

    Raises:
        ValueError: If variance columns are missing or ``threshold_scale`` is
            not finite and positive.
    """
    logger.info("Starting known-noise SVD cleaning of measurements")
    if not np.isfinite(threshold_scale) or threshold_scale <= 0.0:
        raise ValueError("threshold_scale must be finite and positive")

    required_columns = {"x", "y", "turn", "var_x", "var_y"}
    missing_columns = required_columns.difference(meas_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Known-noise SVD cleaning requires columns: {missing}")

    if bpm_list is None:
        bpm_list = meas_df["name"].unique().tolist()

    turn_range = np.arange(int(meas_df["turn"].min()), int(meas_df["turn"].max()) + 1)
    logger.debug("Processing %d BPMs over %d turns", len(bpm_list), len(turn_range))

    x_matrix = _pivot_to_matrix(meas_df, "x", turn_range, bpm_list)
    y_matrix = _pivot_to_matrix(meas_df, "y", turn_range, bpm_list)
    var_x_matrix = _pivot_to_matrix(meas_df, "var_x", turn_range, bpm_list)
    var_y_matrix = _pivot_to_matrix(meas_df, "var_y", turn_range, bpm_list)

    x_cleaned, rank_x, singular_values_x, threshold_x = _svd_clean_matrix(
        x_matrix,
        centre=center,
        rank="known_noise",
        max_nan_gap=max_nan_gap,
        variance_matrix=var_x_matrix,
        variance_name="horizontal measurement",
        fix_zero_columns=fix_zero_columns,
        known_noise_threshold_scale=threshold_scale,
    )
    y_cleaned, rank_y, singular_values_y, threshold_y = _svd_clean_matrix(
        y_matrix,
        centre=center,
        rank="known_noise",
        max_nan_gap=max_nan_gap,
        variance_matrix=var_y_matrix,
        variance_name="vertical measurement",
        fix_zero_columns=fix_zero_columns,
        known_noise_threshold_scale=threshold_scale,
    )

    logger.info("Known-noise SVD cleaning completed successfully")
    return _finalise_cleaned_frame(
        meas_df,
        turn_range,
        bpm_list,
        x_cleaned,
        y_cleaned,
        rank_x=rank_x,
        rank_y=rank_y,
        singular_values_x=singular_values_x,
        singular_values_y=singular_values_y,
        centre=center,
        weighted=True,
        known_noise_threshold_x=threshold_x,
        known_noise_threshold_y=threshold_y,
    )


# Trajectory-matrix (SSA) cleaning. Unlike the matrix cleaners above, this
# operates independently on each BPM's continuous turn-by-turn stream.


def _trajectory_matrix(series: np.ndarray, window: int) -> np.ndarray:
    """Return a read-only Hankel view with the requested window length."""
    n_columns = series.shape[0] - window + 1
    return np.lib.stride_tricks.as_strided(
        series,
        shape=(window, n_columns),
        strides=(series.strides[0], series.strides[0]),
        writeable=False,
    )


def _ssa_clean_series(series: np.ndarray, *, window: int, rank: int) -> np.ndarray:
    """Reconstruct one continuous stream from its leading trajectory modes."""
    trajectory = _trajectory_matrix(series, window)
    u_mat, s_mat, vt_mat = np.linalg.svd(trajectory, full_matrices=False)
    keep = min(rank, s_mat.size)
    reconstructed = (u_mat[:, :keep] * s_mat[:keep]) @ vt_mat[:keep]

    # Diagonal averaging: every sample appears on one anti-diagonal of the
    # trajectory matrix, so averaging that anti-diagonal turns the (no longer
    # Hankel) reconstruction back into a single series.
    n_samples = series.shape[0]
    totals = np.zeros(n_samples, dtype=float)
    counts = np.zeros(n_samples, dtype=float)
    for offset in range(window):
        totals[offset : offset + reconstructed.shape[1]] += reconstructed[offset]
        counts[offset : offset + reconstructed.shape[1]] += 1.0
    return totals / counts


def ssa_clean_measurements(
    meas_df: pd.DataFrame,
    bpm_list: list[str] | None = None,
    *,
    window: int = 200,
    rank: int = 4,
    max_nan_gap: int = 5,
) -> pd.DataFrame:
    """Clean each BPM's turn-by-turn stream with trajectory-matrix SVD.

    Each BPM is cleaned independently along the turn axis.

    Args:
        meas_df: Long-format table with ``turn``, ``name``, ``x`` and ``y``.
        bpm_list: Optional BPM ordering. Defaults to the order in ``meas_df``.
        window: Embedding length in turns.
        rank: Number of trajectory modes to keep per BPM.
        max_nan_gap: Largest contiguous missing span to interpolate before
            decomposition. Longer gaps stay missing.

    Returns:
        Copy of the input with cleaned ``x`` and ``y``.

    Raises:
        ValueError: If ``window`` or ``rank`` is not usable for the data length.
    """
    logger.info("Starting SSA (continuous, per-BPM) cleaning of measurements")
    if bpm_list is None:
        bpm_list = meas_df["name"].unique().tolist()

    turn_range = np.arange(int(meas_df["turn"].min()), int(meas_df["turn"].max()) + 1)
    if not 1 < window <= turn_range.size:
        raise ValueError(f"window must be in (1, n_turns={turn_range.size}], got {window}")
    if not 0 < rank <= window:
        raise ValueError(f"rank must be in (0, window={window}], got {rank}")

    cleaned: dict[str, np.ndarray] = {}
    for component in ("x", "y"):
        matrix = _pivot_to_matrix(meas_df, component, turn_range, bpm_list)
        missing_mask = np.isnan(matrix)
        filled = _fill_small_nans(matrix, max_gap=max_nan_gap)
        # Centre per BPM so the mean (the closed orbit) is never spent on a
        # trajectory mode, and restore it afterwards.
        offsets = np.nanmean(filled, axis=0, keepdims=True)
        centred = np.nan_to_num(filled - offsets, copy=False)
        out = np.empty_like(centred)
        for index in range(centred.shape[1]):
            out[:, index] = _ssa_clean_series(
                np.ascontiguousarray(centred[:, index]), window=window, rank=rank
            )
        out += offsets
        out[missing_mask] = np.nan
        cleaned[component] = out

    result = meas_df.set_index(["turn", "name"])
    frame = pd.DataFrame(
        {
            "turn": np.repeat(turn_range, len(bpm_list)),
            "name": np.tile(bpm_list, len(turn_range)),
            "x": cleaned["x"].reshape(-1),
            "y": cleaned["y"].reshape(-1),
        }
    ).set_index(["turn", "name"])
    result["x"] = frame["x"]
    result["y"] = frame["y"]
    result.attrs["ssa_window"] = window
    result.attrs["ssa_rank"] = rank
    logger.info("SSA cleaning completed successfully")
    return result.reset_index()
