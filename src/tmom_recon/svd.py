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


def _select_rank(shape: tuple[int, ...], singular_values: np.ndarray, rank: int | str) -> int:
    """Resolve the requested SVD rank.

    Args:
        shape: Shape of the decomposed matrix.
        singular_values: Singular values from the decomposition.
        rank: Integer rank or ``"auto"``.

    Returns:
        Chosen rank.
    """
    rows, cols = shape
    logger.debug(
        "SVD completed for %dx%d matrix with %d singular values", rows, cols, len(singular_values)
    )
    chosen_rank = (
        _gd_optimal_threshold(singular_values, rows, cols) if rank == "auto" else int(rank)
    )
    logger.debug("Chosen rank: %d", chosen_rank)
    return chosen_rank


def _truncated_svd(matrix: np.ndarray, rank: int | str) -> tuple[np.ndarray, int, np.ndarray]:
    """Reconstruct a matrix from its leading singular modes.

    Args:
        matrix: Input matrix to decompose.
        rank: Integer rank or ``"auto"``.

    Returns:
        Reconstructed matrix, chosen rank, and full singular value spectrum.
    """
    u_matrix, singular_values, vt_matrix = np.linalg.svd(matrix, full_matrices=False)
    chosen_rank = _select_rank(matrix.shape, singular_values, rank)
    reconstructed = (
        u_matrix[:, :chosen_rank]
        @ np.diag(singular_values[:chosen_rank])
        @ vt_matrix[:chosen_rank, :]
    )
    return reconstructed, chosen_rank, singular_values


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
        raise ValueError(
            f"Weighted SVD cleaning requires finite positive {component} variances for all observed values"
        )

    column_scales = np.full((1, variance_matrix.shape[1]), np.nan, dtype=float)
    for column_index in range(variance_matrix.shape[1]):
        column_variances = variance_matrix[:, column_index][observed_mask[:, column_index]]
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
) -> tuple[np.ndarray, int, np.ndarray]:
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

    Returns:
        Cleaned matrix, chosen rank, and full singular value spectrum.
    """
    missing_mask = np.isnan(matrix)
    filled_matrix = _fill_small_nans(matrix, max_gap=max_nan_gap)

    if variance_matrix is None:
        centre_offset = _compute_centre(filled_matrix, centre)
        centred_matrix = np.nan_to_num(filled_matrix - centre_offset, copy=False)
        reconstructed, chosen_rank, singular_values = _truncated_svd(centred_matrix, rank)
        cleaned_matrix = reconstructed + centre_offset
    else:
        column_scales = _prepare_column_scales(
            variance_matrix, ~missing_mask, component=variance_name
        )
        weights = np.broadcast_to(1.0 / (column_scales**2), filled_matrix.shape)
        centre_offset = _compute_centre(filled_matrix, centre, weights=weights)
        centred_matrix = filled_matrix - centre_offset
        scaled_matrix = np.nan_to_num(centred_matrix / column_scales, copy=False)
        reconstructed, chosen_rank, singular_values = _truncated_svd(scaled_matrix, rank)
        cleaned_matrix = reconstructed * column_scales + centre_offset

    cleaned_matrix[missing_mask] = np.nan
    return cleaned_matrix, chosen_rank, singular_values


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
    return result.reset_index()


def svd_clean_measurements(
    meas_df: pd.DataFrame,
    bpm_list: list[str] | None = None,
    center: str | None = "bpm",
    rank: int | str = "auto",
    max_nan_gap: int = 5,
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
    x_cleaned, rank_x, singular_values_x = _svd_clean_matrix(
        x_matrix, centre=center, rank=rank, max_nan_gap=max_nan_gap
    )
    y_cleaned, rank_y, singular_values_y = _svd_clean_matrix(
        y_matrix, centre=center, rank=rank, max_nan_gap=max_nan_gap
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

    x_cleaned, rank_x, singular_values_x = _svd_clean_matrix(
        x_matrix,
        centre=center,
        rank=rank,
        max_nan_gap=max_nan_gap,
        variance_matrix=var_x_matrix,
        variance_name="horizontal measurement",
    )
    y_cleaned, rank_y, singular_values_y = _svd_clean_matrix(
        y_matrix,
        centre=center,
        rank=rank,
        max_nan_gap=max_nan_gap,
        variance_matrix=var_y_matrix,
        variance_name="vertical measurement",
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
