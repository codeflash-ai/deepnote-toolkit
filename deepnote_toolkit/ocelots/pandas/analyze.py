import math
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd

from deepnote_toolkit.ocelots.constants import DEEPNOTE_INDEX_COLUMN
from deepnote_toolkit.ocelots.types import ColumnsStatsRecord, ColumnStats


def _count_unique(column):
    try:
        dropped = column.dropna()
        return dropped.nunique()
    except TypeError:
        # This happens when the column contains e.g. dictionaries, lists or sets
        # In that case, we fall back to each value being unique
        return len(column)


def _get_categories(np_array):
    # Convert only once and avoid .tolist() call
    pandas_series = pd.Series(np_array)

    # special treatment for empty values
    num_nans = pandas_series.isna().sum()
    # No need for .item(), sum gives int

    # Use astype(str) efficiently only if needed
    non_na = pandas_series.dropna()
    if not non_na.empty:
        values_as_str = non_na.astype(str)
        counter = Counter(values_as_str)
    else:
        counter = Counter()

    max_items = 3
    if num_nans > 0:
        max_items -= 1  # We need to save space for "missing" category

    if len(counter) > max_items:
        most_common = counter.most_common(max_items - 1)
        # Instead of subtracting, use dict directly in sum/count
        other_counts = {k: v for k, v in counter.items() if (k, v) not in most_common}
        sum_others = sum(other_counts.values())
        num_others = len(other_counts)
        most_common.append((f"{num_others} others", sum_others))
        categories = most_common
    else:
        categories = counter.most_common(max_items)

    if num_nans > 0:
        categories.append(("Missing", num_nans))

    return [{"name": name, "count": count} for name, count in categories]


def _is_type_numeric(dtype):
    """
    Returns True if dtype is numeric, False otherwise

    Numeric means either a number (int, float, complex) or a datetime or timedelta.
    It means e.g. that a range of these values can be plotted on a histogram.
    """

    # datetime doesn't play nice with np.issubdtype, so we need to check explicitly
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(
        dtype
    ):
        return True

    try:
        return np.issubdtype(dtype, np.number)
    except TypeError:
        # np.issubdtype crashes on categorical column dtype, and also on others, e.g. geopandas types
        return False


def _get_histogram(pd_series):
    try:
        dtype = pd_series.dtype
        # Avoid repeated isna/isnull/dropna - do once
        cleaned = pd_series.replace([np.inf, -np.inf], np.nan).dropna()
        # Use astype(int) only if datetime/timedelta
        if pd.api.types.is_datetime64_any_dtype(
            dtype
        ) or pd.api.types.is_timedelta64_dtype(dtype):
            np_array = cleaned.astype(int).to_numpy()
        else:
            np_array = cleaned.to_numpy()

        # Check if array is empty after dropping NaN/NaT values
        if np_array.size == 0:
            return None

        y, bins = np.histogram(np_array, bins=10)
        return [
            {"bin_start": bins[i], "bin_end": bins[i + 1], "count": int(count)}
            for i, count in enumerate(y)
        ]
    except (ValueError, IndexError) as e:
        # NumPy 2.2+ raises "Too many bins for data range" when:
        # - Data range is zero (all values identical), or
        # - For integer data, bin width would be < 1.0, or
        # - Floating point precision prevents creating finite-sized bins at large scales
        # Numpy implementation: https://github.com/numpy/numpy/blob/e7a123b2d3eca9897843791dd698c1803d9a39c2/numpy/lib/_histograms_impl.py#L454
        # IndexError can occur in NumPy 2.x with edge cases involving large integers or datetime conversions
        if isinstance(e, ValueError) and "Too many bins for data range" in str(e):
            return None
        # For IndexError or other ValueError cases, return None to gracefully handle edge cases
        return None


def _calculate_min_max(column):
    """
    Calculate min and max values for a given column.
    """
    if _is_type_numeric(column.dtype):
        dropped = column.dropna()
        if not dropped.empty:
            min_value = str(dropped.min())
            max_value = str(dropped.max())
            return min_value, max_value
        else:
            return None, None
    return None, None


def analyze_columns(
    df: pd.DataFrame, color_scale_column_names: Optional[List[str]] = None
) -> List[ColumnsStatsRecord]:
    """
    Analyze columns in a Pandas DataFrame, but only within a certain computational limit.

    This function is used to analyze the columns of a Pandas DataFrame for display in a
    Deepnote data table. It only analyzes a certain number of columns to keep things fast.
    The number of columns it analyzes is determined by the `max_cells_to_analyze` variable,
    which is calculated so that the analysis takes no more than 100ms.

    If the user has applied color scale format rules in a data table, this function will
    calculate additional statistics for the columns that are required for display of the
    color scales.

    Args:
        df: A Pandas DataFrame to analyze.
        color_scale_column_names: A set of column names that have a color scale formatting rule applied.

    Returns:
        A list of ColumnsStatsRecord
    """

    # Analyze only certain number of columns to keep things fast
    max_cells_to_analyze = (
        100000  # calculated so that the analysis takes no more than 100ms
    )
    n_rows = len(df)
    n_cols = len(df.columns)
    if n_rows == 0:
        max_columns_to_analyze = n_cols
    else:
        max_columns_to_analyze = min(
            math.floor(max_cells_to_analyze / n_rows),
            n_cols,
        )

    # Analyze columns
    columns = [
        ColumnsStatsRecord(
            name=str(name),
            dtype=str(dtype),
        )
        for name, dtype in zip(df.columns, df.dtypes)
    ]

    # Add stats to columns, but only within computational limit
    for i in range(max_columns_to_analyze):
        column = df.iloc[:, i]
        col_info = columns[i]
        if col_info.name == DEEPNOTE_INDEX_COLUMN:
            continue  # Do not analyze DEEPNOTE_INDEX_COLUMN column

        # Avoid repeated dropna, isnull calls: combine calcs
        nan_count = column.isnull().sum()
        unique_count = _count_unique(column)

        col_info.stats = ColumnStats(unique_count=unique_count, nan_count=nan_count)

        if _is_type_numeric(column.dtype):
            min_value, max_value = _calculate_min_max(column)
            col_info.stats.min = min_value
            col_info.stats.max = max_value
            col_info.stats.histogram = _get_histogram(column)
        else:
            # Use .to_numpy() instead of np.array for potentially better performance
            col_info.stats.categories = _get_categories(column.to_numpy())

    if not color_scale_column_names:
        return columns

    # Calculate stats for additional columns if user has applied color scale format rules in a data table
    # That’s because showing color scales in columns requires min and max values
    # Use additional cell limit calculated to keep analysis under ~3s
    remaining_cells_to_analyze_for_color_scales = 10000000

    # Process remaining columns for color scale rules
    for i in range(max_columns_to_analyze, n_cols):
        # Ignore columns that are not numeric
        column = df.iloc[:, i]
        col_info = columns[i]

        # Ignore columns that are not numeric
        if not _is_type_numeric(column.dtype):
            continue

        column_name = col_info.name

        if column_name in color_scale_column_names:
            # Check if we still have budget to analyze all of the DataFrame rows for this column
            if remaining_cells_to_analyze_for_color_scales <= n_rows:
                break  # Exceeded budget, stop processing

            nan_count = column.isnull().sum()
            unique_count = _count_unique(column)

            col_info.stats = ColumnStats(unique_count=unique_count, nan_count=nan_count)

            min_value, max_value = _calculate_min_max(column)
            col_info.stats.min = min_value
            col_info.stats.max = max_value
            col_info.stats.histogram = _get_histogram(column)

            remaining_cells_to_analyze_for_color_scales -= n_rows

    return columns
