import pandas as pd
import sys
from collections import defaultdict
from pathlib import Path
sys.path.append("../../../src")
from processing.loader import (
    SantosTestDataset,
)
def process_dataframe(
    df_source,
    start_date,
    end_date,
    freq="1h",
    interp_method=None,
    datetime_col="datetime",
    round_freq="5min",
):
    """
    Process a dataframe to fill gaps in the datetime column and interpolate
    missing values.
    """
    df_original = df_source.copy()

    # Remove timezone from the original dataframe
    df_original[datetime_col] = pd.to_datetime(
        df_original[datetime_col]
    ).dt.tz_localize(None)

    # Round datetime values to match existing values in the original dataframe
    df_original[datetime_col] = df_original[datetime_col].dt.round(round_freq)

    # Create a DataFrame with the new date range
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Create a DataFrame and ensure no timezone and same dtype as df_original
    df_processed = pd.DataFrame({datetime_col: date_range})
    df_processed[datetime_col] = df_processed[datetime_col].dt.tz_localize(None)

    # Ensure both DataFrames have the same datetime dtype ('<M8[ns]')
    df_processed[datetime_col] = df_processed[datetime_col].astype("datetime64[ns]")
    df_original[datetime_col] = df_original[datetime_col].astype("datetime64[ns]")

    # Merge original and processed dataframes to fill gaps in datetime
    merged_df = pd.merge_asof(
        df_processed.sort_values(datetime_col),
        df_original.sort_values(datetime_col),
        on=datetime_col,
        direction="nearest",
    )

    # Interpolate missing values if interpolation method is provided
    if interp_method:
        try:
            interpolated_df = merged_df.interpolate(
                method=interp_method, limit_direction="both"
            )
        except ValueError as e:
            raise ValueError(f"Interpolation failed: {e}")
    else:
        interpolated_df = merged_df

    final_df = interpolated_df.drop_duplicates(
        subset=datetime_col, keep="first"
    ).reset_index(drop=True)

    # Check for NaN values in columns other than 'datetime'
    check_isna = final_df.drop(columns=[datetime_col]).isnull().values.any()
    check_sum = final_df.drop(columns=[datetime_col]).isnull().values.sum()
    if check_isna and interp_method:
        raise ValueError(f"The df contains NaN values in cols. {check_sum}")

    return final_df