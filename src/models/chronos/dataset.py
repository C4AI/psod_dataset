
import pathlib
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_dataframe(df_source,
                        start_date,
                        end_date,
                        freq='1h',
                        interp_method=None,
                        datetime_col='datetime',
                        round_freq='5min'):
    """
    Process a dataframe to fill gaps in the datetime column and interpolate
    missing values.

    Parameters:
    df_source (pd.DataFrame): Source dataframe.
    start_date (str or pd.Timestamp): Start date for the date range.
    end_date (str or pd.Timestamp): End date for the date range.
    freq (str): Frequency for the new date range.
        Default is '1h'.
    interp_method (str, optional): Interpolation method.
        Default is None.
    datetime_col (str): Name of the datetime column.
        Default is 'datetime'.
    round_freq (str): Frequency to round the datetime values.
        Default is '5min'.

    Returns:
    pd.DataFrame: Processed dataframe with interpolated values.
    """
    if not isinstance(df_source, pd.DataFrame):
        raise ValueError("df_source must be a pandas DataFrame")
    if not isinstance(start_date, (str, pd.Timestamp)):
        raise ValueError("start_date must be a string or pandas Timestamp")
    if not isinstance(end_date, (str, pd.Timestamp)):
        raise ValueError("end_date must be a string or pandas Timestamp")
    if not isinstance(freq, str):
        raise ValueError("freq must be a string")
    if interp_method and interp_method not in [
        'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear',
        'quadratic', 'cubic', 'barycentric', 'krogh', 'polynomial', 'spline',
        'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'
    ]:
        intrp_msg = 'interp_method must be a valid interp. method or None'
        raise ValueError(intrp_msg)
    if not isinstance(datetime_col, str):
        raise ValueError("datetime_col must be a string")
    if not isinstance(round_freq, str):
        raise ValueError("round_freq must be a string")

    df_original = df_source.copy()

    # Remove timezone from the original dataframe
    df_original[datetime_col] = pd.to_datetime(
        df_original[datetime_col]).dt.tz_localize(None)

    # Create a DataFrame with the new date range
    df_processed = pd.DataFrame({datetime_col: pd.date_range(start=start_date,
                                                            end=end_date,
                                                            freq=freq)})

    # Round datetime values to match existing values in the original dataframe
    df_original[datetime_col] = df_original[datetime_col].dt.round(round_freq)

    # Merge original and processed dataframes to fill gaps in datetime
    merged_df = pd.merge(df_processed, df_original,
                        how='left', on=datetime_col)

    # Interpolate missing values if interpolation method is provided
    if interp_method:
        try:
            interpolated_df = merged_df.interpolate(method=interp_method,
                                                    limit_direction='both')
        except ValueError as e:
            raise ValueError(f"Interpolation failed: {e}")
    else:
        interpolated_df = merged_df

    final_df = interpolated_df.drop_duplicates(
        subset=datetime_col, keep='first').reset_index(drop=True)

    # Check for NaN values in columns other than 'datetime'
    check_isna = final_df.drop(columns=[datetime_col]).isnull().values.any()
    if check_isna and interp_method:
        raise ValueError("The df contains NaN values in cols.")

    final_df['datetime'] = pd.to_datetime(
        final_df['datetime'].values.astype('datetime64[ms]'), utc=True)

    return final_df