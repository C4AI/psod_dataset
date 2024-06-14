# Exploratory Data Analysis (EDA) for Time Series Data using Polars

import pathlib
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import missingno as msno


# Read Parquet file
def read_parquet(file_path: pathlib.Path) -> pl.DataFrame:
    return pl.read_parquet(file_path)


# Plot time series
def plot_time_series(
    df: pl.DataFrame, datetime_col: str, value_cols: list[str], title: str
) -> None:
    for col in value_cols:
        plt.figure(figsize=(12, 6))
        plt.plot(df[datetime_col], df[col], label=col)
        plt.title(f"{title} - {col}")
        plt.xlabel("Datetime")
        plt.ylabel(col)
        plt.legend()
        plt.savefig(f"data/03_eda/{title}_{col}.pdf")


# Resample and plot time series
def resample_and_plot(
    df: pl.DataFrame, datetime_col: str, value_cols: list[str], freq: str = "1h"
) -> None:
    df_resampled: pl.DataFrame = df.group_by_dynamic(datetime_col, every=freq).agg(
        pl.mean(value_cols)
    )
    plot_time_series(
        df_resampled, datetime_col, value_cols, f"Resampled Time Series ({freq})"
    )


# Plot autocorrelation
def plot_autocorrelations(
    df: pl.DataFrame, datetime_col: str, value_col: str, lags: int = 200
) -> None:
    values: np.ndarray = df.select(value_col).to_numpy()
    plot_pacf(values, lags=lags)
    plt.title(f"Partial Autocorrelation of {value_col}")
    plt.savefig(f"data/03_eda/pacf_{value_col}.pdf")

    plot_acf(values, lags=lags)
    plt.title(f"Autocorrelation of {value_col}")
    plt.savefig(f"data/03_eda/acf_{value_col}.pdf")


# Seasonal decomposition
def seasonal_decompose(
    df: pl.DataFrame,
    datetime_col: str,
    value_col: str,
    model: str = "additive",
    freq: int = 365,
) -> None:
    from statsmodels.tsa.seasonal import seasonal_decompose

    values: np.ndarray = df.select(value_col).to_numpy().flatten()
    result = seasonal_decompose(values, model=model, period=freq)
    result.plot()
    plt.show()


def plot_missing_data(dfs: dict[str, pl.DataFrame]) -> None:

    grouped_dfs = {
        file_name: df.sort("datetime")
        .select("datetime")
        .group_by_dynamic(pl.col("datetime"), every="1h")
        .agg(pl.count())
        .rename({"count": file_name})
        for file_name, df in dfs.items()
    }

    joined_df = None
    for file_name, df in grouped_dfs.items():
        if joined_df is None:
            joined_df = df
            continue

        joined_df = joined_df.join(df, on="datetime", how="outer_coalesce")

    fig, ax = plt.subplots(figsize=(12, 16))

    msno.matrix(joined_df.drop("datetime").to_pandas(), ax=ax)
    fig.savefig("data/03_eda/missing_data.png")


# Function to analyze a single time series file
def analyze_time_series(dfs: dict[str, pl.DataFrame]) -> None:
    datetime_col: str = "datetime"

    plot_missing_data(dfs)

    for file_name, df in dfs.items():
        value_cols: list[str] = [col for col in df.columns if col != datetime_col]

        # Plot original time series
        plot_time_series(df, datetime_col, value_cols, file_name)

    # Plot autocorrelation
    for col in value_cols:
        plot_autocorrelations(df, datetime_col, col, lags=60)


# List of .parquet files to analyze
files: list[str] = [
    "data/02_processed/train/astronomical_tide.parquet",
    "data/02_processed/train/sofs_praticagem.parquet",
    "data/02_processed/train/ssh_praticagem.parquet",
    "data/02_processed/train/waves_palmas.parquet",
    "data/02_processed/train/wind_praticagem.parquet",
    "data/02_processed/train/current_praticagem.parquet",
]

# Analyze each file

dfs = {
    (path := pathlib.Path(file_path)).stem: read_parquet(path) for file_path in files
}
analyze_time_series(dfs)
