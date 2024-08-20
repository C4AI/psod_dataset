import sys
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import time
import logging
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import DistributionLoss, HuberLoss, MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import AirPassengers, AirPassengersPanel, AirPassengersStatic
import pathlib
from collections import defaultdict
from tqdm import tqdm
from IPython.display import clear_output

sys.path.append("../../../src")
from processing.loader import (
    SantosTestDataset,
)

from dataset import process_dataframe


dict_target_feature_params = {
    "current_praticagem": ["cross_shore_current"],
    "waves_palmas": ["hs", "tp", "ws"],
}

for target_idx_feature in dict_target_feature_params.keys():
    for target_idx_feature_var in dict_target_feature_params[target_idx_feature]:
        print(f"{target_idx_feature}: {target_idx_feature_var}")

# print("\ndict_context_feature_params")

dict_context_feature_params = {
    "ssh_praticagem": ["ssh"],
    "waves_palmas": ["hs", "tp", "ws"],
    #'sofs_praticagem': ['cross_shore_current', 'ssh'],
    #'astronomical_tide': ['astronomical_tide'],
    "wind_praticagem": ["vx", "vy"],
    "current_praticagem": ["cross_shore_current"],
}

for context_idx_feature in dict_context_feature_params.keys():
    for context_idx_feature_var in dict_context_feature_params[context_idx_feature]:
        print(f"{context_idx_feature}: {context_idx_feature_var}")

model_name = "nhits"

test_data_path = "../../../data/02_processed/test"
out_path = f"../../../data/05_inference_results/{model_name}"


freq = "1h"
datetime_col = "datetime"
start_train = "2018-01-01 00:00:00"
end_train = "2021-12-31 23:55:00"

# Define the context and forecast window lengths and shift
context_len = 168
forecast_len = 48
shift = 24
mode = "sliding"

for feature in dict_target_feature_params.keys():
    for var in dict_target_feature_params[feature]:

        target_feature_concat = f"{feature}_{var}"

        # print(target_feature_concat)

        out_path = pathlib.Path(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        if not ((test_data_path := pathlib.Path(test_data_path)).exists()):
            raise FileNotFoundError(f"Test data path {test_data_path} does not exist")

        max_context_window_lengths = defaultdict(
            float,
            {
                "astronomical_tide": 60 * 24.0 * 2,
                "current_praticagem": 60 * 24.0 * 7,
                "sofs_praticagem": 60 * 24.0 * 2,
                "ssh_praticagem": 60 * 24.0 * 7,
                "waves_palmas": 60 * 24.0 * 7,
                "wind_praticagem": 60 * 24.0 * 7,
            },
        )

        experiment_list = [0, 20, 40, 60, 80]

        for exp in experiment_list:
            experiment = f"missing_ratio_{exp}"

            test_dataset = SantosTestDataset(
                data_path=test_data_path,
                context_masks_path=test_data_path / experiment / "context_masks",
                target_masks_path=test_data_path / experiment / "target_masks",
                max_context_size=max_context_window_lengths,
            )

            df_train_composed = pd.DataFrame()

            out_exp_path = pathlib.Path(f"{out_path}/{feature}")
            out_exp_path.mkdir(parents=True, exist_ok=True)

            out_exp_path_var = pathlib.Path(f"{out_exp_path}/{var}/{experiment}")
            out_exp_path_var.mkdir(parents=True, exist_ok=True)

            train_range = pd.date_range(start=start_train, end=end_train, freq=freq)

            if exp == 0:  # TRAIN IF EXP == 0
                df_train_composed[datetime_col] = train_range

                # Iterate over each ocean variable defined in the parameters
                for ocean_variable in dict_context_feature_params.keys():

                    # Retrieve target features and experiment IDs
                    features = dict_context_feature_params[ocean_variable]

                    # Load train and test data for the target feature
                    df_train_target = pd.read_parquet(
                        f"../../../data/02_processed/train/{ocean_variable}.parquet"
                    )

                    df_train_processed_target = process_dataframe(
                        df_train_target,
                        start_train,
                        end_train,
                        freq,
                        "linear",
                        datetime_col,
                        "5min",
                    )

                    # Iterate over each target feature for prediction
                    for tgt_feature in features:
                        # Add training data to improve the size of the inference data
                        train_signal = df_train_processed_target.loc[:, tgt_feature]

                        df_target_col_name = f"{ocean_variable}_{tgt_feature}"

                        df_train_composed[df_target_col_name] = train_signal.values

                # Create the new column list with 'ds' and 'y' replacements
                endg_list_nhits = [
                    "y" if item == target_feature_concat else item
                    for item in df_train_composed.columns
                    if item != "datetime"
                ]

                # Initialize the new DataFrame
                Y_train_df = pd.DataFrame()
                Y_train_df["unique_id"] = [1] * df_train_composed.shape[0]
                Y_train_df["ds"] = df_train_composed["datetime"]
                Y_train_df[endg_list_nhits] = df_train_composed.drop(columns="datetime")

                model = NHITS(h=48, input_size=168, max_steps=500, learning_rate=1e-3)

                fcst = NeuralForecast(models=[model], freq="1h")

                fcst.fit(df=Y_train_df, val_size=48)

            dict_test_context_df = {}
            datetime_col = "datetime"

            list_cols = [f"{i:03}" for i in range(1, 357)]
            #list_cols = ["001"]

            for idx in tqdm(list_cols):
                checksum_ftr = False

                feature_vars = []
                df_test_target_feature_vars = pd.DataFrame()

                for (
                    ftr
                ) in (
                    dict_context_feature_params.keys()
                ): 
                    mask_sum = test_dataset.original_context_masks[ftr][idx].sum()

                    if mask_sum != 0:

                        last_datetime = (
                            test_dataset.original_data[ftr]
                            .filter(test_dataset.original_context_masks[ftr][idx])
                            .to_pandas()
                            .iloc[-1]["datetime"]
                        )

                        # Remove timezone information before creating the date range
                        target_daterange_target = (
                            pd.date_range(
                                start=pd.to_datetime(last_datetime)
                                .tz_localize(None)
                                .ceil("h"),
                                periods=48,
                                freq="1h",
                            )
                            .astype("datetime64[ms]")
                            .tz_localize("UTC")
                        )

                        # Remove timezone information before creating the date range
                        target_daterange_context = (
                            pd.date_range(
                                end=pd.to_datetime(last_datetime)
                                .tz_localize(None)
                                .floor("h"),
                                periods=context_len,
                                freq="1h",
                            )
                            .astype("datetime64[ms]")
                            .tz_localize("UTC")
                        )

                        df_test_target_feature_vars["ds"] = target_daterange_context

                        df_pandas = (
                            test_dataset.original_data[ftr]
                            .filter(test_dataset.original_context_masks[ftr][idx])
                            .to_pandas()
                        )

                        filtered_df = df_pandas.loc[
                            (df_pandas["datetime"] >= target_daterange_context[0])
                            & (df_pandas["datetime"] <= target_daterange_context[-1])
                        ]

                        df_test_processed_target = process_dataframe(
                            filtered_df,
                            target_daterange_context[0],
                            target_daterange_context[-1],
                            freq,
                            "linear",
                            datetime_col,
                            "5min",
                        )

                    else:
                        checksum_ftr = True

                    for feature_var in dict_context_feature_params[
                        ftr
                    ]:
                        col_name_concat = f"{ftr}_{feature_var}"
                        feature_vars.append(col_name_concat)
                        df_test_target_feature_vars[col_name_concat] = (
                            0 if checksum_ftr else df_test_processed_target[feature_var]
                        )

                dict_test_context_df[idx] = df_test_target_feature_vars

            dict_Y_hat_df = {}

            for i in tqdm(list_cols):
                clear_output(wait=False)
                Y_test_df = dict_test_context_df[i].rename(
                    columns={target_feature_concat: "y"}
                )
                Y_test_df["unique_id"] = 1

                Y_hat_df = fcst.predict(Y_test_df).reset_index(
                    drop=True
                )  ############# PREDICT #############
                dict_Y_hat_df[i] = Y_hat_df

            index_feature_var = 0
            dict_df_output = {}

            for col_label in tqdm(list_cols):
                df_output = pd.DataFrame()

                reference_df = dict_Y_hat_df[col_label]
                reference_df = reference_df.rename(columns={"ds": "datetime"})

                irregular_df = pd.DataFrame(
                    {
                        "datetime": test_dataset.original_data[feature]
                        .filter(test_dataset.original_target_masks[feature][col_label])
                        .to_pandas()
                        .datetime
                    }
                )
                irregular_df["datetime"] = pd.to_datetime(
                    irregular_df["datetime"].values.astype("datetime64[ms]"), utc=True
                )

                merged_df = pd.merge(
                    irregular_df, reference_df, on="datetime", how="left"
                ).sort_values("datetime")

                merged_df = merged_df.rename(columns={"NHITS": str(index_feature_var)})

                merged_df[str(index_feature_var)] = merged_df[
                    str(index_feature_var)
                ].interpolate(method="linear")

                result_df = merged_df[
                    merged_df["datetime"].isin(irregular_df["datetime"])
                ].reset_index(drop=True)

                df_output[index_feature_var] = result_df[str(index_feature_var)]
                df_output.to_parquet(f"{out_exp_path_var}/{col_label}.parquet")
                dict_df_output[col_label] = df_output

# test plot
ftr = "current_praticagem"
var = "cross_shore_current"
win = "001"
col = 0

for i in [0, 20, 40, 60, 80]:
    plt.plot(
        pd.read_parquet(
            f"../../../data/05_inference_results/nhits/{ftr}/{var}/missing_ratio_{i}/{win}.parquet"
        )[col].values,
        label=f"missing_ratio_{i}",
    )
plt.legend()
plt.show()