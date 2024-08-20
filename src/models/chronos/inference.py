from chronos import ChronosPipeline
from collections import defaultdict
import pathlib

import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 2345678
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

sys.path.append("../../../src")
from processing.loader import (
    SantosTestDataset,
)

from dataset import process_dataframe
from model import load_chronos_pipeline

model_name = "chronos"

# Load the ChronosPipeline model from the pretrained
# 'amazon/chronos-t5-large' model
# Load the ChronosPipeline model
chronos_pipeline = load_chronos_pipeline()

test_data_path = "../../../data/02_processed/test"
out_path = f"../../../data/05_inference_results/{model_name}"

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

range_experiments = range(0,100,20)

for exp in range_experiments:

    experiment = f'missing_ratio_{exp}'

    print(f'EXPERIMENT : {experiment}')

    test_dataset = SantosTestDataset(
        data_path=test_data_path,
        context_masks_path=test_data_path / experiment / "context_masks",
        target_masks_path=test_data_path / experiment / "target_masks",
        max_context_size=max_context_window_lengths,
    )

    list_target_features = ['current_praticagem','waves_palmas']

    for feature in list_target_features:

        print(f'Feature    : {feature}')

        out_exp_path = pathlib.Path(f'{out_path}/{feature}/{experiment}')
        out_exp_path.mkdir(parents=True, exist_ok=True)

        list_target_feature_vars = test_dataset.feature_names[feature]

        for col_label in (test_dataset.original_target_masks[feature].columns):

            val_mask = test_dataset.original_target_masks[
                feature][col_label].sum()
            if val_mask == 0:
                pd.DataFrame(columns=['0']).to_parquet(
                    f'{out_exp_path}/{col_label}.parquet')
            else:
                df_test_target = test_dataset.original_data[feature].filter(
                    test_dataset.original_context_masks[
                        feature][col_label]).to_pandas() 

                end_context = df_test_target[
                    'datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') 
                freq_model = '1h'

                train_data_path = (
                    f"../../../data/02_processed/train/{feature}.parquet")
                df_train_target = pd.read_parquet(train_data_path)

                # Process the training dataframe with specified parameters
                df_train_processed_target = process_dataframe(
                    df_train_target,
                    "2021-01-01 00:00:00",
                    "2021-12-31 23:55:00",
                    "1h",
                    "linear",
                    "datetime",
                    "5min")

                df_test_processed_target = process_dataframe(
                    df_test_target,
                    "2022-01-01 00:00:00",
                    end_context,
                    "1h",
                    "linear",
                    "datetime",
                    "5min")

                df_output = pd.DataFrame()

                for index_feature_var, target_feature_var in enumerate(
                    list_target_feature_vars):
                    print(f'Feature var: {col_label} {target_feature_var}')

                    train_signal = df_train_processed_target[
                        target_feature_var].values
                    test_signal = df_test_processed_target[
                        target_feature_var].values

                    composed_signal = np.concatenate(
                        (train_signal, test_signal))

                    batch_context = torch.tensor(composed_signal)

                    # Generate forecast using the Chronos pipeline
                    forecast = chronos_pipeline.predict(
                        batch_context, 48)
                    predictions = np.quantile(
                        forecast.numpy(), 0.5, axis=1)

                    y_hat = np.array(predictions[0])

                    # Convert: end to Timestamp
                    end_timestamp = pd.Timestamp(end_context)

                    reference_df = pd.DataFrame()
                    reference_df['datetime'] = pd.date_range(
                        start=end_timestamp.ceil(freq_model),
                        periods=48, freq=freq_model)
                    reference_df['datetime'] = pd.to_datetime(
                        reference_df['datetime'].values.astype(
                            'datetime64[ms]'), utc=True)
                    reference_df[str(index_feature_var)] =  y_hat

                    # DataFrame w/ irregular freq
                    irregular_df = pd.DataFrame({
                        'datetime': test_dataset.original_data[feature].filter(
                            test_dataset.original_target_masks[
                                feature][col_label]).to_pandas().datetime
                    })
                    irregular_df['datetime'] = pd.to_datetime(
                        irregular_df['datetime'].values.astype(
                            'datetime64[ms]'), utc=True)

                    # Merge using datetime col
                    merged_df = pd.merge(irregular_df,
                        reference_df,
                        on='datetime',
                        how='outer').sort_values('datetime')

                    # Linear interp
                    merged_df[str(index_feature_var)] = merged_df[
                        str(index_feature_var)].interpolate(method='linear')

                    # Filtering interp. data for irregular df
                    result_df = merged_df[merged_df['datetime'].isin(
                        irregular_df['datetime'])].reset_index(drop=True)

                    df_output[index_feature_var] = result_df[
                        str(index_feature_var)] 

                df_output.to_parquet(f'{out_exp_path}/{col_label}.parquet')
