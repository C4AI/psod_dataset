from src.processing.loader import SantosTestDatasetNumpy
import pathlib
from prophet import Prophet
import numpy as np
import pandas as pd 
import polars as pl
import matplotlib.pyplot as plt
import datetime

missing_ratios = [0, 20, 40, 60, 80]
data_path = pathlib.Path("data/02_processed/test")
out_folder = 'data_out'

for m in missing_ratios:
    context_masks_path = pathlib.Path(f"data/02_processed/test/missing_ratio_{m}/context_masks")
    target_masks_path = pathlib.Path(f"data/02_processed/test/missing_ratio_{m}/target_masks")

    dataset = SantosTestDatasetNumpy(
        data_path=data_path,
        context_masks_path=context_masks_path,
        target_masks_path=target_masks_path,
    )

    for t in range(356):
        try:

            (x_timestamps, x_features, y_timestamps), y_features = dataset[t]

            x_t = [datetime.datetime(2020, 1, 1) + datetime.timedelta(minutes=float(t)) for t in x_timestamps['waves_palmas']]
            y_t = [datetime.datetime(2020, 1, 1) + datetime.timedelta(minutes=float(t)) for t in y_timestamps['waves_palmas']]

            forecast = []

            for i in range(3):
                data = {'ds': x_t, 'y': x_features['waves_palmas'][:,i]}
                dataframe = pd.DataFrame(data)

                model = Prophet()
                model.fit(dataframe)

                timestamp_future =  pd.DataFrame({'ds': y_t})
                forecast.append(model.predict(timestamp_future)['yhat'])


            forecasts = pd.concat(forecast, axis = 1)
            forecasts.columns = ['yhat_0', 'yhat_1', 'yhat_2']
            pl.DataFrame(forecasts).write_parquet(f'{out_folder}/waves_palmas_missing_{m}_{t}.parquet')

        except:
            pass

        try:

            (x_timestamps, x_features, y_timestamps), y_features = dataset[t]

            x_t = [datetime.datetime(2020, 1, 1) + datetime.timedelta(minutes=float(t)) for t in x_timestamps['current_praticagem']]
            y_t = [datetime.datetime(2020, 1, 1) + datetime.timedelta(minutes=float(t)) for t in y_timestamps['current_praticagem']]

            data = {'ds': x_t, 'y': x_features['current_praticagem'][:,0]}
            dataframe = pd.DataFrame(data)

            model = Prophet()
            model.fit(dataframe)

            timestamp_future =  pd.DataFrame({'ds': y_t})
            forecast = pd.DataFrame(model.predict(timestamp_future)['yhat'])

            pl.DataFrame(forecast).write_parquet(f'{out_folder}/current_praticagem_missing_{m}_{t}.parquet')

        except:
            pass