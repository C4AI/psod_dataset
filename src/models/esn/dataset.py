import torch
import pandas as pd
import numpy as np

def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
        index of agreement

        Willmott (1981, 1982)
        input:
        s: simulated
        o: observed
    output:
        ia: index of agreement
    """
    ia = 1 - (torch.sum((o - s) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(s - torch.mean(o, dim=0)) +
             torch.abs(o - torch.mean(o, dim=0)))
            ** 2,
            dim=0,
        )
        + 1e-8  # avoid division by 0
    )

    return ia.mean()

def rmse_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
        index of agreement

        input:
        s: simulated
        o: observed
    output:
        rmse: rmse
    """
    rmse = torch.sqrt(torch.mean(torch.sum((o - s) ** 2, dim=0)))

    return rmse


class zscore:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def apply(self, x):
        if self.mean is None:
            self.mean = x.mean(axis=0)
        if self.std is None:
            self.std = x.std(axis=0)

        return (x - self.mean)/self.std

    def reverse(self, x):
        return x*self.std + self.mean

class test_batcher:
    def __init__(self, batches, inp_columns, pred_columns, device=torch.device('cpu'), dtype=torch.float32, warmup_window=pd.Timedelta(7, unit='days')):
        self.columns = dict()
        self.batches = list()
        for idx, batch in enumerate(batches):
            try:
                ((x_timestamps, x_features), y_timestamps), y_features = batch
                context = pd.DataFrame()
                forecast_start = 10e10
                for targ_col in pred_columns:
                    if y_timestamps[targ_col][0] < forecast_start:
                        forecast_start = y_timestamps[targ_col][0]
                context_start = pd.to_datetime(forecast_start, unit= 'm', utc=True) - warmup_window
                for column in inp_columns:
                    df = pd.DataFrame(data={'datetime': pd.to_datetime(x_timestamps[column],unit= 'm', utc=True)})
                    for i in range(x_features[column].shape[1]):
                        dfc = pd.DataFrame(data={column + '_' + str(i): x_features[column][:, i]})
                        df = pd.concat( (df, dfc), axis=1)
                    df.set_index('datetime', inplace=True)
                    context = pd.concat((context, df), axis=0)
                targets = pd.DataFrame()
                for targ_col in pred_columns:
                    df = pd.DataFrame(data={'datetime': pd.to_datetime(y_timestamps[targ_col], unit='m', utc=True)})
                    for i in range(y_features[targ_col].shape[1]):
                        dfc = pd.DataFrame(data={targ_col + '_' + str(i): y_features[targ_col][:, i]})
                        df = pd.concat( (df, dfc), axis=1)
                    df.set_index('datetime', inplace=True)
                    targets = pd.concat((targets, df), axis=0)
                context = context.groupby(level=0).mean(numeric_only=True)
                context = context.loc[context.index >= context_start]
                context.sort_index(inplace=True)
                self.batches.append((context, targets))
            except Exception:
                print('Skipped batch ', idx)
                self.batches.append((None, None))
        self.device = device
        self.dtype = dtype

    def create_batches(self):
        return [context_target_batch(b, self.device, self.dtype, None) for b in self.batches]


class context_train_dataset:
    def __init__(self, **kwargs):
        self.context = pd.DataFrame()
        columns_description = dict()
        self.kwargs = kwargs
        self.predicted_columns = list()
        self.is_forecast = list()
        self.input_datasets = kwargs['input_datasets']
        self.target_datasets = kwargs['target_datasets']
        self.targets = pd.DataFrame()
        for ts in kwargs['timeseries']:
            df = pd.read_parquet(ts['path'])
            df.set_index('datetime', inplace=True)
            for column in df:
                if 'columns' in ts.keys():
                    if column not in ts['columns']:
                        df.drop(columns=column, inplace=True)

                df.rename(columns={column: column + '_' + ts['description']}, inplace=True)
            for transformation in ts['transformations']:
                df = transformation.apply(df)

            if ts['is_predicted'] is True:
                self.targets = pd.concat((self.targets, df), axis=0)
            self.context = pd.concat((self.context, df), axis=0)
            columns_description[ts['description']] = []
            for column in df:
                columns_description[ts['description']].append(self.context.columns.get_loc(column))
                if ts['is_predicted'] is True:
                    self.predicted_columns.append(self.context.columns.get_loc(column))
                if 'forecast' in ts.keys():
                    self.is_forecast.append(ts['forecast'])
                else:
                    self.is_forecast.append(False)

        self.columns_description = columns_description

        self.targets = self.targets.groupby(level=0).mean(numeric_only=True)
        self.targets.sort_index(inplace=True)
        self.targets = self.targets.shift(-1)

        self.context = self.context.groupby(level=0).mean(numeric_only=True)
        self.context.sort_index(inplace=True)

        self.warmup_period = kwargs['warmup']
        self.forecast_horizon = kwargs['forecast_horizon']

        self.device = kwargs['device']
        self.dtype = kwargs['dtype']

    def apply_transformations(self):
        if self.transformations is not None:
            for transformation in self.transformations:
                self.context = transformation(self.context)

    def get_dimensions(self):
        out_map = [np.nan for i in range(self.context.shape[1])]
        for en, i in enumerate(self.predicted_columns):
            out_map[i] = en
        return self.context.shape[1], self.targets.shape[1], out_map, self.columns_description

    def get_columns_names(self):
        return self.input_datasets, self.target_datasets

    def split_batched_train_val(self, val_period_percentage, train_stride, val_stride):
        warmup = self.warmup_period
        forecast_horizon = self.forecast_horizon
        data_range = self.context.index[-1] - self.context.index[0]
        batch_start = val_start = self.context.index[-1] - data_range*val_period_percentage
        train_context = self.context[self.context.index < val_start]
        t_batches = list()
        v_batches = list()
        # Validation batches
        batch_end = batch_start + warmup + forecast_horizon
        batch_f_start = batch_start + warmup
        skipped_batches=0
        validation_batches=0
        while batch_end < self.context.index[-1]:
            context = self.context[batch_start <= self.context.index]
            context = context[context.index < batch_end]
            context.loc[context.index < batch_f_start, self.is_forecast] = np.nan
            context.loc[context.index >= batch_f_start, np.bitwise_not (self.is_forecast)] = np.nan
            context.dropna(how='all', inplace=True)
            context.loc[context.index < batch_f_start, self.is_forecast] = 0
            try:
                context.iloc[-1] = 0
                target = self.targets[batch_f_start < self.targets.index]
                target = target[target.index <= batch_end]
                if len(target) == 0:
                    raise ValueError('Empty target')
                v_batches.append(context_target_batch((context, target), self.device, self.dtype, self.predicted_columns))
                validation_batches+= 1
            except:
                skipped_batches+= 1
            batch_start = batch_start + val_stride
            batch_end = batch_start + warmup + forecast_horizon
            batch_f_start = batch_start + warmup

        print('Total validation batches: ', validation_batches)
        print('Total skipped batches: ', skipped_batches)

        # Training batches
        batch_start = self.context.index[0]
        batch_end = batch_start + warmup + forecast_horizon
        batch_f_start = batch_start + warmup
        train_batches = 0
        skipped_batches=0
        while batch_end < val_start:
            context = self.context[batch_start <= self.context.index]
            context = context[context.index < batch_end]
            context.loc[context.index < batch_f_start, self.is_forecast] = np.nan
            context.loc[context.index >= batch_f_start, np.bitwise_not (self.is_forecast)] = np.nan
            context.dropna(how='all', inplace=True)
            context.loc[context.index < batch_f_start, self.is_forecast] = 0
            try:
                context.iloc[-1] = 0
                target = self.targets[batch_f_start < self.targets.index]
                target = target[target.index <= batch_end]

                if len(target) == 0:
                    raise ValueError('Empty target')
                t_batches.append(
                    context_target_batch((context, target), self.device, self.dtype, self.predicted_columns, filler=0))
                train_batches+= 1
            except:
                skipped_batches+= 1
            batch_start = batch_start + train_stride
            batch_end = batch_start + warmup + forecast_horizon
            batch_f_start = batch_start + warmup

        print('Total training batches: ', train_batches)
        print('Total training batches: ', train_batches)
        print('Total skipped batches: ', skipped_batches)

        return t_batches, v_batches


class context_target_batch:
    def __init__(self, context_target_tuple, device, dtype,pred_columns, filler=torch.nan):
        self.context, self.target = context_target_tuple
        if self.target is not None:
            self.target_size = self.target.shape[1]
        else:
            self.target_size = 0
        self.pred_columns = pred_columns
        self.device = device
        self.dtype = dtype
        self.filler = filler

    def set_warmup(self, *args):
        pass

    def __iter__(self):
        if self.context is None or self.target is None:
            self.idx_context = 0
            self.idx_target = 0
            self.max_idx_target = 0
            self.max_idx_context = 0
            self.has_next = False
        else:
            self.idx_context = 0
            self.idx_target = 0
            self.last_row = torch.zeros(self.context.shape[1], device=self.device, dtype=self.dtype)
            self.last_output = [None, torch.tensor(self.target_size, device=self.device, dtype=torch.bool), None]
            self.next_context_update = None
            self.future_context_update = None
            self.next_target_update = None
            self.future_target_update = None
            self.max_idx_context = self.context.shape[0]
            self.max_idx_target = self.target.shape[0]
            self.current_time = min((self.context.index[0],self.target.index[0]))
            self.has_next = True
            self.inputs = None
            self.outputs = None
        return self

    def __next__(self):
        if self.idx_target >= self.max_idx_target:
            self.has_next = False

        if self.has_next:
            if self.idx_context < self.max_idx_context - 1:
                self.next_context_update= self.context.index[self.idx_context]
                self.future_context_update = self.context.index[self.idx_context + 1]
            elif self.idx_context >= self.max_idx_context:
                self.next_context_update = pd.to_datetime(3162240000, unit='s', utc=True)
            else:
                self.next_context_update = self.context.index[self.idx_context]
                self.future_context_update = pd.to_datetime(3162240000, unit='s', utc=True)

            if self.idx_target < self.max_idx_target - 1:
                self.next_target_update = self.target.index[self.idx_target]
                self.future_target_update = self.target.index[self.idx_target + 1]
            else:
                self.next_target_update = self.target.index[self.idx_target]
                self.future_target_update = pd.to_datetime(3162240000, unit='s', utc=True)

            # update either the output or the input provided
            if self.next_context_update <= self.next_target_update:
                # Update only the inputs (no inference in that step)
                this_row = torch.tensor(self.context.iloc[self.idx_context], device = self.device, dtype= self.dtype)
                this_row[torch.isnan(this_row)] = self.last_row[torch.isnan(this_row)]
                self.last_row = this_row.clone().detach()
                if self.future_context_update < self.next_target_update:
                    dt = (self.future_context_update - self.next_context_update)/ np.timedelta64(1, 's')
                else:
                    dt = (self.next_target_update - self.next_context_update)/ np.timedelta64(1, 's')
                self.inputs = [this_row, dt, self.context.index[self.idx_context]]
                self.outputs = [None, torch.zeros(self.target_size, device=self.device, dtype=torch.bool), None]
                self.current_time = self.context.index[self.idx_context]
                self.idx_context += 1
            else:
                next_prediction = torch.tensor(self.target.iloc[self.idx_target], device=self.device, dtype=self.dtype)
                self.outputs = [next_prediction, ~torch.isnan(next_prediction), self.target.index[self.idx_target]]
                if self.future_target_update <= self.next_context_update:
                    dt = (self.next_target_update - self.current_time)/ np.timedelta64(1, 's')
                else:
                    dt = (self.next_context_update - self.current_time)/ np.timedelta64(1, 's')
                self.inputs[1] = dt
                self.inputs[0][self.pred_columns] = self.filler
                self.current_time = self.target.index[self.idx_target]
                self.idx_target += 1
            return self.inputs, self.outputs
        else:
            raise StopIteration