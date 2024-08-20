from collections import defaultdict
import pathlib
import torch
from torch.utils.data import Dataset as TorchDataset

import sys

sys.path.append("src")

from processing.loader import (
    SantosDataset,
    MultivariateFeaturesSample,
    MultivariateTimestampsSample,
    AsyncMTSSample,
    AsyncMTSWindowPair,
    merge_dicts_with_lists,
)


class SantosTrainDataset(SantosDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        context_window_lengths: dict[str, float],
        target_window_lengths: dict[str, float],
        look_ahead_lengths: dict[str, float],
        samples_per_epoch: int,
    ):

        super().__init__(
            data_path=data_path,
            ts_to_load=set(
                list(context_window_lengths.keys()) + list(target_window_lengths.keys())
            ),
        )

        if len(context_window_lengths) == 0:
            raise ValueError("Context window lengths must be provided.")

        if len(target_window_lengths) == 0:
            raise ValueError("Target window lengths must be provided.")

        self._context_window_lengths = defaultdict(float, context_window_lengths)
        self._target_window_lengths = defaultdict(float, target_window_lengths)
        self._look_ahead_lengths = defaultdict(float, look_ahead_lengths)

        self.all_ts_names = set(
            list(self.context_window_lengths.keys())
            + list(self.target_window_lengths.keys())
            + list(self.look_ahead_lengths.keys())
        )

        self.max_context_size = max(self.context_window_lengths.values())
        self.max_target_size = max(self.target_window_lengths.values())

        self.n_window_pairs = samples_per_epoch

    @property
    def context_window_lengths(self):
        return self._context_window_lengths

    @context_window_lengths.setter
    def context_window_lengths(self, value: dict[str, float]):
        self._context_window_lengths = value

    @property
    def target_window_lengths(self):
        return self._target_window_lengths

    @target_window_lengths.setter
    def target_window_lengths(self, value: dict[str, float]):
        self._target_window_lengths = value

    @property
    def look_ahead_lengths(self):
        return self._look_ahead_lengths

    def __len__(self):
        return self.n_window_pairs


class SantosTrainDatasetTorch(SantosTrainDataset, TorchDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        context_window_lengths: dict[str, float],
        target_window_lengths: dict[str, float],
        look_ahead_lengths: dict[str, float],
        sample_size_per_epoch: int,
    ):
        super().__init__(
            data_path=data_path,
            context_window_lengths=context_window_lengths,
            target_window_lengths=target_window_lengths,
            look_ahead_lengths=look_ahead_lengths,
            samples_per_epoch=sample_size_per_epoch,
        )

        self.data = {
            ts_name: torch.tensor(
                df.drop("datetime").to_numpy(),
                dtype=torch.float32,
            )
            for ts_name, df in self.original_data.items()
        }

    def __getitem__(self, idx) -> tuple[AsyncMTSWindowPair, torch.Tensor]:

        t_inf = torch.empty(1).uniform_(
            self.min_timestamp + self.max_context_size,
            self.max_timestamp - self.max_target_size,
        )

        bounds = {
            ts_name: (
                (
                    t_inf
                    + self.look_ahead_lengths[ts_name]
                    - self.context_window_lengths[ts_name]
                ),
                (t_inf + self.look_ahead_lengths[ts_name]),
                (
                    t_inf
                    + self.look_ahead_lengths[ts_name]
                    + self.target_window_lengths.get(ts_name, 0)
                ),
            )
            for ts_name in self.all_ts_names
        }
        indices = {
            ts_name: torch.searchsorted(
                self.data[ts_name][:, 0], torch.cat(bounds[ts_name])
            )
            for ts_name in self.all_ts_names
        }

        context_data = {
            ts_name: self.data[ts_name][indices[ts_name][0] : indices[ts_name][1]]
            for ts_name, context_length in self.context_window_lengths.items()
            if context_length > 0
        }

        context_timestamps = {ts_name: c[:, 0] for ts_name, c in context_data.items()}
        context_features = {ts_name: c[:, 1:] for ts_name, c in context_data.items()}

        target_data = {
            ts_name: self.data[ts_name][indices[ts_name][1] : indices[ts_name][2]]
            for ts_name, tgt_length in self.target_window_lengths.items()
            if tgt_length > 0
        }

        if len(target_data) == 0:
            return self.__getitem__(idx)

        target_timestamps = {ts_name: c[:, 0] for ts_name, c in target_data.items()}
        target_features = {ts_name: c[:, 1:] for ts_name, c in target_data.items()}

        x = ((context_timestamps, context_features), target_timestamps)
        y = target_features

        return (x, y), t_inf

    @classmethod
    def collate_fn(
        cls,
        elements: list[tuple[AsyncMTSWindowPair, torch.Tensor]],
    ) -> tuple[AsyncMTSSample, torch.Tensor]:

        data, t_inferences = zip(*elements)

        x, y_features = zip(*data)
        context_group, y_timestamps = zip(*x)
        x_timestamps, x_features = zip(*context_group)

        x_timestamps_: MultivariateTimestampsSample = merge_dicts_with_lists(
            x_timestamps
        )
        x_features_: MultivariateFeaturesSample = merge_dicts_with_lists(x_features)

        y_timestamps_: MultivariateTimestampsSample = merge_dicts_with_lists(
            y_timestamps
        )
        y_features_: MultivariateFeaturesSample = merge_dicts_with_lists(y_features)
        t_inferences_ = torch.cat(t_inferences)

        return (
            ((x_timestamps_, x_features_), y_timestamps_),
            y_features_,
        ), t_inferences_
