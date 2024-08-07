from collections import defaultdict
import pathlib
import polars as pl
import torch
from torch.utils.data import Dataset as TorchDataset

MultivariateWindow = dict[str, torch.Tensor]
MultivariateFeatures = MultivariateWindow
MultivariateTimestamps = MultivariateWindow

AsyncMTS = tuple[
    MultivariateTimestamps,
    MultivariateFeatures,
]

AsyncMTSWindowPair = tuple[
    tuple[
        AsyncMTS,
        MultivariateTimestamps,
    ],
    MultivariateFeatures,
]

WindowSample = list[torch.Tensor]

TimestampsSample = WindowSample
FeaturesSample = WindowSample

MultivariateTimestampsSample = dict[str, TimestampsSample]
MultivariateFeaturesSample = dict[str, FeaturesSample]

AsyncMTSSample = tuple[
    tuple[
        tuple[
            MultivariateTimestampsSample,
            MultivariateFeaturesSample,
        ],
        MultivariateTimestampsSample,
    ],
    MultivariateFeaturesSample,
]


def merge_dicts_with_lists(dicts: list[dict]):
    merged_dict = defaultdict(list)

    for d in dicts:
        for key, value in d.items():
            merged_dict[key].append(value)

    return dict(merged_dict)


class SantosDataset:
    def __init__(
        self,
        data_path: pathlib.Path,
    ):
        if not data_path.exists():
            raise FileNotFoundError(f"Path {data_path} does not exist.")

        if not data_path.is_dir():
            raise NotADirectoryError(f"Path {data_path} is not a directory.")

        self.data_path = data_path

        self.original_data = {
            f.stem: (df := pl.read_parquet(f))
            for f in self.data_path.glob("*.parquet")
            if f.is_file()
        }

        self.feature_names = {
            ts_name: [col for col in df.columns if col != "datetime"]
            for ts_name, df in self.original_data.items()
        }

        base_date = min(
            [df["datetime"].min() for df in self.original_data.values()]  # type: ignore
        )

        self.original_data = {
            ts_name: df.with_columns(
                [
                    (pl.col("datetime") - base_date)
                    .dt.total_minutes()
                    .cast(pl.Float32)
                    .alias("rel_datetime")
                ]
            ).select(
                ["datetime", "rel_datetime"]
                + [col for col in df.columns if col not in ["rel_datetime", "datetime"]]
            )
            for ts_name, df in self.original_data.items()
        }

        self.min_timestamp: float = min(
            df["rel_datetime"].min() for df in self.original_data.values()  # type: ignore
        )
        self.max_timestamp: float = max(
            df["rel_datetime"].max() for df in self.original_data.values()  # type: ignore
        )

        self.n_features = {
            ts_name: len(df.columns) - 2 for ts_name, df in self.original_data.items()
        }


class SantosTestDataset(SantosDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        context_masks_path: pathlib.Path,
        target_masks_path: pathlib.Path,
        max_context_size: dict[str, float] | None = None,
    ):

        super().__init__(data_path=data_path)

        if not context_masks_path.exists() or not target_masks_path.exists():
            raise FileNotFoundError(f"Path {data_path} does not exist.")

        if not context_masks_path.is_dir() or not target_masks_path.is_dir():
            raise NotADirectoryError(f"Path {data_path} is not a directory.")

        self.context_masks_path = context_masks_path
        self.target_masks_path = target_masks_path

        context_sample_size = None

        self.max_context_size = max_context_size

        self.original_context_masks = {
            f.stem.replace("_context", ""): pl.read_parquet(f)
            for f in self.context_masks_path.glob("*.parquet")
            if f.is_file()
        }
        mask_size_set = {mask.shape[1] for mask in self.original_context_masks.values()}

        if len(mask_size_set) > 1:
            raise ValueError("All context masks must have the same number of columns.")

        context_sample_size = mask_size_set.pop()

        self.original_target_masks = {
            f.stem.replace("_target", ""): pl.read_parquet(f)
            for f in self.target_masks_path.glob("*.parquet")
            if f.is_file()
        }
        mask_size_set = {mask.shape[1] for mask in self.original_target_masks.values()}

        if len(mask_size_set) > 1:
            raise ValueError("All target masks must have the same number of columns.")

        if context_sample_size != mask_size_set.pop():
            raise ValueError(
                "All target masks must have the same number of columns as context masks."
            )

        self.n_window_pairs = context_sample_size

    def __len__(self):
        return self.n_window_pairs


class SantosTestDatasetTorch(SantosTestDataset, TorchDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        context_masks_path: pathlib.Path,
        target_masks_path: pathlib.Path,
        max_context_size: dict[str, float] | None = None,
    ):
        super().__init__(
            data_path=data_path,
            context_masks_path=context_masks_path,
            target_masks_path=target_masks_path,
            max_context_size=max_context_size,
        )

        self.data = {
            ts_name: torch.tensor(
                df.drop("datetime").to_numpy(),
                dtype=torch.float32,
            )
            for ts_name, df in self.original_data.items()
        }

        self.context_masks = {
            ts_name: torch.tensor(mask.to_numpy(), dtype=torch.bool)
            for ts_name, mask in self.original_context_masks.items()
        }

        self.target_masks = {
            ts_name: torch.tensor(mask.to_numpy(), dtype=torch.bool)
            for ts_name, mask in self.original_target_masks.items()
        }

    def get_context_data(
        self, idx
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        context_data = {
            ts_name: self.data[ts_name][mask[:, idx]]
            for ts_name, mask in self.context_masks.items()
        }

        if self.max_context_size is not None:
            for ts_name, c in context_data.items():
                if c.shape[0] == 0:
                    continue
                last_measurement = c[-1, 0]
                lbound = torch.searchsorted(
                    c[:, 0].contiguous(),
                    last_measurement - self.max_context_size[ts_name],
                )
                context_data[ts_name] = c[lbound:]

        context_timestamps = {ts_name: c[:, 0] for ts_name, c in context_data.items()}
        context_features = {ts_name: c[:, 1:] for ts_name, c in context_data.items()}

        return context_timestamps, context_features

    def get_target_data(self, idx):
        target_data = {
            ts_name: self.data[ts_name][mask[:, idx]]
            for ts_name, mask in self.target_masks.items()
        }

        target_timestamps = {ts_name: c[:, 0] for ts_name, c in target_data.items()}
        target_features = {ts_name: c[:, 1:] for ts_name, c in target_data.items()}

        return target_timestamps, target_features

    def __getitem__(self, idx) -> AsyncMTSWindowPair:

        context_timestamps, context_features = self.get_context_data(idx)

        target_timestamps, target_features = self.get_target_data(idx)

        x = ((context_timestamps, context_features), target_timestamps)
        y = target_features

        return x, y

    @classmethod
    def collate_fn(
        cls,
        elements: list[AsyncMTSWindowPair],
    ) -> AsyncMTSSample:

        x, y_features = zip(*elements)
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

        return ((x_timestamps_, x_features_), y_timestamps_), y_features_


class SantosTestDatasetNumpy(SantosTestDataset, TorchDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        context_masks_path: pathlib.Path,
        target_masks_path: pathlib.Path,
    ):
        super().__init__(
            data_path=data_path,
            context_masks_path=context_masks_path,
            target_masks_path=target_masks_path,
        )

        self.data = {
            ts_name: df.drop("datetime").to_numpy()
            for ts_name, df in self.original_data.items()
        }

        self.context_masks = {
            ts_name: mask.to_numpy()
            for ts_name, mask in self.original_context_masks.items()
        }

        self.target_masks = {
            ts_name: mask.to_numpy()
            for ts_name, mask in self.original_target_masks.items()
        }

    def __getitem__(self, idx):
        context_data = {
            ts_name: self.data[ts_name][mask[:, idx]]
            for ts_name, mask in self.context_masks.items()
        }

        context_timestamps = {ts_name: c[:, 0] for ts_name, c in context_data.items()}
        context_features = {ts_name: c[:, 1:] for ts_name, c in context_data.items()}

        target_data = {
            ts_name: self.data[ts_name][mask[:, idx]]
            for ts_name, mask in self.target_masks.items()
        }

        target_timestamps = {ts_name: c[:, 0] for ts_name, c in target_data.items()}
        target_features = {ts_name: c[:, 1:] for ts_name, c in target_data.items()}

        x = ((context_timestamps, context_features), target_timestamps)
        y = target_features

        return x, y

    def __len__(self):
        return self.n_window_pairs


if __name__ == "__main__":
    data_path = pathlib.Path("data/02_processed/test")
    context_masks_path = pathlib.Path("data/02_processed/test/context_masks")
    target_masks_path = pathlib.Path("data/02_processed/test/target_masks")

    dataset = SantosTestDatasetTorch(
        data_path=data_path,
        context_masks_path=context_masks_path,
        target_masks_path=target_masks_path,
    )

    ((x_timestamps, x_features), y_timestamps), y_features = dataset[0]
    print(len(dataset))
    print(dataset.n_window_pairs)
    print(dataset.min_timestamp)
    print(dataset.max_timestamp)
    print(x_timestamps)
    print(x_features)
    print(y_timestamps)

    dataset = SantosTestDatasetNumpy(
        data_path=data_path,
        context_masks_path=context_masks_path,
        target_masks_path=target_masks_path,
    )

    ((x_timestamps, x_features), y_timestamps), y_features = dataset[0]

    print(len(dataset))
    print(dataset.n_window_pairs)
    print(dataset.min_timestamp)
    print(dataset.max_timestamp)
    print(x_timestamps)
    print(x_features)
    print(y_timestamps)
