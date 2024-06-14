import pathlib
import polars as pl
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


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
            .with_columns(
                [
                    (pl.col("datetime") - pl.col("datetime").min())
                    .dt.total_minutes()
                    .cast(pl.Float32)
                    .alias("rel_datetime")
                ]
            )
            .select(
                ["datetime", "rel_datetime"]
                + [col for col in df.columns if col not in ["rel_datetime", "datetime"]]
            )
            for f in self.data_path.glob("*.parquet")
            if f.is_file()
        }

        self.min_timestamp: float = min(
            df["rel_datetime"].min() for df in self.original_data.values()
        )
        self.max_timestamp: float = max(
            df["rel_datetime"].max() for df in self.original_data.values()
        )


class SantosTestDataset(SantosDataset):
    def __init__(
        self,
        data_path: pathlib.Path,
        context_masks_path: pathlib.Path,
        target_masks_path: pathlib.Path,
    ):

        super().__init__(data_path=data_path)

        if not context_masks_path.exists() or not target_masks_path.exists():
            raise FileNotFoundError(f"Path {data_path} does not exist.")

        if not context_masks_path.is_dir() or not target_masks_path.is_dir():
            raise NotADirectoryError(f"Path {data_path} is not a directory.")

        self.context_masks_path = context_masks_path
        self.target_masks_path = target_masks_path

        context_sample_size = None

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


class SantosTestDatasetTorch(SantosTestDataset, TorchDataset):
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

        x = (context_timestamps, context_features, target_timestamps)
        y = target_features

        return x, y

    def __len__(self):
        return self.n_window_pairs


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

        x = (context_timestamps, context_features, target_timestamps)
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

    (x_timestamps, x_features, y_timestamps), y_features = dataset[0]
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

    (x_timestamps, x_features, y_timestamps), y_features = dataset[0]

    print(len(dataset))
    print(dataset.n_window_pairs)
    print(dataset.min_timestamp)
    print(dataset.max_timestamp)
    print(x_timestamps)
    print(x_features)
    print(y_timestamps)
