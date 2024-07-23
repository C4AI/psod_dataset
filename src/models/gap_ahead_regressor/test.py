from collections import defaultdict
import datetime
import pathlib
import torch
from torch.utils.data import DataLoader
from model import GapAheadAMTSRegressor, PositionalEncoding, index_agreement_torch
import polars as pl

import sys

sys.path.append("src")
from processing.loader import (
    SantosTestDatasetTorch,
)

from train import (
    NUM_LAYERS_RNN,
    NUM_LAYERS_GNN,
    TIME_ENCODING_SIZE,
    POS_ENC_DROPOUT,
    HIDDEN_UNITS,
)


def init_lazy_modules(
    test_dl: DataLoader, model: GapAheadAMTSRegressor, model_path: str = ""
):
    ((x_timestamps, x_features), y_timestamps), y_features = next(iter(test_dl))

    t_inferences = torch.stack(
        [
            torch.stack(
                [
                    (
                        timestamp_tensor[0]
                        if timestamp_tensor.size(0) > 0
                        else torch.tensor(torch.inf, device=timestamp_tensor.device)
                    )
                    for i, timestamp_tensor in enumerate(y)
                ]
            )
            for ts_name, y in y_timestamps.items()
        ],
        dim=-1,
    ).min(dim=-1)[0]

    model(
        context_timestamps=x_timestamps,
        context_features=x_features,
        target_timestamps=y_timestamps,
        t_inferences=t_inferences,
    )


def main():

    batch_size = 64
    test_data_path = "data/02_processed/test"
    model_name = pathlib.Path(__file__).parent.name
    model_version = "20240714042433"
    model_epoch = 20

    model_path = (
        f"data/04_trained_models/{model_name}/{model_version}/epoch_{model_epoch}.pt"
    )

    out_path = f"data/05_inference_results/{model_name}"

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    test_dataset = SantosTestDatasetTorch(
        data_path=test_data_path,
        context_masks_path=test_data_path / "context_masks",
        target_masks_path=test_data_path / "target_masks",
        max_context_size=max_context_window_lengths,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SantosTestDatasetTorch.collate_fn,
    )

    model = GapAheadAMTSRegressor(
        input_sizes=test_dataset.n_features,
        num_layers_rnns=NUM_LAYERS_RNN,
        hidden_units=HIDDEN_UNITS,
        time_encoder=PositionalEncoding(
            time_encoding_size=TIME_ENCODING_SIZE, dropout=POS_ENC_DROPOUT
        ),
        num_layers_gnn=NUM_LAYERS_GNN,
    )
    # init_lazy_modules(test_dl, model, model_path)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():

        model.to(device)
        elem_counts = {ts_name: 1 for ts_name in test_dataset.feature_names.keys()}

        for (
            ((x_timestamps, x_features), y_timestamps),
            y_features,
        ) in test_dl:

            x_timestamps = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_timestamps.items()
            }
            x_features = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_features.items()
            }
            y_timestamps = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_timestamps.items()
            }

            y_features = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_features.items()
            }

            t_inferences = torch.stack(
                [
                    torch.stack(
                        [
                            (
                                timestamp_tensor[0]
                                if timestamp_tensor.size(0) > 0
                                else torch.tensor(
                                    torch.inf, device=timestamp_tensor.device
                                )
                            )
                            for i, timestamp_tensor in enumerate(y)
                        ]
                    )
                    for ts_name, y in y_timestamps.items()
                ],
                dim=-1,
            ).min(dim=-1)[0]

            forecast = model(
                context_timestamps=x_timestamps,
                context_features=x_features,
                target_timestamps=y_timestamps,
                t_inferences=t_inferences,
            )

            for ts_name in y_features:
                for elem in forecast[ts_name]:
                    df = pl.DataFrame(
                        elem.detach().cpu().numpy(),
                        schema={
                            feature_name: pl.Float32
                            for feature_name in test_dataset.feature_names[ts_name]
                        },
                    )
                    full_out_path = (
                        out_path
                        / ts_name
                        / f"{str(elem_counts[ts_name]).rjust(3,"0")}.parquet"
                    )
                    full_out_path.parent.mkdir(parents=True, exist_ok=True)
                    df.write_parquet(full_out_path)

                    elem_counts[ts_name] += 1


if __name__ == "__main__":
    main()
