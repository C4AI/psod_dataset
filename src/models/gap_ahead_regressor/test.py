from collections import defaultdict
import numpy as np
import pathlib
import torch
from torch.utils.data import DataLoader
from model import (
    GapAheadAMTSRegressor,
    ModelConfig,
    PositionalEncoding,
    index_agreement_torch,
)
import polars as pl
import yaml
import sys

sys.path.append("src")
from processing.loader import (
    SantosTestDatasetTorch,
)


def main():

    batch_size = 64
    test_data_path = "data/02_processed/test"
    model_name = pathlib.Path(__file__).parent.name
    model_out_name = "multivariate_gap_ahead_regressor"
    model_version = "20240807030811"
    model_epoch = 80
    model_path = f"data/04_trained_models/{model_name}/{model_version}"

    model_path_ = pathlib.Path(model_path)

    missing_percentage = 80
    out_path = f"data/05_inference_results/{model_out_name}"

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "model_version.txt", "w") as f:
        f.write(model_version)
        f.write("\n")
        f.write(f"Epoch: {model_epoch}")

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
        context_masks_path=test_data_path
        / f"missing_ratio_{missing_percentage}"
        / "context_masks",
        target_masks_path=test_data_path
        / f"missing_ratio_{missing_percentage}"
        / "target_masks",
        max_context_size=max_context_window_lengths,
    )

    test_dl = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=SantosTestDatasetTorch.collate_fn,
    )

    with open(model_path_ / "model_config.yml", "r") as f:
        model_config = ModelConfig(**yaml.safe_load(f))

    model = GapAheadAMTSRegressor(
        input_sizes=test_dataset.n_features,
        num_layers_rnns=model_config.num_layers_rnn,
        hidden_units=model_config.hidden_units,
        time_encoder=(
            PositionalEncoding(time_encoding_size=model_config.time_encoding_size)
            if model_config.time_encoding_size > 0
            else None
        ),
        num_layers_gnn=model_config.num_layers_gnn,
        num_heads_gnn=model_config.num_heads_gnn,
    )
    # init_lazy_modules(test_dl, model, model_path)
    model.load_state_dict(torch.load(model_path_ / f"epoch_{model_epoch}.pt"))

    model.eval()

    all_losses: dict[str, list[pl.DataFrame]] = {}
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

            losses_by_ts = {
                inner_ts: torch.stack(
                    [
                        1.0 - index_agreement_torch(f[i], y_features[inner_ts][i])
                        for i in range(len(f))
                        if f[i].size(0) > 0
                    ]
                )
                for inner_ts, f in forecast.items()
            }

            for ts_name, losses in losses_by_ts.items():
                if ts_name not in all_losses:
                    all_losses[ts_name] = []
                all_losses[ts_name].append(losses.cpu().numpy())

            for ts_name in y_features:
                full_out_path = (
                    out_path / str(ts_name) / f"missing_ratio_{missing_percentage}"
                )

                full_out_path.mkdir(parents=True, exist_ok=True)

                for elem in forecast[ts_name]:
                    df = pl.DataFrame(
                        elem.detach().cpu().numpy(),
                        schema={
                            feature_name: pl.Float32
                            for feature_name in test_dataset.feature_names[ts_name]
                        },
                    )

                    df.write_parquet(
                        full_out_path
                        / f"{str(elem_counts[ts_name]).rjust(3,"0")}.parquet"
                    )

                    elem_counts[ts_name] += 1

    for ts_name, dfs in all_losses.items():
        print(f"{ts_name}: {np.concatenate(dfs).mean(axis=0)}")


if __name__ == "__main__":
    main()
