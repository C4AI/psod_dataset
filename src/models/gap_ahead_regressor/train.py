from collections import defaultdict
import datetime
import pathlib
import torch
import uniplot
from torch.utils.data import DataLoader
from model import GapAheadAMTSRegressor, PositionalEncoding, index_agreement_torch

import sys

sys.path.append("src")
from processing.loader import (
    SantosTestDatasetTorch,
)

from dataset import SantosTrainDatasetTorch

NUM_LAYERS_RNN = 1
NUM_LAYERS_GNN = 2
TIME_ENCODING_SIZE = 50
POS_ENC_DROPOUT = 0.0
HIDDEN_UNITS = 100


def main():

    epochs = 100
    batch_size = 64
    context_increase_step = 60 * 12.0
    forecast_increase_step = 60 * 4.0
    sample_size_per_epoch = 50000
    train_data_path = "data/02_processed/train"
    test_data_path = "data/02_processed/test"
    lr = 0.001
    model_version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    start_model_path = ""
    model_name = pathlib.Path(__file__).parent.name

    # sample_size_per_epoch = 50
    # start_model_path = "data/04_model_output/20240714032850/epoch_11.pt"

    out_path = f"data/04_trained_models/{model_name}/{model_version}"

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not ((train_data_path := pathlib.Path(train_data_path)).exists()):
        raise FileNotFoundError(f"Train data path {train_data_path} does not exist")

    if not ((test_data_path := pathlib.Path(test_data_path)).exists()):
        raise FileNotFoundError(f"Test data path {test_data_path} does not exist")

    context_window_lengths = defaultdict(
        float,
        {
            "astronomical_tide": 60 * 48.0,
            "current_praticagem": 60 * 48.0,
            "sofs_praticagem": 60 * 48.0,
            "ssh_praticagem": 60 * 48.0,
            "waves_palmas": 60 * 48.0,
            "wind_praticagem": 60 * 48.0,
        },
    )

    target_window_lengths = defaultdict(
        float,
        {
            "current_praticagem": 60 * 4.0,
            "waves_palmas": 60 * 4.0,
        },
    )

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

    max_forecast_window_lengths = defaultdict(
        float,
        {
            "current_praticagem": 60 * 24.0 * 2,
            "waves_palmas": 60 * 24.0 * 2,
        },
    )

    look_ahead_lengths = defaultdict(
        float,
        {
            "astronomical_tide": 60 * 48.0,
            "sofs_praticagem": 60 * 48.0,
        },
    )

    train_dataset = SantosTrainDatasetTorch(
        data_path=train_data_path,
        context_window_lengths=context_window_lengths,
        target_window_lengths=target_window_lengths,
        look_ahead_lengths=look_ahead_lengths,
        sample_size_per_epoch=sample_size_per_epoch,
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=SantosTrainDatasetTorch.collate_fn,
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
        input_sizes=train_dataset.n_features,
        num_layers_rnns=NUM_LAYERS_RNN,
        hidden_units=HIDDEN_UNITS,
        time_encoder=PositionalEncoding(
            time_encoding_size=TIME_ENCODING_SIZE, dropout=POS_ENC_DROPOUT
        ),
        num_layers_gnn=NUM_LAYERS_GNN,
    )

    if start_model_path != "":
        model.load_state_dict(torch.load(start_model_path))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        batch_count = 0
        for (
            ((x_timestamps, x_features), y_timestamps),
            y_features,
        ), t_inferences in train_dl:

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
            t_inferences = t_inferences.to(device)
            model.train()

            model.zero_grad()
            forecast = model(
                context_timestamps=x_timestamps,
                context_features=x_features,
                target_timestamps=y_timestamps,
                t_inferences=t_inferences,
            )
            losses_by_ts = {
                ts_name: torch.stack(
                    [
                        1.0 - index_agreement_torch(f[i], y_features[ts_name][i])
                        for i in range(len(f))
                        if f[i].size(0) > 0
                    ]
                )
                for ts_name, f in forecast.items()
            }

            loss = torch.cat(
                [mean.mean(dim=0) for mean in losses_by_ts.values()],
                dim=-1,
            ).mean()
            loss.backward()
            optimizer.step()
            current_loss = losses_by_ts["current_praticagem"].detach().cpu()
            waves_loss = losses_by_ts["waves_palmas"].detach().cpu()
            print(
                (
                    f"Iteration: {batch_count+1}/{len(train_dl)}, "
                    f"Forecast Sizes: {(target_window_lengths["current_praticagem"],target_window_lengths["waves_palmas"])}, "
                    f"Current: {[f"({m:.2f},{s:.2f})" for (m,s) in zip(current_loss.mean(dim=0).tolist(),current_loss.std(dim=0).tolist())]}, "
                    f"Waves: {[f"({m:.2f},{s:.2f})" for (m,s) in zip(waves_loss.mean(dim=0).tolist(),waves_loss.std(dim=0).tolist())]}"
                )
            )
            batch_count += 1

        test_results: dict[str, list[float]] = defaultdict(list)
        model.zero_grad()

        batch_count = 0

        x_timestamps_ = x_timestamps
        x_features_ = x_features
        y_timestamps_ = y_timestamps
        y_features_ = y_features

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
            model.eval()

            with torch.no_grad():
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
                    ts_name: torch.stack(
                        [
                            1.0 - index_agreement_torch(f[i], y_features[ts_name][i])
                            for i in range(len(f))
                            if f[i].size(0) > 0
                        ]
                    )
                    for ts_name, f in forecast.items()
                }
                current_loss = (
                    losses_by_ts["current_praticagem"].detach().cpu().mean(dim=0)
                )
                waves_loss = losses_by_ts["waves_palmas"].detach().cpu().mean(dim=0)

            test_results["current_praticagem"].append(current_loss)  # type: ignore
            test_results["waves_palmas"].append(waves_loss)  # type: ignore

            if batch_count == len(test_dl) // 2:
                for ts_name, y_ts_list in y_timestamps.items():
                    if ts_name == "current_praticagem":
                        uniplot.plot(
                            xs=[
                                y_ts_list[0].cpu().numpy(),
                                y_ts_list[0].cpu().numpy(),
                            ],
                            ys=[
                                y_features[ts_name][0][:, 0].squeeze().cpu().numpy(),
                                forecast[ts_name][0][:, 0]
                                .squeeze()
                                .cpu()
                                .detach()
                                .numpy(),
                            ],
                            color=True,
                            legend_labels=["Target", "Forecast"],
                            title=ts_name,
                        )
                    elif ts_name == "waves_palmas":

                        feats = ["hs", "tp", "ws"]
                        for i, f in enumerate(feats):
                            uniplot.plot(
                                xs=[
                                    y_ts_list[0].cpu().numpy(),
                                    y_ts_list[0].cpu().numpy(),
                                ],
                                ys=[
                                    y_features[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .numpy(),
                                    forecast[ts_name][0][:, i]
                                    .squeeze()
                                    .cpu()
                                    .detach()
                                    .numpy(),
                                ],
                                color=True,
                                legend_labels=["Target", "Forecast"],
                                title=f"{ts_name} {f}",
                            )

            batch_count += 1

        test_results_ = {ts_name: torch.stack(losses).mean(dim=0) for ts_name, losses in test_results.items()}  # type: ignore
        print(f"\n\nTEST RESULTS EPOCH {epoch}: {test_results_}\n\n")
        context_window_lengths = {
            ts_name: min(
                context_window_lengths[ts_name] + context_increase_step,
                max_context_window_lengths[ts_name],
            )
            for ts_name in context_window_lengths
        }
        train_dataset.context_window_lengths = context_window_lengths

        target_window_lengths = {
            ts_name: min(
                target_window_lengths[ts_name] + forecast_increase_step,
                max_forecast_window_lengths[ts_name],
            )
            for ts_name in target_window_lengths
        }
        train_dataset.target_window_lengths = target_window_lengths
        torch.save(model.state_dict(), out_path / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
