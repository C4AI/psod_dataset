from collections import defaultdict
import datetime
import pathlib
import pickle
import torch
import uniplot
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import os

from model import CGNN, PositionalEncoding

import sys

from models.neural_ode.model.misc.log_utils import CachedRunningAverageMeter

sys.path.append("src")
from processing.loader import (
    SantosTestDatasetTorch,
)

from dataset import SantosTrainDatasetTorch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def index_agreement_torch(s: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    """
    index of agreement
    Willmott (1981, 1982)

    Args:
        s: simulated
        o: observed

    Returns:
        ia: index of agreement
    """
    o_bar = torch.mean(o, dim=0)
    ia = 1 - (torch.sum((o - s) ** 2, dim=0)) / (
        torch.sum(
            (torch.abs(s - o_bar) + torch.abs(o - o_bar)) ** 2,
            dim=0,
        )
    )

    return ia


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def main():

    num_layers_rnn = 1
    hidden_units = 100
    epochs = 200
    batch_size = 64
    sample_size_per_epoch = 20000
    time_encoding_size = 50
    pos_enc_dropout = 0.0
    train_data_path = "data/02_processed/train"
    test_data_path = "data/02_processed/test"
    lr = 0.5e-3
    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    N_increments = 50
    normalize = True
    model_path = None
    start_epoch = 0
    max_context_window_length = 60 * 24 * 1

    seed_everything(42)

    out_path = f"data/04_model_output/{now_str}_cgnn"

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    if not ((train_data_path := pathlib.Path(train_data_path)).exists()):
        raise FileNotFoundError(f"Train data path {train_data_path} does not exist")

    if not ((test_data_path := pathlib.Path(test_data_path)).exists()):
        raise FileNotFoundError(f"Test data path {test_data_path} does not exist")

    context_window_lengths = defaultdict(
        float,
        {
            "astronomical_tide": 60 * 12,
            "current_praticagem": 60 * 24,
            "sofs_praticagem": 60 * 12,
            "ssh_praticagem": 60 * 12,
            "waves_palmas": 60 * 24,
            "wind_praticagem": 60 * 12,
        },
    )

    target_window_lengths = defaultdict(
        float,
        {
            "current_praticagem": 60 * 48.0,
            "waves_palmas": 60 * 48.0,
        },
    )

    context_window_lengths_increment = {
        ts_name: (max_context_window_length - length) // N_increments
        for ts_name, length in context_window_lengths.items()
    }

    target_window_lengths_increment = {
        ts_name: length // N_increments
        for ts_name, length in target_window_lengths.items()
    }

    max_context_window_lengths = defaultdict(
        float,
        {
            "astronomical_tide": 60 * 24.0 * 1,
            "current_praticagem": 60 * 24.0 * 1,
            "sofs_praticagem": 60 * 24.0 * 1,
            "ssh_praticagem": 60 * 24.0 * 1,
            "waves_palmas": 60 * 24.0 * 1,
            "wind_praticagem": 60 * 24.0 * 1,
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

    context_time_series = ["current_praticagem", "waves_palmas"]
    target_time_series = ["current_praticagem", "waves_palmas"]
    propagate_time_series = ["current_praticagem", "waves_palmas"]

    train_dataset = SantosTrainDatasetTorch(
        data_path=train_data_path,
        context_window_lengths=context_window_lengths,
        target_window_lengths=target_window_lengths,
        look_ahead_lengths=look_ahead_lengths,
        sample_size_per_epoch=sample_size_per_epoch,
        normalize=normalize,
        scaler="standard",
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

    model = CGNN(
        input_sizes=train_dataset.n_features,
        num_layers_rnns=num_layers_rnn,
        hidden_units=hidden_units,
        time_encoder=PositionalEncoding(
            time_encoding_size=time_encoding_size, dropout=pos_enc_dropout
        ),
        context_time_series=context_time_series,
        target_time_series=target_time_series,
        propagate_time_series=propagate_time_series,
        device=device,
        dtype=dtype,
    )

    if normalize:

        if model_path:
            scaler = list(pathlib.Path(model_path).parent.glob("scaler.pkl"))
            if len(scaler) != 0:
                train_dataset.scalers = pickle.load(open(scaler[0], "rb"))

        with open(out_path / "scaler.pkl", "wb") as f:
            pickle.dump(train_dataset.scalers, f)

    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.to(device).to(dtype)

    print(f"{count_params(model)} parameters")

    train_dataset._target_window_lengths = target_window_lengths_increment.copy()

    loss_meter_current = CachedRunningAverageMeter(0.97)
    loss_meter_waves = CachedRunningAverageMeter(0.97)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):

        if (epoch != 0) and ((epoch + 1) % (epochs // N_increments)) == 0:
            for ts_name in target_window_lengths.keys():
                train_dataset.change_target_length(
                    target_window_lengths_increment[ts_name], ts_name
                )

                # train_dataset._context_window_lengths[
                #    ts_name
                # ] += context_window_lengths_increment[ts_name]

        # if (epoch != 0) and ((epoch + 1) % (100)) == 0:
        #    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.8

        if epoch < start_epoch:
            continue

        batch_count = 0
        for (
            ((x_timestamps, x_features), y_timestamps),
            y_features,
        ), t_inferences in tqdm(train_dl):

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
                        torch.nn.functional.mse_loss(
                            f[i],
                            torch.cat([x_features[ts_name][i], y_features[ts_name][i]]),
                        )
                        # 1.0
                        # - index_agreement_torch(
                        #    f[i],
                        #    torch.cat(
                        #        [x_features[ts_name][i], y_features[ts_name][i]]
                        #    ).squeeze(),
                        # )
                        for i in range(len(f))
                        if f[i].size(0) > 0
                    ]
                )
                for ts_name, f in forecast.items()
            }

            loss = torch.cat(
                [mean.mean(dim=0).reshape(-1) for mean in losses_by_ts.values()],
                dim=-1,
            ).mean()
            loss.backward()
            optimizer.step()

            loss_meter_current.update(
                losses_by_ts["current_praticagem"].detach().cpu().mean().item(), 0
            )
            loss_meter_waves.update(
                losses_by_ts["waves_palmas"].detach().cpu().mean().item(), 0
            )

            batch_count += 1

        print(
            f"TRAIN: Epoch {epoch} Loss Current: {round(loss_meter_current.val, 3)} ({round(loss_meter_current.avg, 3)}) Loss Waves: {round(loss_meter_waves.val, 3)} ({round(loss_meter_waves.avg, 3)})"
        )
        print(
            f" ContextLength: {train_dataset._context_window_lengths['waves_palmas']} ForecastLength: {train_dataset._target_window_lengths['waves_palmas']} Lr: {optimizer.param_groups[0]['lr']}"
        )

        ts_id = 0
        x_timestamp_current = x_timestamps["current_praticagem"][ts_id]
        x_timestamp_waves = x_timestamps["waves_palmas"][ts_id]
        y_timestamp_current = y_timestamps["current_praticagem"][ts_id]
        y_timestamp_waves = y_timestamps["waves_palmas"][ts_id]

        current_timestamp = (
            torch.cat([x_timestamp_current, y_timestamp_current]).cpu().numpy()
        )
        current_timestamp = (
            current_timestamp - current_timestamp[0]
            if current_timestamp.size > 0
            else current_timestamp
        )
        waves_timestamp = (
            torch.cat([x_timestamp_waves, y_timestamp_waves]).cpu().numpy()
        )
        waves_timestamp = (
            waves_timestamp - waves_timestamp[0]
            if waves_timestamp.size > 0
            else waves_timestamp
        )

        current_features = (
            torch.cat(
                [
                    x_features["current_praticagem"][ts_id],
                    y_features["current_praticagem"][ts_id],
                ]
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        current_features_hat = (
            forecast["current_praticagem"][ts_id].squeeze().cpu().detach().numpy()
        )

        waves_features = (
            torch.cat(
                [x_features["waves_palmas"][ts_id], y_features["waves_palmas"][ts_id]]
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        waves_features_hat = forecast["waves_palmas"][ts_id].cpu().detach().numpy()

        if current_timestamp.size > 0:
            uniplot.plot(
                xs=[current_timestamp, current_timestamp],
                ys=[current_features, current_features_hat],
                color=True,
                legend_labels=["Target", "Forecast"],
                title="Current Praticagem",
            )

        if waves_timestamp.size > 0:
            uniplot.plot(
                xs=[waves_timestamp, waves_timestamp],
                ys=[waves_features[:, 0], waves_features_hat[:, 0]],
                color=True,
                legend_labels=["Target", "Forecast"],
                title="Waves Palmas hs",
            )

            uniplot.plot(
                xs=[waves_timestamp, waves_timestamp],
                ys=[waves_features[:, 1], waves_features_hat[:, 1]],
                color=True,
                legend_labels=["Target", "Forecast"],
                title="Waves Palmas tp",
            )

            uniplot.plot(
                xs=[waves_timestamp, waves_timestamp],
                ys=[waves_features[:, 2], waves_features_hat[:, 2]],
                color=True,
                legend_labels=["Target", "Forecast"],
                title="Waves Palmas ws",
            )

        model.zero_grad()

        batch_count = 0

        io_loss_list = {
            "current_praticagem": [],
            "waves_palmas": [],
        }
        mse_loss_list = {
            "current_praticagem": [],
            "waves_palmas": [],
        }

        for (
            ((x_timestamps, x_features), y_timestamps),
            y_features,
        ) in tqdm(test_dl):

            x_timestamps = {
                ts_name: [xi.to(device) for xi in x]
                for ts_name, x in x_timestamps.items()
            }
            x_features = {
                ts_name: [
                    (
                        train_dataset.normalize_ts(ts_name, xi).to(device)
                        if normalize
                        else xi.to(device)
                    )
                    for xi in x
                ]
                for ts_name, x in x_features.items()
            }
            y_timestamps = {
                ts_name: [yi.to(device) for yi in y]
                for ts_name, y in y_timestamps.items()
            }

            y_features = {
                ts_name: [
                    (
                        train_dataset.normalize_ts(ts_name, yi).to(device)
                        if normalize
                        else yi.to(device)
                    )
                    for yi in y
                ]
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

                X = {
                    ts_name: [
                        torch.cat(
                            [x_features[ts_name][ts_id], y_features[ts_name][ts_id]]
                        )
                        for ts_id in range(len(t_inferences))
                    ]
                    for ts_name in y_features.keys()
                }

                mse_by_ts = {
                    ts_name: torch.stack(
                        [
                            torch.nn.functional.mse_loss(
                                f[i],
                                X[ts_name][i],
                            )
                            for i in range(len(f))
                            if f[i].size(0) > 0
                        ]
                    )
                    for ts_name, f in forecast.items()
                }

                io_by_ts = {
                    ts_name: torch.stack(
                        [
                            index_agreement_torch(
                                f[i],
                                X[ts_name][i],
                            ).mean()
                            for i in range(len(f))
                            if f[i].size(0) > 0
                        ]
                    )
                    for ts_name, f in forecast.items()
                }

                io_loss_list["current_praticagem"].append(
                    io_by_ts["current_praticagem"].mean().item()
                )
                io_loss_list["waves_palmas"].append(
                    io_by_ts["waves_palmas"].mean().item()
                )
                mse_loss_list["current_praticagem"].append(
                    mse_by_ts["current_praticagem"].mean().item()
                )
                mse_loss_list["waves_palmas"].append(
                    mse_by_ts["waves_palmas"].mean().item()
                )

        io_loss_list["current_praticagem"] = [
            elem for elem in io_loss_list["current_praticagem"] if not np.isnan(elem)
        ]

        io_loss_list["waves_palmas"] = [
            elem for elem in io_loss_list["waves_palmas"] if not np.isnan(elem)
        ]

        mse_loss_list["current_praticagem"] = [
            elem for elem in mse_loss_list["current_praticagem"] if not np.isnan(elem)
        ]

        mse_loss_list["waves_palmas"] = [
            elem for elem in mse_loss_list["waves_palmas"] if not np.isnan(elem)
        ]

        print(f"TEST: Epoch {epoch}")
        print(
            f"Current IO: {np.mean(io_loss_list['current_praticagem'])} MSE: {np.mean(mse_loss_list['current_praticagem'])}"
        )
        print(
            f"Waves IO: {np.mean(io_loss_list['waves_palmas'])} MSE: {np.mean(mse_loss_list['waves_palmas'])}"
        )

        for ts_name in y_features.keys():
            ts_id = np.random.randint(0, len(y_features[ts_name]))
            x_timestamp = x_timestamps[ts_name][ts_id]
            x_feature = x_features[ts_name][ts_id]
            y_timestamp = y_timestamps[ts_name][ts_id]
            y_feature = y_features[ts_name][ts_id]

            timestamps = torch.cat([x_timestamp, y_timestamp]).cpu().numpy()
            features = torch.cat([x_feature, y_feature]).cpu().numpy()
            timestamps = timestamps - timestamps[0]

            data_dim = features.shape[1]
            for dim in range(data_dim):
                feature_rec = forecast[ts_name][ts_id]
                if len(feature_rec.shape) == 1:
                    feature_rec = feature_rec.unsqueeze(-1)
                feature_rec = feature_rec[:, dim].cpu().detach().numpy()
                feature_true = features[:, dim]

                if timestamps.size > 0:
                    uniplot.plot(
                        xs=[timestamps, timestamps],
                        ys=[feature_true, feature_rec],
                        legend_labels=["True", "Predicted"],
                        title=f"{ts_name}: {dim}",
                        color=True,
                    )

        if ((epoch + 1) % 10) == 0:
            torch.save(model.state_dict(), out_path / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    main()
