import pathlib
from collections import defaultdict
from dataset import SantosTrainDatasetTorch
from model.build_model import build_model
from model.model_misc import compute_loss
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.misc.log_utils import CachedRunningAverageMeter
import uniplot
from tqdm import tqdm
import datetime
import pickle
import random
import os

from processing.loader import (
    SantosTestDatasetTorch,
)


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

    double_model = 1

    args = {
        "Nobj": 1,
        "ode_latent_dim": 32 * double_model,
        "de_L": 2,
        "de_H": 100 * double_model,
        "modulator_dim": 32 * double_model,
        "inv_fnc": "MLP",
        "content_dim": 0,
        "T_inv": 350,
        "order": 1,
        "solver": "rk4",
        "dt": 0.0001,
        "use_adjoint": "adjoint",
        "T_in": 382,
        "rnn_hidden": 100 * double_model,
        "dec_H": 100 * double_model,
        "dec_L": 2,
        "dec_act": "relu",
        "enc_H": 100 * double_model,
        "train_data_path": "data/02_processed/train",
        "test_data_path": "data/02_processed/test",
        "sample_size_per_epoch": 20000,
        "ts_name": "waves_palmas",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.float32,
        "batch_size": 64,
        "epochs": 500,
        "lr": 2e-3,
        "data_dim": 3,
        "N_increments": 50,
        "start_epoch": 0,
        "model_path": None,
        "normalize": True,
        "max_context_window_length": 60 * 24 * 4,
    }

    seed_everything(42)

    ts_name = args["ts_name"]

    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_path = f"data/04_model_output/{now_str}_monode_{ts_name}_{args['normalize']}_{double_model}"

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if not ((train_data_path := pathlib.Path(args["train_data_path"])).exists()):
        raise FileNotFoundError(f"Train data path {train_data_path} does not exist")

    if not ((test_data_path := pathlib.Path(args["test_data_path"])).exists()):
        raise FileNotFoundError(f"Test data path {test_data_path} does not exist")

    context_window_lengths = defaultdict(
        float,
        {
            "current_praticagem": 60 * 18,
            "waves_palmas": 60 * 24,
        },
    )

    target_window_lengths = defaultdict(
        float,
        {
            "current_praticagem": 60 * 48.0,
            "waves_palmas": 60 * 48.0,
        },
    )

    max_context_window_lengths = defaultdict(
        float,
        {
            "astronomical_tide": 60 * 24.0 * 2,
            "current_praticagem": 60 * 24,
            "sofs_praticagem": 60 * 24.0 * 2,
            "ssh_praticagem": 60 * 24.0 * 7,
            "waves_palmas": 60 * 24,
            "wind_praticagem": 60 * 24.0 * 7,
        },
    )

    look_ahead_lengths = defaultdict(
        float,
        {
            "astronomical_tide": 60 * 48.0,
            "sofs_praticagem": 60 * 48.0,
        },
    )

    target_window_lengths_increment = {
        ts_name: length // args["N_increments"]
        for ts_name, length in target_window_lengths.items()
    }

    context_window_lengths_increment = {
        ts_name: (args["max_context_window_length"] - length) // args["N_increments"]
        for ts_name, length in context_window_lengths.items()
    }

    train_dataset = SantosTrainDatasetTorch(
        data_path=train_data_path,
        context_window_lengths=context_window_lengths,
        target_window_lengths=target_window_lengths,
        look_ahead_lengths=look_ahead_lengths,
        sample_size_per_epoch=args["sample_size_per_epoch"],
        ts_name=ts_name,
        normalize=args["normalize"],
        scaler="standard",
    )

    if args["normalize"]:

        if args["model_path"]:
            scaler = list(pathlib.Path(args["model_path"]).parent.glob("scaler.pkl"))
            if len(scaler) != 0:
                train_dataset.scalers = pickle.load(open(scaler[0], "rb"))

        with open(out_path / "scaler.pkl", "wb") as f:
            pickle.dump(train_dataset.scalers, f)

    train_dl = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
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
        batch_size=args["batch_size"],
        shuffle=False,
        collate_fn=SantosTestDatasetTorch.collate_fn,
    )

    model = build_model(**args)

    if args["model_path"]:
        model.load_state_dict(torch.load(args["model_path"]))

    model.to(args["device"]).to(args["dtype"])

    print(count_params(model), "parameters")

    device = args["device"]

    loss_meter = CachedRunningAverageMeter(0.97)
    tr_mse_meter = CachedRunningAverageMeter(0.97)
    io_loss_test = CachedRunningAverageMeter(0.97)
    mse_meter_test = CachedRunningAverageMeter(0.97)
    global_itr = 0

    train_dataset._target_window_lengths = target_window_lengths_increment.copy()

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    epochs = args["epochs"]
    for epoch in range(epochs):

        if (epoch != 0) and ((epoch + 1) % (epochs / args["N_increments"])) == 0:
            train_dataset.change_target_length(
                target_window_lengths_increment[ts_name], ts_name
            )

        if (epoch != 0) and ((epoch + 1) % (100)) == 0:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.8

            train_dataset._context_window_lengths[
                ts_name
            ] += context_window_lengths_increment[ts_name]

        # if epoch == epochs // 2:
        #    train_dataset.context_window_lengths[ts_name] *= 2

        if epoch < args["start_epoch"]:
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

            Xrec, s0_mu, s0_logv, v0_mu, v0_logv = model(
                context_timestamps=x_timestamps,
                context_features=x_features,
                target_timestamps=y_timestamps,
                t_inferences=t_inferences,
                ts_name=ts_name,
            )

            X = [
                torch.cat([x_features[ts_name][ts_id], y_features[ts_name][ts_id]])
                for ts_id in range(len(t_inferences))
            ]

            loss, nlhood, kl_z0, mse = compute_loss(
                model, X, Xrec, s0_mu, s0_logv, v0_mu, v0_logv
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), global_itr)
            tr_mse_meter.update(mse.item(), global_itr)
            global_itr += 1

        print(
            f"TRAIN: Epoch {epoch} Loss: {loss_meter.val} MSE: {tr_mse_meter.val} ContextLength: {train_dataset._context_window_lengths['current_praticagem']} ForecastLength: {train_dataset._target_window_lengths[ts_name]} Lr: {optimizer.param_groups[0]['lr']}"
        )

        ts_id = 0
        x_timestamp = x_timestamps[ts_name][ts_id]
        x_feature = x_features[ts_name][ts_id]
        y_timestamp = y_timestamps[ts_name][ts_id]
        y_feature = y_features[ts_name][ts_id]

        timestamps = torch.cat([x_timestamp, y_timestamp]).cpu().numpy()
        features = torch.cat([x_feature, y_feature]).cpu().numpy()
        timestamps = timestamps - timestamps[0]

        data_dim = features.shape[1]
        for dim in range(data_dim):
            feature_rec = Xrec[ts_id][:, dim].cpu().detach().numpy()
            feature_true = features[:, dim]

            uniplot.plot(
                xs=[timestamps, timestamps],
                ys=[feature_true, feature_rec],
                legend_labels=["True", "Predicted"],
                title=f"{ts_name}: {dim}",
                color=True,
            )

        io_loss_list = []
        mse_loss_list = []

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
                        if args["normalize"]
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
                        if args["normalize"]
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

                Xrec, s0_mu, s0_logv, v0_mu, v0_logv = model(
                    context_timestamps=x_timestamps,
                    context_features=x_features,
                    target_timestamps=y_timestamps,
                    t_inferences=t_inferences,
                    ts_name=ts_name,
                )

                X = [
                    torch.cat([x_features[ts_name][ts_id], y_features[ts_name][ts_id]])
                    for ts_id in range(len(t_inferences))
                ]

                mse_loss = torch.tensor(
                    [
                        torch.nn.functional.mse_loss(Xrec[ts_id], X[ts_id])
                        for ts_id in range(len(t_inferences))
                    ]
                ).mean()

                io_loss = torch.tensor(
                    [
                        index_agreement_torch(Xrec[ts_id], X[ts_id]).mean()
                        for ts_id in range(len(t_inferences))
                    ]
                ).mean()

                io_loss_list.append(io_loss.item())
                mse_loss_list.append(mse_loss.item())
                mse_meter_test.update(mse_loss.item(), global_itr)
                io_loss_test.update(io_loss.item(), global_itr)

        print(
            f"TEST: Epoch {epoch} IO: {np.mean(io_loss_list)} MSE: {np.mean(mse_loss_list)}"
        )

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
            feature_rec = Xrec[ts_id][:, dim].cpu().detach().numpy()
            feature_true = features[:, dim]

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
