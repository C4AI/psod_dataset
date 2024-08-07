import pathlib
from collections import defaultdict
import pickle
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
import polars as pl

from processing.loader import (
    SantosTestDatasetTorch,
)


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


def normalize(ts_name, ts, scalers):
    if ts.size(0) == 0:
        return ts
    return torch.tensor(scalers[ts_name].transform(ts.cpu()), dtype=torch.float32)


def inverse_normalize(ts_name, ts, scalers):
    if ts.size(0) == 0:
        return ts
    return torch.tensor(
        scalers[ts_name].inverse_transform(ts.cpu()), dtype=torch.float32
    )


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
        "dt": 0.1,
        "use_adjoint": "no_adjoint",
        "T_in": 382,
        "rnn_hidden": 100 * double_model,
        "dec_H": 100 * double_model,
        "dec_L": 2,
        "dec_act": "relu",
        "enc_H": 100 * double_model,
        "train_data_path": "data/02_processed/train",
        "test_data_path": "data/02_processed/test",
        "sample_size_per_epoch": 14000,
        "ts_name": "waves_palmas",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.float32,
        "batch_size": 64,
        "epochs": 800,
        "lr": 2e-3,
        "data_dim": 3,
        "N_increments": 50,
        "start_epoch": 0,
        "model_path": "20240730093048_monode_waves_palmas_False_1/epoch_759.pt",
        "normalize": False,
    }
    ts_name = args["ts_name"]

    model_folder = "data/04_model_output/"
    model_path = pathlib.Path(model_folder) / args["model_path"]

    out_path = f"data/05_inference_results/" + model_path.parent.name

    out_path = pathlib.Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if not ((train_data_path := pathlib.Path(args["train_data_path"])).exists()):
        raise FileNotFoundError(f"Train data path {train_data_path} does not exist")

    if not ((test_data_path := pathlib.Path(args["test_data_path"])).exists()):
        raise FileNotFoundError(f"Test data path {test_data_path} does not exist")

    context_window_lengths = defaultdict(
        float,
        {
            "current_praticagem": 60 * 12,
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
            "current_praticagem": 60 * 12,
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
    model.load_state_dict(torch.load(model_path))

    model.eval()
    device = args["device"]
    model.to(device)

    if args["normalize"]:
        scaler = list(model_path.parent.glob("scaler.pkl"))
        scalers = pickle.load(open(scaler[0], "rb"))

    elem_counts = {ts_name: 1}

    for (
        ((x_timestamps, x_features), y_timestamps),
        y_features,
    ) in test_dl:

        x_timestamps = {
            ts_name: [xi.to(device) for xi in x] for ts_name, x in x_timestamps.items()
        }
        x_features = {
            ts_name: [
                (
                    normalize(ts_name, xi, scalers).to(device)
                    if args["normalize"]
                    else xi.to(device)
                )
                for xi in x
            ]
            for ts_name, x in x_features.items()
        }
        y_timestamps = {
            ts_name: [yi.to(device) for yi in y] for ts_name, y in y_timestamps.items()
        }

        y_features = {
            ts_name: [yi.to(device) for yi in y] for ts_name, y in y_features.items()
        }

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

            for ts_id, elem in enumerate(Xrec):
                elem_ = elem[x_features[ts_name][ts_id].size(0) :, :]
                elem_ = (
                    inverse_normalize(ts_name, elem_, scalers)
                    if args["normalize"]
                    else elem_
                )
                df = pl.DataFrame(
                    elem_.detach().cpu().numpy(),
                    schema={
                        feature_name: pl.Float32
                        for feature_name in test_dataset.feature_names[ts_name]
                    },
                )
                full_out_path = (
                    out_path
                    / ts_name
                    / f"{str(elem_counts[ts_name]).rjust(3,'0')}.parquet"
                )
                full_out_path.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(full_out_path)

                elem_counts[ts_name] += 1


if __name__ == "__main__":
    main()
