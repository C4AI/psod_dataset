import datetime
import pathlib
import polars as pl


def main():
    files = list(pathlib.Path("data/01_raw").glob("*.csv"))

    targets = ["current_praticagem", "waves_palmas"]

    min_datetime = datetime.datetime(2018, 1, 1, tzinfo=datetime.timezone.utc)

    max_datetime = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)

    test_cutoff = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)

    dfs = {file.stem: pl.read_csv(file) for file in files}
    train_folder = pathlib.Path("data/02_processed/train")
    train_folder.mkdir(parents=True, exist_ok=True)

    test_folder = pathlib.Path("data/02_processed/test")
    test_folder.mkdir(parents=True, exist_ok=True)

    ### Astronomical Tide
    df = dfs["astronomical_tide"]
    df = (
        df.drop([""])
        .with_columns(
            [
                pl.col("datetime")
                .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms", time_zone="UTC")
                .alias("datetime"),
                pl.col("elevation").cast(pl.Float32).alias("astronomical_tide"),
            ]
        )
        .drop(["elevation"])
        .sort("datetime")
    )

    dfs["astronomical_tide"] = df

    ### SOFS
    df = dfs["sofs_praticagem"]
    df = (
        (
            df.with_columns(
                [
                    pl.col("Datetime")
                    .str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f", time_unit="ms", time_zone="UTC"
                    )
                    .alias("datetime"),
                    pl.col("Projected_current")
                    .cast(pl.Float32)
                    .alias("cross_shore_current"),
                ]
            )
            .sort("datetime")
            .drop(["Datetime", "Direction", "Velocity_ms", "Projected_current"])
        )
        .rename({"SSH": "ssh"})
        .with_columns([pl.col("ssh").cast(pl.Float32)])
        .select(["datetime", "cross_shore_current", "ssh"])
        .filter(
            pl.col("cross_shore_current").is_not_null() & pl.col("ssh").is_not_null()
        )
    )

    dfs["sofs_praticagem"] = df

    ### SSH
    df = dfs["ssh_praticagem"]
    df = (
        df.with_columns(
            [
                pl.col("Datetime")
                .str.to_datetime(
                    "%Y-%m-%d %H:%M:%S", time_unit="ms", time_zone="Etc/GMT+3"
                )  # sign is inverted to conform to Posix standard
                .dt.convert_time_zone("UTC")
                .alias("datetime")
            ]
        )
        .sort("datetime")
        .rename({"Maré real": "ssh"})
        .with_columns([pl.col("ssh").cast(pl.Float32)])
        .drop(["Datetime"])
        .filter(pl.col("ssh").is_not_null())
    ).select(["datetime", "ssh"])

    dfs["ssh_praticagem"] = df

    ### Waves
    df = dfs["waves_palmas"]
    df = (
        df.with_columns(
            [
                pl.col("Data")
                .str.to_datetime(
                    "%d/%m/%Y %H:%M", time_unit="ms", time_zone="Etc/GMT+3"
                )
                .dt.convert_time_zone("UTC")
                .alias("datetime"),
                pl.col("Altura").cast(pl.Float32).alias("hs"),
                pl.col("Período").cast(pl.Float32).alias("tp"),
                pl.col("Direção").alias("dp"),
                pl.col("Intensidade").alias("ws"),
            ]
        )
        .with_columns(
            [
                (((pl.col("dp") - 25).radians().cos()) * pl.col("ws"))
                .cast(pl.Float32)
                .round(2)
                .alias("ws"),
            ]
        )
        .drop(["Data", "Altura", "Período", "Direção", "Intensidade"])
        .select(["datetime", "hs", "tp", "ws"])
        .sort("datetime")
        .filter(
            pl.col("hs").is_not_null()
            & pl.col("tp").is_not_null()
            & pl.col("ws").is_not_null()
        )
    )

    dfs["waves_palmas"] = df

    ### Wind
    df = dfs["wind_praticagem"]
    df = (
        df.with_columns(
            [
                pl.col("datetime")
                .str.to_datetime("%Y-%m-%d %H:%M:%S", time_unit="ms", time_zone="UTC")
                .alias("datetime"),
                pl.col("vx").round(2).cast(pl.Float32).alias("vx"),
                pl.col("vy").round(2).cast(pl.Float32).alias("vy"),
            ]
        )
        .sort("datetime")
        .filter(pl.col("vx").is_not_null() & pl.col("vy").is_not_null())
    ).select(["datetime", "vx", "vy"])

    dfs["wind_praticagem"] = df

    ### Current

    df = dfs["current_praticagem"]
    df = (
        df.with_columns(
            [
                pl.col("Datetime")
                .str.to_datetime(
                    "%Y-%m-%d %H:%M:%S%:z", time_unit="ms", time_zone="UTC"
                )
                .alias("datetime"),
                pl.col("Projected_current")
                .cast(pl.Float32)
                .round(2)
                .alias("cross_shore_current"),
            ]
        )
        .sort("datetime")
        .drop(["Direction", "Velocity_ms", "Projected_current", "Datetime"])
        .filter(pl.col("cross_shore_current").is_not_null())
    ).select(["datetime", "cross_shore_current"])

    dfs["current_praticagem"] = df

    #############

    train_dfs = {
        ts_name: df.filter(
            (
                (pl.col("datetime") >= min_datetime)
                & (pl.col("datetime") < max_datetime)
                & (pl.col("datetime") < test_cutoff)
            )
        )
        for ts_name, df in dfs.items()
    }

    for ts_name, df in train_dfs.items():
        df.write_parquet(train_folder / f"{ts_name}.parquet")

    test_dfs = {
        ts_name: df.filter(
            (pl.col("datetime") >= min_datetime)
            & (pl.col("datetime") < max_datetime)
            & (pl.col("datetime") >= test_cutoff)
        )
        for ts_name, df in dfs.items()
    }

    t_inferences = (
        (base_df := test_dfs["astronomical_tide"])
        .group_by_dynamic(pl.col("datetime"), every="1d")
        .agg(pl.count())
        .drop(["count"])
        .filter(
            (
                pl.col("datetime")
                >= (base_df["datetime"].min() + datetime.timedelta(days=7))
            )
            & (
                pl.col("datetime")
                < (base_df["datetime"].max() - datetime.timedelta(days=2))
            )
        )
        .sort("datetime")
    )

    context_indices: dict[str, pl.DataFrame] = {}
    target_indices: dict[str, pl.DataFrame] = {}

    for (t_inf,) in t_inferences.iter_rows():
        for ts_name, df in test_dfs.items():
            if ts_name in [
                "waves_palmas",
                "ssh_praticagem",
                "current_praticagem",
                "wind_praticagem",
            ]:
                df = df.with_columns(
                    [
                        ((pl.col("datetime") < t_inf)).alias("context_index"),
                    ]
                )

            elif ts_name in ["sofs_praticagem", "astronomical_tide"]:
                df = df.with_columns(
                    [
                        (
                            (pl.col("datetime") >= t_inf)
                            & (
                                pl.col("datetime")
                                < (t_inf + datetime.timedelta(days=2))
                            )
                        ).alias("context_index"),
                    ]
                )

            if ts_name not in context_indices:
                context_indices[ts_name] = pl.DataFrame(
                    {
                        "1".rjust(3, "0"): df["context_index"],
                    }
                )
            else:
                curr_width = str(context_indices[ts_name].width + 1).rjust(3, "0")
                context_indices[ts_name] = context_indices[ts_name].hstack(
                    [df["context_index"].rename(curr_width)]
                )

            if ts_name in targets:
                df = df.with_columns(
                    [
                        (
                            (pl.col("datetime") >= t_inf)
                            & (
                                pl.col("datetime")
                                < (t_inf + datetime.timedelta(days=2))
                            )
                        ).alias("target_index"),
                    ]
                )

                if ts_name not in target_indices:
                    target_indices[ts_name] = pl.DataFrame(
                        {
                            "1".rjust(3, "0"): df["target_index"],
                        }
                    )

                else:
                    curr_width = str(target_indices[ts_name].width + 1).rjust(3, "0")
                    target_indices[ts_name] = target_indices[ts_name].hstack(
                        [df["target_index"].rename(curr_width)]
                    )
    for ts_name, df in test_dfs.items():
        df.write_parquet(test_folder / f"{ts_name}.parquet")

    for ts_name, df in context_indices.items():
        (tgt_folder := test_folder / "context_masks").mkdir(parents=True, exist_ok=True)
        df.write_parquet(tgt_folder / f"{ts_name}_context.parquet")

    for ts_name, df in target_indices.items():
        (tgt_folder := test_folder / "target_masks").mkdir(parents=True, exist_ok=True)
        df.write_parquet(tgt_folder / f"{ts_name}_target.parquet")


if __name__ == "__main__":
    main()
