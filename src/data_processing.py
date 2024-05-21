import datetime
import pathlib
import polars as pl


def main():
    files = list(pathlib.Path("data/01_raw").glob("*.csv"))

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
    ).filter((pl.col("datetime") >= min_datetime) & (pl.col("datetime") < max_datetime))



    df.filter(pl.col("datetime") < test_cutoff).write_parquet(
        f"{train_folder}/astronomical_tide.parquet"
    )


    ### SOFS
    df = dfs["current_sofs_praticagem_15min_projected_2019-2023"]
    df = (
        (
            df.with_columns(
                [
                    pl.col("Datetime")
                    .str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%.9f", time_unit="ms", time_zone="UTC"
                    )
                    .alias("datetime"),
                    pl.col("Projected_current").cast(pl.Float32).alias("cross_shore"),
                    (
                        pl.col("Velocity_ms")
                        * ((pl.col("Direction") - 77).radians().sin())
                    )
                    .cast(pl.Float32)
                    .alias("along_shore"),
                ]
            )
            .sort("datetime")
            .drop(["Datetime", "Direction", "Velocity_ms", "Projected_current"])
        )
        .rename({"SSH": "ssh"})
        .select(["datetime", "cross_shore", "along_shore", "ssh"])
    ).filter((pl.col("datetime") >= min_datetime) & (pl.col("datetime") < max_datetime))

    df.filter(pl.col("datetime") < test_cutoff).write_parquet(
        f"{train_folder}/sofs_praticagem.parquet"
    )

    ### SSH
    df = dfs["ssh_praticagem"]
    df = (
        (
            df.with_columns(
                [
                    pl.col("Datetime")
                    .str.to_datetime(
                        "%Y-%m-%d %H:%M:%S", time_unit="ms", time_zone="UTC"
                    )
                    .alias("datetime")
                ]
            )
            .sort("datetime")
            .rename({"Maré real": "ssh"})
            .with_columns([pl.col("ssh").cast(pl.Float32)])
            .drop(["Datetime"])
        ).select(["datetime", "ssh"])
    ).filter((pl.col("datetime") >= min_datetime) & (pl.col("datetime") < max_datetime))

    df.filter(pl.col("datetime") < test_cutoff).write_parquet(
        f"{train_folder}/ssh_praticagem.parquet"
    )

    ### Waves
    df = dfs["waves_palmas"]
    df = (
        df.with_columns(
            [
                pl.col("Data")
                .str.to_datetime("%d/%m/%Y %H:%M", time_unit="ms", time_zone="UTC")
                .alias("datetime"),
                pl.col("Altura").cast(pl.Float32).alias("hs"),
                pl.col("Período").cast(pl.Float32).alias("tp"),
                pl.col("Direção").cast(pl.Float32).alias("dp"),
                pl.col("Intensidade").cast(pl.Float32).alias("ws"),
            ]
        )
        .drop(["Data", "Altura", "Período", "Direção", "Intensidade"])
        .select(["datetime", "hs", "tp", "dp", "ws"])
        .sort("datetime")
    ).filter((pl.col("datetime") >= min_datetime) & (pl.col("datetime") < max_datetime))

    df.filter(pl.col("datetime") < test_cutoff).write_parquet(
        f"{train_folder}/waves_palmas.parquet"
    )

    ### Wind
    df = dfs["wind_praticagem"]
    df = (
        (
            df.with_columns(
                [
                    pl.col("datetime")
                    .str.to_datetime(
                        "%Y-%m-%d %H:%M:%S", time_unit="ms", time_zone="UTC"
                    )
                    .alias("datetime"),
                    pl.col("vx").cast(pl.Float32),
                    pl.col("vy").cast(pl.Float32),
                ]
            ).sort("datetime")
        ).select(["datetime", "vx", "vy"])
    ).filter((pl.col("datetime") >= min_datetime) & (pl.col("datetime") < max_datetime))

    df.filter(pl.col("datetime") < test_cutoff).write_parquet(
        f"{train_folder}/wind_praticagem.parquet"
    )

    ### Current

    df = dfs["Currents_praticagem_projected_2015-2023"]
    df = (
        (
            df.with_columns(
                [
                    pl.col("Datetime")
                    .str.to_datetime(
                        "%Y-%m-%d %H:%M:%S%:z", time_unit="ms", time_zone="UTC"
                    )
                    .alias("datetime"),
                    pl.col("Projected_current").cast(pl.Float32).alias("cross_shore"),
                    (
                        pl.col("Velocity_ms")
                        * ((pl.col("Direction") - 91).radians().sin()).cast(pl.Float32)
                    ).alias("along_shore"),
                ]
            )
            .sort("datetime")
            .drop(["Direction", "Velocity_ms", "Projected_current", "Datetime"])
        ).select(["datetime", "cross_shore", "along_shore"])
    ).filter((pl.col("datetime") >= min_datetime) & (pl.col("datetime") < max_datetime))

    df.filter(pl.col("datetime") < test_cutoff).write_parquet(
        f"{train_folder}/current_praticagem.parquet"
    )

    print(dfs)


if __name__ == "__main__":
    main()
