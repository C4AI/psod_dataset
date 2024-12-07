import datetime
import pathlib
import numpy as np
import polars as pl


def main():
    base_folder = pathlib.Path("data/02_processed/test")
    files = list(base_folder.glob("*.parquet"))

    targets = ["odas_corrente", "cattalini_meteorologia", "porto_maregrafo"]

    missing_data_ratios = [0.0]

    test_dfs = {file.stem: pl.read_parquet(file) for file in files}

    no_look_ahead = [
        "cattalini_corrente",
        "cattalini_maregrafo",
        "cattalini_meteorologia",
        "odas_corrente",
        "odas_meteorologia",
        "porto_maregrafo",
    ]

    look_ahead = ["porto_astronomica", "porto_harmonico"]

    t_inferences = (
        (base_df := test_dfs["porto_harmonico"])
        .group_by_dynamic(pl.col("datetime"), every="1d")
        .agg(pl.len())
        .drop(["len"])
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

    for missing_ratio in missing_data_ratios:
        context_out_folder = (
            base_folder / f"missing_ratio_{int(missing_ratio*100)}" / "context_masks"
        )
        context_out_folder.mkdir(parents=True, exist_ok=True)

        target_out_folder = (
            base_folder / f"missing_ratio_{int(missing_ratio*100)}" / "target_masks"
        )
        target_out_folder.mkdir(parents=True, exist_ok=True)
        context_indices: dict[str, pl.DataFrame] = {}
        target_indices: dict[str, pl.DataFrame] = {}
        for ts_name, df in test_dfs.items():
            missing_mask = create_missing_data_mask(df, missing_ratio)
            for (t_inf,) in t_inferences.iter_rows():

                if ts_name in no_look_ahead:
                    df_missing = df.with_columns(
                        [
                            ((pl.col("datetime") < t_inf)).alias("context_index"),
                        ]
                    )

                elif ts_name in look_ahead:
                    df_missing = df.with_columns(
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

                df_missing = df_missing.with_columns(
                    [(pl.col("context_index") & missing_mask).alias("context_index")]
                )

                if ts_name not in context_indices:
                    context_indices[ts_name] = pl.DataFrame(
                        {
                            "1".rjust(3, "0"): df_missing["context_index"],
                        }
                    )
                else:
                    curr_width = str(context_indices[ts_name].width + 1).rjust(3, "0")
                    context_indices[ts_name] = context_indices[ts_name].hstack(
                        [df_missing["context_index"].rename(curr_width)]
                    )

                if ts_name in targets:
                    df_missing = df.with_columns(
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
                                "1".rjust(3, "0"): df_missing["target_index"],
                            }
                        )

                    else:
                        curr_width = str(target_indices[ts_name].width + 1).rjust(
                            3, "0"
                        )
                        target_indices[ts_name] = target_indices[ts_name].hstack(
                            [df_missing["target_index"].rename(curr_width)]
                        )

        for ts_name, df in context_indices.items():

            df.write_parquet(context_out_folder / f"{ts_name}_context.parquet")

        for ts_name, df in target_indices.items():

            df.write_parquet(target_out_folder / f"{ts_name}_target.parquet")


def create_missing_data_mask(
    df: pl.DataFrame, ratio: float, missing_window_len: datetime.timedelta | None = None
) -> pl.Series:

    if missing_window_len is None:
        missing_window_len = datetime.timedelta(hours=4)

    mask = pl.Series("missing_mask", [True] * df.height)
    while mask.not_().sum() < ratio * df.height:
        # sample index from 0 to height-1
        lbound = np.random.randint(0, df.height)
        ubound = df["datetime"].search_sorted(
            df["datetime"][lbound] + missing_window_len
        )

        new_mask = pl.concat(
            [
                mask.slice(0, lbound),
                pl.Series("missing_mask", [False] * (ubound - lbound)),
                mask.slice(ubound, df.height),
            ]
        )

        mask = mask & new_mask

    return mask


if __name__ == "__main__":
    main()
