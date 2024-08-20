import polars as pl
import pathlib


def main():
    base_folder = pathlib.Path("data/05_inference_results/nhits/waves_palmas_sep")
    output_folder = pathlib.Path("data/05_inference_results/nhits/waves_palmas")

    ratios = [0, 20, 40, 60, 80]

    for ratio in ratios:

        ref_files = list(
            (base_folder / "hs" / f"missing_ratio_{ratio}").glob("*.parquet")
        )

        for f in ref_files:

            out_f = output_folder / f"missing_ratio_{ratio}"
            out_f.mkdir(parents=True, exist_ok=True)
            df = pl.concat(
                [
                    pl.read_parquet(f).select(pl.col("0").alias("hs")),
                    pl.read_parquet(
                        f.parent.parent.parent
                        / "tp"
                        / f"missing_ratio_{ratio}"
                        / f"{f.stem}.parquet"
                    ).select(pl.col("0").alias("tp")),
                    pl.read_parquet(
                        f.parent.parent.parent
                        / "ws"
                        / f"missing_ratio_{ratio}"
                        / f"{f.stem}.parquet"
                    ).select(pl.col("0").alias("ws")),
                ],
                how="horizontal",
            )
            df.write_parquet(out_f / f"{f.stem}.parquet")


if __name__ == "__main__":
    main()
