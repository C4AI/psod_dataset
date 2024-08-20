import polars as pl
import pathlib
import sys

sys.path.append("src")

from processing.loader import SantosTestDatasetNumpy


def index_agreement(s: pl.DataFrame, o: pl.DataFrame) -> pl.DataFrame:
    """
    index of agreement
    Willmott (1981, 1982)

    Args:
        s: simulated
        o: observed

    Returns:
        ia: index of agreement
    """
    if s.height != o.height:
        raise ValueError("s and o must have the same length")

    if len(s.columns) != len(o.columns):
        raise ValueError("s and o must have the same number of columns")

    features = list(o.columns)

    o = o.rename({col: f"o_{col}" for col in o.columns})
    s = s.rename(
        {s_col: o_col.replace("o_", "s_") for s_col, o_col in zip(s.columns, o.columns)}
    )

    full_df = pl.concat([s, o], how="horizontal")

    ioa = full_df.select(
        [
            (
                1
                - (((pl.col(f"o_{col}") - pl.col(f"s_{col}")) ** 2).sum())
                / (
                    (
                        (pl.col(f"s_{col}") - pl.col(f"o_{col}").mean()).abs()
                        + (pl.col(f"o_{col}") - pl.col(f"o_{col}").mean()).abs()
                    )
                    ** 2
                ).sum()
            ).alias(f"ioa_{col}")
            for col in features
        ]
    )

    return ioa


def mae(s: pl.DataFrame, o: pl.DataFrame) -> pl.DataFrame:
    """
    Mean Absolute Error

    Args:
        s: simulated
        o: observed

    Returns:
        mae: mean absolute error
    """
    if s.height != o.height:
        raise ValueError("s and o must have the same length")

    if len(s.columns) != len(o.columns):
        raise ValueError("s and o must have the same number of columns")

    features = list(o.columns)

    o = o.rename({col: f"o_{col}" for col in o.columns})
    s = s.rename(
        {s_col: o_col.replace("o_", "s_") for s_col, o_col in zip(s.columns, o.columns)}
    )

    full_df = pl.concat([s, o], how="horizontal")

    mae = full_df.select(
        [
            (pl.col(f"s_{col}") - pl.col(f"o_{col}")).abs().mean().alias(f"mae_{col}")
            for col in features
        ]
    )

    return mae


def main():
    base_path = "data/05_inference_results"
    output_path = "data/06_metrics"

    test_path = pathlib.Path("data/02_processed/test")

    metrics = [("ioa", index_agreement), ("mae", mae)]

    y_features: dict[str, list[pl.DataFrame]] = {}
    y_features_names: dict[str, list[str]] = {}

    missing_ratios = [0, 20, 40, 60, 80]

    models = [
        "continuous_gnn",
        "ESN",
        "multivariate_gap_ahead_regressor",
        "multivariate_rnn_gnn",
        "univariate_monodes",
        "univariate_rnn",
        "univariate_rnn_with_time_encoding",
        "chronos",
        "continuous_gnn_with_time_encoding",
        "prophet",
        "nhits",
    ]

    for target_mask_path in (test_path / "missing_ratio_20" / "target_masks").glob(
        "*.parquet"
    ):
        target_mask = pl.read_parquet(target_mask_path)
        ts_name = target_mask_path.stem.replace("_target", "")
        ts_data = pl.read_parquet(test_path / f"{ts_name}.parquet")

        full_df = pl.concat([ts_data, target_mask], how="horizontal")
        y_features_names[ts_name] = [
            col for col in ts_data.columns if col != "datetime"
        ]
        y_features[ts_name] = [
            full_df.filter(pl.col(col)).select(
                [pl.col(feature) for feature in y_features_names[ts_name]]
            )
            for col in target_mask.columns
            if col != "datetime" and full_df.filter(pl.col(col)).height > 0
        ]

    for ts_name in y_features_names.keys():
        all_results: dict[str, dict[int, pl.DataFrame]] = {}

        for model_folder in pathlib.Path(base_path).glob("*"):
            if not model_folder.is_dir():
                continue

            model_name = model_folder.stem
            if model_name not in models:
                continue

            if model_name not in all_results:
                all_results[model_name] = {}

            ts = pathlib.Path(model_folder) / ts_name

            for missing_ratio_folder in ts.glob("*"):
                if not missing_ratio_folder.is_dir():
                    continue

                missing_ratio = int(missing_ratio_folder.stem.split("_")[-1])
                if missing_ratio not in missing_ratios:
                    continue

                results = []
                for result_file in missing_ratio_folder.glob("*.parquet"):
                    results.append(
                        (int(result_file.stem), pl.read_parquet(result_file))
                    )

                results.sort(key=lambda x: x[0])
                results = [result[1] for result in results if result[1].height > 0]

                if len(results) != len(y_features[ts_name]):
                    print(
                        f"Number of results ({len(results)}) for ts {ts_name} model {model_name} is different from number of target masks ({len(y_features[ts_name])})"
                    )
                    continue

                metrics_result = pl.concat(
                    [
                        pl.concat(
                            [metric_func(result, y) for _, metric_func in metrics],
                            how="horizontal",
                        )
                        for result, y in zip(results, y_features[ts_name])
                    ]
                )
                metrics_result = pl.concat(
                    [
                        pl.DataFrame(
                            {
                                "id": [
                                    f"{str(elem+1).rjust(3,"0")}"
                                    for elem in range(metrics_result.height)
                                ]
                            }
                        ),
                        metrics_result,
                    ],
                    how="horizontal",
                )
                all_results[model_name][missing_ratio] = metrics_result

        output_path_ = pathlib.Path(output_path)

        for feature in y_features_names[ts_name]:
            for metric_name, _ in metrics:
                for missing_ratio in missing_ratios:
                    output_file = (
                        output_path_
                        / ts_name
                        / f"missing_ratio_{missing_ratio}"
                        / f"{feature}_{metric_name}.parquet"
                    )

                    output_content = pl.concat(
                        [
                            results[missing_ratio].select(
                                [pl.col(f"{metric_name}_{feature}").alias(model_name)]
                            )
                            for model_name, results in all_results.items()
                            if missing_ratio in results
                        ],
                        how="horizontal",
                    )
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    output_content.write_parquet(output_file)
                    if metric_name == "ioa":
                        print(f"{ts_name} {feature} {metric_name}")
                        print(output_content.mean())


if __name__ == "__main__":
    main()
