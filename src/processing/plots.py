import datetime
import pathlib
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter


def main():
    source_folder = "data/06_metrics"

    out_folder = "data/07_plots"
    out_folder_ = pathlib.Path(out_folder)

    metrics_dict = {
        "ioa": "Index of Agreement",
        "mae": "Mean Absolute Error",
        "rmse": "Root Mean Squared Error",
    }

    model_dict = {
        "ESN": "LiESN-D",
        "multivariate_gap_ahead_regressor": "Gap-Ahead",
        "multivariate_rnn_gnn": "GRU+GNN",
        "univariate_monodes": "MoNODE",
        "univariate_rnn": "GRU",
        "univariate_rnn_with_time_encoding": "GRU+Time Encoding",
        "continuous_gnn": "CGNN",
        "chronos": "Chronos",
        "continuous_gnn_with_time_encoding": "CGNN+Time Encoding",
        "nhits": "N-Hits",
        "prophet": "Prophet",
    }

    plot_order = [
        "chronos",
        "continuous_gnn",
        "continuous_gnn_with_time_encoding",
        "multivariate_gap_ahead_regressor",
        "univariate_rnn",
        "multivariate_rnn_gnn",
        "univariate_rnn_with_time_encoding",
        "ESN",
        "univariate_monodes",
        "nhits",
        "prophet",
    ]
    ts_dict = {
        "current_praticagem": "Water Current - Praticagem",
        "waves_palmas": "Waves - Palmas",
    }

    feature_dict = {
        "cross_shore_current": "Cross-Shore Speed",
        "hs": "Significant Height",
        "tp": "Peak Period",
        "ws": "Cross-Shore Speed",
    }

    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-green
        "#17becf",  # Teal
        "#aec7e8",  # Light Blue
    ]

    color_dict = {model_name: colors.pop(0) for model_name in model_dict.keys()}

    source_folder_ = pathlib.Path(source_folder)

    ts_names = [ts_name.stem for ts_name in source_folder_.glob("*")]
    missing_ratios = [0, 20, 40, 60, 80]

    # plot_kde(
    #     ts_names,
    #     source_folder_,
    #     out_folder_,
    #     metrics_dict,
    #     model_dict,
    #     ts_dict,
    #     feature_dict,
    #     missing_ratios,
    #     color_dict,
    #     plot_order,
    # )

    # plot_degradation_curve(
    #     ts_names,
    #     source_folder_,
    #     out_folder_,
    #     metrics_dict,
    #     model_dict,
    #     ts_dict,
    #     feature_dict,
    #     missing_ratios,
    #     color_dict,
    #     plot_order,
    # )

    print_tables(source_folder_, model_dict, missing_ratios, plot_order, ts_dict)

    orig_src = pathlib.Path("data/02_processed")
    test_sources = {
        f.stem: pl.read_parquet(f) for f in (orig_src / "test").glob("*.parquet")
    }
    train_sources = {
        f.stem: pl.read_parquet(f) for f in (orig_src / "train").glob("*.parquet")
    }

    ts_name_dict = {
        "cross_shore_current": (
            "Cross-Shore Water Current Speed",
            "Current Speed (m/s)",
        ),
        "hs": ("Wave Height", "Wave Height (m)"),
        "tp": ("Wave Period", "Wave Period (s)"),
        "ws": ("Wave Speed", "Wave Speed (m/s)"),
        "ssh": ("Sea Surface Height", "Sea Surface Height (m)"),
        "vx": ("Wind Speed to East", "Wind Speed to East (m/s)"),
        "vy": ("Wind Speed to North", "Wind Speed to North (m/s)"),
        "astronomical_tide": ("Astronomical Tide", "Astronomical Tide (m)"),
    }
    for ts_name in test_sources.keys():
        df = pl.concat([train_sources[ts_name], test_sources[ts_name]]).sort("datetime")
        for feat in df.columns:
            if feat == "datetime":
                continue

            fig, ax = plt.subplots(1, figsize=(7, 5))
            name, label = ts_name_dict[feat]
            sns.kdeplot(
                data=df[feat].to_numpy(),
                color="blue",
                label=name,
                # cumulative=True,
                # common_norm=False,
                common_grid=True,
                ax=ax,
                # clip=(0, 1 if metric_name == "ioa" else None),
            )
            # add vertical line
            ax.axvline(
                x=df[feat].median(),
                color="red",
                linestyle="--",
                label=f"Median",
            )
            ax.legend()
            ax.set_xlabel(label, fontsize=15)
            ax.set_ylabel("Probability Density", fontsize=15)
            fig.tight_layout()
            fig.savefig(out_folder_ / f"{ts_name}_{feat}_kde.pdf")

            fig, ax = plt.subplots(1, figsize=(7, 5))

            base_date = datetime.datetime(
                2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc
            )
            late_date = datetime.datetime(
                2021, 1, 8, 0, 0, 0, tzinfo=datetime.timezone.utc
            )

            lbound = df["datetime"].search_sorted(base_date)
            ubound = df["datetime"].search_sorted(late_date)

            ax.plot(
                df["datetime"][lbound:ubound],
                df[feat][lbound:ubound],
                label=name,
                color="blue",
            )
            ax.legend()
            ax.set_xlabel("Date", fontsize=15)
            # rotate x labels
            plt.xticks(rotation=45)
            ax.set_ylabel(ts_name_dict[feat][1], fontsize=15)
            fig.tight_layout()
            fig.savefig(out_folder_ / f"{ts_name}_{feat}_time_series.pdf")

    print("Done")


def print_tables(source_folder_, model_dict, missing_ratios, plot_order, ts_dict):

    results = pl.DataFrame({"Model": [model_dict[model] for model in plot_order]})

    for ts_name in ts_dict.keys():
        for ratio in missing_ratios:
            ts_folder = source_folder_ / ts_name / f"missing_ratio_{ratio}"
            for metric_file in ts_folder.glob("*.parquet"):

                metric_name = metric_file.stem.split("_")[-1]
                feature_name = "_".join(metric_file.stem.split("_")[:-1])
                if feature_name in ["cross_shore_current", "tp", "hs"]:
                    continue
                if metric_name == "ioa":
                    continue
                metric_df = pl.read_parquet(metric_file)
                col = []
                for model_name in plot_order:
                    if model_name not in model_dict:
                        continue

                    values = metric_df.select(
                        [
                            pl.when(pl.col(model_name).is_not_nan())
                            .then(pl.col(model_name))
                            .otherwise(0)
                            .alias(model_name)
                        ]
                    )[model_name]

                    model_info = f"{values.mean():.2f} ± {values.std():.2f}"
                    col.append(model_info)

                results = results.with_columns(pl.Series(f"{ratio:02}%", col))

        print(results.to_pandas().to_latex(index=False))


def plot_degradation_curve(
    ts_names,
    source_folder_,
    out_folder_,
    metrics_dict,
    model_dict,
    ts_dict,
    feature_dict,
    missing_ratios,
    color_dict,
    plot_order,
):
    plot_pos = {
        "cross_shore_current": 0,
        "ws": 1,
        "hs": 2,
        "tp": 3,
    }
    all_results = {}

    models = [
        "univariate_monodes",
        "multivariate_gap_ahead_regressor",
        "multivariate_rnn_gnn",
        "univariate_rnn",
        "univariate_rnn_with_time_encoding",
        "ESN",
        "continuous_gnn",
        "continuous_gnn_with_time_encoding",
        "chronos",
        "nhits",
        "prophet",
    ]
    for missing_ratio in missing_ratios:

        for ts_name in ts_names:
            ts_folder = source_folder_ / ts_name / f"missing_ratio_{missing_ratio}"

            for metric_file in ts_folder.glob("*.parquet"):
                metric_name = metric_file.stem.split("_")[-1]
                feature_name = "_".join(metric_file.stem.split("_")[:-1])
                if metric_name != "ioa":
                    continue
                if (ts_name, feature_name) not in all_results:
                    all_results[(ts_name, feature_name)] = {}
                metric_df = pl.read_parquet(metric_file)

                for model_name in plot_order:
                    if model_name not in models:
                        continue
                    if model_name not in all_results[(ts_name, feature_name)]:
                        all_results[(ts_name, feature_name)][model_name] = {
                            "missing_ratio": [],
                            "ioa": [],
                            "color": color_dict[model_name],
                        }

                    all_results[(ts_name, feature_name)][model_name][
                        "missing_ratio"
                    ].append(missing_ratio)
                    all_results[(ts_name, feature_name)][model_name]["ioa"].append(
                        metric_df.filter(metric_df[model_name].is_not_nan())[
                            model_name
                        ].mean()
                    )

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    for (ts_name, feature_name), feature_results in all_results.items():
        for model_name, model_results in feature_results.items():
            ax = axs.flatten()[plot_pos[feature_name]]
            ax.plot(
                model_results["missing_ratio"],
                model_results["ioa"],
                label=model_dict[model_name],
                color=model_results["color"],
            )

        ax.set_xlabel(
            "Missing Ratio (%)",
            fontsize=15,
        )
        ax.set_ylabel("Index of Agreement", fontsize=15)

        # increase x and y tick font
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # set title and increase size
        ax.title.set_text(f"{ts_dict[ts_name]} | {feature_dict[feature_name]}")
        ax.title.set_fontsize(16)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        fontsize=13,
        ncols=4,
        bbox_to_anchor=(
            0.5,
            0.01,
        ),
    )

    plt.tight_layout(pad=1.5, rect=[0, 0.1, 1, 1])
    fig.savefig(out_folder_ / f"degradation_plot_ioa.pdf")


def plot_kde(
    ts_names,
    source_folder_,
    out_folder_,
    metrics_dict,
    model_dict,
    ts_dict,
    feature_dict,
    missing_ratios,
    color_dict,
    plot_order,
):
    plot_pos = {
        ("current_praticagem", "cross_shore_current"): 0,
        ("waves_palmas", "ws"): 1,
        ("waves_palmas", "hs"): 2,
        ("waves_palmas", "tp"): 3,
    }
    for missing_ratio in missing_ratios:
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))

        for ts_name in ts_names:
            ts_folder = source_folder_ / ts_name / f"missing_ratio_{missing_ratio}"

            for metric_file in ts_folder.glob("*.parquet"):
                metric_name = metric_file.stem.split("_")[-1]
                feature_name = "_".join(metric_file.stem.split("_")[:-1])
                if metric_name != "ioa":
                    continue
                ax = axs.flatten()[plot_pos[(ts_name, feature_name)]]
                metric_df = pl.read_parquet(metric_file)

                for model_name in plot_order:
                    if model_name not in model_dict:
                        continue
                    color = color_dict[model_name]
                    sns.kdeplot(
                        data=metric_df[model_name].to_numpy(),
                        color=color,
                        label=model_dict[model_name],
                        # cumulative=True,
                        # common_norm=False,
                        common_grid=True,
                        ax=ax,
                        clip=(0, 1 if metric_name == "ioa" else None),
                    )
                    # add vertical line
                    ax.axvline(
                        x=metric_df[model_name].median(),
                        color=color,
                        linestyle="--",
                        # label=f"{model_name} mean",
                    )

                # labels with bold font
                ax.set_xlabel(
                    metrics_dict[metric_name],
                    fontsize=15,
                )
                ax.set_ylabel("Probability Density", fontsize=15)

                # increase x and y tick font
                ax.tick_params(axis="x", labelsize=14)
                ax.tick_params(axis="y", labelsize=14)
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

                # set title and increase size
                ax.title.set_text(f"{ts_dict[ts_name]} | {feature_dict[feature_name]}")
                ax.title.set_fontsize(16)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            fontsize=13,
            ncols=4,
            bbox_to_anchor=(
                0.5,
                0.01,
            ),
        )

        plt.tight_layout(pad=1.5, rect=[0, 0.1, 1, 1])
        fig.savefig(out_folder_ / f"consolidated_ioa_{missing_ratio}.pdf")


if __name__ == "__main__":
    main()
