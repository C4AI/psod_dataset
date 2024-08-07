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
        "ESN": "Echo State Network",
        "gap_ahead_regressor_0": "Time Gap Ahead Encoding - 0",
        "gap_ahead_regressor_20": "Time Gap Ahead Encoding - 20",
        "univariate_monodes": "Univariate MoNODE",
        "univariate_rnn": "Univariate RNN",
        "univariate_rnn_time_encoding": "Univariate RNN w/ Time Encoding",
    }

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

    plot_pos = {
        ("current_praticagem", "cross_shore_current"): 0,
        ("waves_palmas", "ws"): 1,
        ("waves_palmas", "hs"): 2,
        ("waves_palmas", "tp"): 3,
    }

    source_folder_ = pathlib.Path(source_folder)

    ts_names = [ts_name.stem for ts_name in source_folder_.glob("*")]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))

    for ts_name in ts_names:
        ts_folder = source_folder_ / ts_name

        for metric_file in ts_folder.glob("*.parquet"):
            metric_name = metric_file.stem.split("_")[-1]
            feature_name = "_".join(metric_file.stem.split("_")[:-1])
            if metric_name != "ioa":
                continue
            ax = axs.flatten()[plot_pos[(ts_name, feature_name)]]
            metric_df = pl.read_parquet(metric_file)

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            for model_name in metric_df.columns:

                color = colors.pop(0)
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
        loc="upper left",
        bbox_to_anchor=(0.06, 0.95),
        fontsize=13,
    )
    # layout tight
    plt.tight_layout(pad=1.5)
    fig.savefig(out_folder_ / "consolidated_IoA_20.pdf")

    print("Done")


if __name__ == "__main__":
    main()
