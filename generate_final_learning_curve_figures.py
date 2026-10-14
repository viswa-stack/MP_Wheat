from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\MP_Wheat_Project")

INPUT_CONFIG = {
    "District-grouped": {
        "path": ROOT
        / "mp_wheat_project_pipeline"
        / "validation"
        / "district_grouped"
        / "deep_learning_models"
        / "spatial_validation"
        / "deep_learning_curves.csv",
        "validation_column": "inner_validation_mae_kg_ha",
    },
    "Rolling-origin": {
        "path": ROOT
        / "mp_wheat_project_pipeline"
        / "validation"
        / "rolling_origin"
        / "deep_learning_models"
        / "rolling_validation"
        / "deep_learning_curves.csv",
        "validation_column": "validation_mae_kg_ha",
    },
}

OUTPUT_DIR = (
    ROOT
    / "mp_wheat_project_pipeline"
    / "reporting"
    / "figures"
    / "learning_curves"
)

MODEL_ORDER = [
    "deep_learning__compact_mlp",
    "deep_learning__ft_transformer",
    "deep_learning__gru",
    "deep_learning__lstm",
    "deep_learning__simple_rnn",
]

MODEL_LABELS = {
    "deep_learning__compact_mlp": "Compact MLP",
    "deep_learning__ft_transformer": "FT-Transformer",
    "deep_learning__gru": "GRU",
    "deep_learning__lstm": "LSTM",
    "deep_learning__simple_rnn": "SimpleRNN",
}

MODEL_COLORS = {
    "deep_learning__compact_mlp": "#08306B",
    "deep_learning__ft_transformer": "#08519C",
    "deep_learning__gru": "#2171B5",
    "deep_learning__lstm": "#4292C6",
    "deep_learning__simple_rnn": "#6BAED6",
}

MODEL_LINESTYLES = {
    "deep_learning__compact_mlp": "-",
    "deep_learning__ft_transformer": "--",
    "deep_learning__gru": "-.",
    "deep_learning__lstm": ":",
    "deep_learning__simple_rnn": (0, (5, 2, 1, 2)),
}

MODEL_MARKERS = {
    "deep_learning__compact_mlp": "o",
    "deep_learning__ft_transformer": "s",
    "deep_learning__gru": "^",
    "deep_learning__lstm": "D",
    "deep_learning__simple_rnn": "P",
}

ACTIVE_RUN_FRACTION = 0.20
MINIMUM_ACTIVE_RUNS = 3
PNG_DPI = 400


def calculate_sha256(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_protocol_data(protocol_name, configuration):
    file_path = configuration["path"]
    validation_column = configuration["validation_column"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required learning-curve file was not found: {file_path}"
        )

    data = pd.read_csv(file_path)

    if protocol_name == "District-grouped":
        if "run_phase" not in data.columns:
            raise ValueError(
                "District-grouped learning curves are missing run_phase."
            )

        data = data.loc[
            data["run_phase"].eq("inner_epoch_selection")
        ].copy()

    required_columns = {
        "model_id",
        "run_id",
        "epoch",
        "training_mse_scaled",
        validation_column,
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            f"{protocol_name} is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data[
        [
            "model_id",
            "run_id",
            "epoch",
            "training_mse_scaled",
            validation_column,
        ]
    ].copy()

    data = data.rename(
        columns={
            validation_column: "validation_mae_kg_ha",
        }
    )

    data["model_id"] = data["model_id"].astype(str)
    data["run_id"] = data["run_id"].astype(str)

    for column in [
        "epoch",
        "training_mse_scaled",
        "validation_mae_kg_ha",
    ]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = data.dropna(
        subset=[
            "model_id",
            "run_id",
            "epoch",
        ]
    ).copy()

    data["epoch"] = data["epoch"].astype(int)

    unexpected_models = sorted(
        set(data["model_id"].unique()).difference(MODEL_ORDER)
    )

    if unexpected_models:
        raise ValueError(
            f"Unexpected models in {protocol_name}: "
            + ", ".join(unexpected_models)
        )

    duplicate_mask = data.duplicated(
        subset=[
            "model_id",
            "run_id",
            "epoch",
        ],
        keep=False,
    )

    duplicate_row_count = int(duplicate_mask.sum())

    if duplicate_row_count > 0:
        raise ValueError(
            f"{protocol_name} contains {duplicate_row_count} duplicate "
            "model-run-epoch rows."
        )

    data["protocol"] = protocol_name
    data["source_file"] = str(file_path)
    data["source_sha256"] = calculate_sha256(file_path)

    return data


def aggregate_metric(
    protocol_data,
    metric_column,
    metric_name,
):
    records = []

    for model_id in MODEL_ORDER:
        model_data = protocol_data.loc[
            protocol_data["model_id"].eq(model_id)
        ].copy()

        if model_data.empty:
            continue

        total_runs = int(model_data["run_id"].nunique())

        active_run_threshold = max(
            MINIMUM_ACTIVE_RUNS,
            int(
                math.ceil(
                    total_runs
                    * ACTIVE_RUN_FRACTION
                )
            ),
        )

        valid_metric_data = model_data.dropna(
            subset=[metric_column]
        ).copy()

        grouped = (
            valid_metric_data
            .groupby(
                [
                    "model_id",
                    "epoch",
                ],
                as_index=False,
            )
            .agg(
                active_runs=(
                    "run_id",
                    "nunique",
                ),
                median_value=(
                    metric_column,
                    "median",
                ),
                q1_value=(
                    metric_column,
                    lambda values: values.quantile(0.25),
                ),
                q3_value=(
                    metric_column,
                    lambda values: values.quantile(0.75),
                ),
                mean_value=(
                    metric_column,
                    "mean",
                ),
                standard_deviation=(
                    metric_column,
                    "std",
                ),
            )
        )

        grouped = grouped.loc[
            grouped["active_runs"].ge(
                active_run_threshold
            )
        ].copy()

        grouped["protocol"] = protocol_data[
            "protocol"
        ].iloc[0]

        grouped["metric"] = metric_name
        grouped["metric_column"] = metric_column
        grouped["model_label"] = MODEL_LABELS[model_id]
        grouped["total_runs"] = total_runs

        grouped[
            "minimum_active_run_threshold"
        ] = active_run_threshold

        grouped["active_run_fraction_required"] = (
            ACTIVE_RUN_FRACTION
        )

        records.append(grouped)

    if not records:
        raise ValueError(
            f"No valid rows were available for metric {metric_name}."
        )

    return pd.concat(
        records,
        ignore_index=True,
    )


def create_run_summary(protocol_data):
    summary = (
        protocol_data
        .groupby(
            "model_id",
            as_index=False,
        )
        .agg(
            unique_runs=(
                "run_id",
                "nunique",
            ),
            minimum_epoch=(
                "epoch",
                "min",
            ),
            maximum_epoch=(
                "epoch",
                "max",
            ),
            total_epoch_rows=(
                "epoch",
                "size",
            ),
            nonmissing_training_rows=(
                "training_mse_scaled",
                "count",
            ),
            nonmissing_validation_rows=(
                "validation_mae_kg_ha",
                "count",
            ),
        )
    )

    summary["protocol"] = protocol_data[
        "protocol"
    ].iloc[0]

    summary["model_label"] = summary[
        "model_id"
    ].map(MODEL_LABELS)

    summary["source_file"] = protocol_data[
        "source_file"
    ].iloc[0]

    summary["source_sha256"] = protocol_data[
        "source_sha256"
    ].iloc[0]

    summary["minimum_active_run_threshold"] = (
        summary["unique_runs"]
        .apply(
            lambda value: max(
                MINIMUM_ACTIVE_RUNS,
                int(
                    math.ceil(
                        value
                        * ACTIVE_RUN_FRACTION
                    )
                ),
            )
        )
    )

    return summary


def configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def plot_learning_curve(
    plot_data,
    protocol_name,
    metric_name,
    title,
    y_axis_label,
    output_stem,
):
    subset = plot_data.loc[
        plot_data["protocol"].eq(protocol_name)
        & plot_data["metric"].eq(metric_name)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No plot data found for {protocol_name}, {metric_name}."
        )

    figure, axis = plt.subplots(
        figsize=(8.8, 5.6)
    )

    plotted_models = 0

    for model_id in MODEL_ORDER:
        model_data = (
            subset.loc[
                subset["model_id"].eq(model_id)
            ]
            .sort_values("epoch")
            .copy()
        )

        if model_data.empty:
            continue

        x_values = model_data[
            "epoch"
        ].to_numpy(dtype=float)

        median_values = model_data[
            "median_value"
        ].to_numpy(dtype=float)

        q1_values = model_data[
            "q1_value"
        ].to_numpy(dtype=float)

        q3_values = model_data[
            "q3_value"
        ].to_numpy(dtype=float)

        marker_interval = max(
            1,
            len(model_data) // 12,
        )

        axis.plot(
            x_values,
            median_values,
            label=MODEL_LABELS[model_id],
            color=MODEL_COLORS[model_id],
            linestyle=MODEL_LINESTYLES[model_id],
            linewidth=2.1,
            marker=MODEL_MARKERS[model_id],
            markersize=4.2,
            markevery=marker_interval,
            markerfacecolor="white",
            markeredgewidth=0.9,
        )

        axis.fill_between(
            x_values,
            q1_values,
            q3_values,
            color=MODEL_COLORS[model_id],
            alpha=0.10,
            linewidth=0,
        )

        plotted_models += 1

    if plotted_models == 0:
        plt.close(figure)
        raise ValueError(
            f"No model curves were generated for {output_stem}."
        )

    axis.set_title(
        title,
        pad=12,
        fontweight="bold",
    )

    axis.set_xlabel("Training epoch")
    axis.set_ylabel(y_axis_label)

    axis.grid(
        visible=True,
        which="major",
        axis="both",
        linewidth=0.55,
        alpha=0.28,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        title="Deep-learning model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
    )

    active_thresholds = sorted(
        subset[
            "minimum_active_run_threshold"
        ].unique()
    )

    total_runs = sorted(
        subset["total_runs"].unique()
    )

    threshold_text = ", ".join(
        str(int(value))
        for value in active_thresholds
    )

    total_run_text = ", ".join(
        str(int(value))
        for value in total_runs
    )

    figure.text(
        0.01,
        0.01,
        (
            "Lines show the epoch-wise median; shaded regions show the "
            "interquartile range across active runs. "
            f"Total runs per model: {total_run_text}. "
            f"Minimum active runs retained per epoch: {threshold_text}."
        ),
        ha="left",
        va="bottom",
        fontsize=8.2,
    )

    figure.tight_layout(
        rect=[
            0,
            0.075,
            0.82,
            1,
        ]
    )

    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"

    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(figure)

    return [
        png_path,
        pdf_path,
    ]


def build_output_manifest(file_paths):
    manifest_rows = []

    for file_path in sorted(
        file_paths,
        key=lambda value: value.name,
    ):
        manifest_rows.append(
            {
                "file_name": file_path.name,
                "full_path": str(file_path),
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": file_path.stat().st_size,
                "sha256": calculate_sha256(file_path),
            }
        )

    return pd.DataFrame(manifest_rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_plot_style()

    protocol_frames = []
    run_summaries = []
    source_records = []

    for protocol_name, configuration in INPUT_CONFIG.items():
        protocol_data = load_protocol_data(
            protocol_name,
            configuration,
        )

        protocol_frames.append(protocol_data)

        run_summaries.append(
            create_run_summary(protocol_data)
        )

        source_records.append(
            {
                "protocol": protocol_name,
                "source_file": str(
                    configuration["path"]
                ),
                "source_file_size_bytes": (
                    configuration["path"].stat().st_size
                ),
                "source_sha256": calculate_sha256(
                    configuration["path"]
                ),
                "validation_metric_column": (
                    configuration[
                        "validation_column"
                    ]
                ),
            }
        )

    all_protocol_data = pd.concat(
        protocol_frames,
        ignore_index=True,
    )

    run_summary = pd.concat(
        run_summaries,
        ignore_index=True,
    )

    aggregated_frames = []

    for protocol_name in INPUT_CONFIG:
        protocol_data = all_protocol_data.loc[
            all_protocol_data[
                "protocol"
            ].eq(protocol_name)
        ].copy()

        aggregated_frames.append(
            aggregate_metric(
                protocol_data,
                metric_column="validation_mae_kg_ha",
                metric_name="validation_mae_kg_ha",
            )
        )

        aggregated_frames.append(
            aggregate_metric(
                protocol_data,
                metric_column="training_mse_scaled",
                metric_name="training_mse_scaled",
            )
        )

    plot_data = pd.concat(
        aggregated_frames,
        ignore_index=True,
    )

    generated_figure_paths = []

    generated_figure_paths.extend(
        plot_learning_curve(
            plot_data=plot_data,
            protocol_name="District-grouped",
            metric_name="validation_mae_kg_ha",
            title=(
                "District-Grouped Validation Learning Curves"
            ),
            y_axis_label=(
                "Inner-validation MAE (kg ha$^{-1}$)"
            ),
            output_stem=(
                "Figure_LC1_District_Grouped_Validation_MAE"
            ),
        )
    )

    generated_figure_paths.extend(
        plot_learning_curve(
            plot_data=plot_data,
            protocol_name="Rolling-origin",
            metric_name="validation_mae_kg_ha",
            title=(
                "Rolling-Origin Validation Learning Curves"
            ),
            y_axis_label=(
                "Validation MAE (kg ha$^{-1}$)"
            ),
            output_stem=(
                "Figure_LC2_Rolling_Origin_Validation_MAE"
            ),
        )
    )

    generated_figure_paths.extend(
        plot_learning_curve(
            plot_data=plot_data,
            protocol_name="District-grouped",
            metric_name="training_mse_scaled",
            title=(
                "District-Grouped Training-Loss Curves"
            ),
            y_axis_label="Training loss (scaled MSE)",
            output_stem=(
                "Figure_LC3_District_Grouped_Training_Loss"
            ),
        )
    )

    generated_figure_paths.extend(
        plot_learning_curve(
            plot_data=plot_data,
            protocol_name="Rolling-origin",
            metric_name="training_mse_scaled",
            title=(
                "Rolling-Origin Training-Loss Curves"
            ),
            y_axis_label="Training loss (scaled MSE)",
            output_stem=(
                "Figure_LC4_Rolling_Origin_Training_Loss"
            ),
        )
    )

    plot_data_path = (
        OUTPUT_DIR
        / "learning_curve_plot_data.csv"
    )

    run_summary_path = (
        OUTPUT_DIR
        / "learning_curve_run_summary.csv"
    )

    source_manifest_path = (
        OUTPUT_DIR
        / "learning_curve_source_manifest.csv"
    )

    caption_path = (
        OUTPUT_DIR
        / "learning_curve_figure_captions.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / "learning_curve_generation_metadata.json"
    )

    plot_data.to_csv(
        plot_data_path,
        index=False,
        encoding="utf-8-sig",
    )

    run_summary.to_csv(
        run_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        source_records
    ).to_csv(
        source_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    captions = pd.DataFrame(
        [
            {
                "figure_id": "Figure LC1",
                "file_stem": (
                    "Figure_LC1_District_Grouped_Validation_MAE"
                ),
                "caption": (
                    "Epoch-wise district-grouped inner-validation MAE "
                    "for Compact MLP, FT-Transformer, GRU, LSTM and "
                    "SimpleRNN. Lines represent median performance and "
                    "shaded bands represent the interquartile range "
                    "across active runs."
                ),
            },
            {
                "figure_id": "Figure LC2",
                "file_stem": (
                    "Figure_LC2_Rolling_Origin_Validation_MAE"
                ),
                "caption": (
                    "Epoch-wise rolling-origin validation MAE for "
                    "Compact MLP, FT-Transformer, GRU, LSTM and "
                    "SimpleRNN. Lines represent median performance and "
                    "shaded bands represent the interquartile range "
                    "across active runs."
                ),
            },
            {
                "figure_id": "Figure LC3",
                "file_stem": (
                    "Figure_LC3_District_Grouped_Training_Loss"
                ),
                "caption": (
                    "Epoch-wise scaled training MSE under "
                    "district-grouped validation. Lines represent the "
                    "median and shaded bands represent the "
                    "interquartile range across active runs."
                ),
            },
            {
                "figure_id": "Figure LC4",
                "file_stem": (
                    "Figure_LC4_Rolling_Origin_Training_Loss"
                ),
                "caption": (
                    "Epoch-wise scaled training MSE under "
                    "rolling-origin validation. Lines represent the "
                    "median and shaded bands represent the "
                    "interquartile range across active runs."
                ),
            },
        ]
    )

    captions.to_csv(
        caption_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "models_trained_by_this_script": False,
        "source_data_modified": False,
        "aggregation": {
            "central_tendency": "median",
            "uncertainty_band": "interquartile_range",
            "active_run_fraction_required": (
                ACTIVE_RUN_FRACTION
            ),
            "minimum_active_runs": (
                MINIMUM_ACTIVE_RUNS
            ),
        },
        "protocols": list(INPUT_CONFIG.keys()),
        "models": MODEL_ORDER,
        "output_directory": str(OUTPUT_DIR),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file_handle:
        json.dump(
            metadata,
            file_handle,
            indent=2,
        )

    supporting_paths = [
        plot_data_path,
        run_summary_path,
        source_manifest_path,
        caption_path,
        metadata_path,
    ]

    manifest_input_paths = (
        generated_figure_paths
        + supporting_paths
    )

    output_manifest = build_output_manifest(
        manifest_input_paths
    )

    output_manifest_path = (
        OUTPUT_DIR
        / "learning_curve_output_sha256_manifest.csv"
    )

    output_manifest.to_csv(
        output_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 88)
    print("FINAL LEARNING-CURVE FIGURE GENERATION COMPLETE")
    print("=" * 88)
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        "Model training performed by this script: False"
    )
    print(
        "Original learning-curve CSV files modified: False"
    )
    print(
        f"Figure files generated: {len(generated_figure_paths)}"
    )
    print(
        f"Aggregated plot-data rows: {len(plot_data):,}"
    )
    print(
        f"Run-summary rows: {len(run_summary):,}"
    )
    print("")
    print("GENERATED FILES")

    for file_path in (
        generated_figure_paths
        + supporting_paths
        + [output_manifest_path]
    ):
        print(
            f"- {file_path.name} "
            f"({file_path.stat().st_size:,} bytes)"
        )


if __name__ == "__main__":
    main()
