from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:\MP_Wheat_Project")
PIPELINE = ROOT / "mp_wheat_project_pipeline"

PREDICTION_PATH = (
    PIPELINE
    / "validation"
    / "ablation"
    / "unified_synthesis_and_robustness_protocol"
    / "development_set_robustness"
    / "robustness_enriched_predictions.csv"
)

THRESHOLD_PATH = (
    PIPELINE
    / "exploratory_analysis"
    / "primary_tabular"
    / "primary_target_outlier_split_summary.csv"
)

OUTPUT_DIR = (
    PIPELINE
    / "reporting"
    / "robustness"
    / "outlier_sensitivity"
)

SCRIPT_PATH = Path(__file__).resolve()

MODEL_ORDER = [
    "baseline__prior_2year_moving_average",
    "linear__elastic_net",
    "deep_learning__ft_transformer",
    "deep_learning__lstm",
    "deep_learning__simple_rnn",
]

MODEL_NAMES = {
    "baseline__prior_2year_moving_average":
        "Prior 2-year moving average",
    "linear__elastic_net":
        "Elastic Net",
    "deep_learning__ft_transformer":
        "FT-Transformer",
    "deep_learning__lstm":
        "LSTM",
    "deep_learning__simple_rnn":
        "Simple RNN",
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def save_figure(figure, stem):
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"

    figure.savefig(
        png_path,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return [png_path, pdf_path]


def main():
    require_file(PREDICTION_PATH)
    require_file(THRESHOLD_PATH)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.read_csv(
        PREDICTION_PATH
    )

    thresholds = pd.read_csv(
        THRESHOLD_PATH
    )

    required_prediction_columns = {
        "protocol",
        "model_id",
        "district_id",
        "crop_year_start",
        "observed_yield_kg_ha",
        "absolute_error_kg_ha",
        "target_yield_tail",
    }

    missing_columns = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction table is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    train_threshold = thresholds.loc[
        thresholds["evaluated_split"]
        .astype(str)
        .eq("train")
    ]

    if len(train_threshold) != 1:
        raise ValueError(
            "Expected exactly one training threshold row."
        )

    lower_threshold = float(
        train_threshold.iloc[0][
            "lower_threshold"
        ]
    )

    upper_threshold = float(
        train_threshold.iloc[0][
            "upper_threshold"
        ]
    )

    development_protocols = {
        "district_grouped",
        "rolling_origin",
    }

    actual_protocols = set(
        predictions["protocol"].astype(str)
    )

    if actual_protocols != development_protocols:
        raise RuntimeError(
            "Unexpected protocols in development predictions: "
            + str(sorted(actual_protocols))
        )

    observed = pd.to_numeric(
        predictions["observed_yield_kg_ha"],
        errors="coerce",
    )

    errors = pd.to_numeric(
        predictions["absolute_error_kg_ha"],
        errors="coerce",
    )

    if observed.isna().any() or errors.isna().any():
        raise ValueError(
            "Observed yields or absolute errors contain missing values."
        )

    predictions = predictions.copy()

    predictions[
        "target_iqr_outlier_class"
    ] = np.select(
        [
            observed < lower_threshold,
            observed > upper_threshold,
        ],
        [
            "low_iqr_outlier",
            "high_iqr_outlier",
        ],
        default="non_outlier",
    )

    predictions[
        "model_display_name"
    ] = predictions[
        "model_id"
    ].map(MODEL_NAMES)

    if predictions[
        "model_display_name"
    ].isna().any():
        unknown = sorted(
            predictions.loc[
                predictions[
                    "model_display_name"
                ].isna(),
                "model_id",
            ]
            .astype(str)
            .unique()
        )

        raise ValueError(
            "Unexpected model IDs: "
            + ", ".join(unknown)
        )

    unique_observation_columns = [
        "protocol",
        "district_id",
        "crop_year_start",
        "observed_yield_kg_ha",
        "target_yield_tail",
        "target_iqr_outlier_class",
    ]

    unique_observations = (
        predictions[
            unique_observation_columns
        ]
        .drop_duplicates()
        .sort_values(
            [
                "protocol",
                "crop_year_start",
                "district_id",
            ]
        )
        .reset_index(drop=True)
    )

    duplicated_observation_count = int(
        unique_observations.duplicated(
            [
                "protocol",
                "district_id",
                "crop_year_start",
            ]
        ).sum()
    )

    if duplicated_observation_count != 0:
        raise RuntimeError(
            "Duplicate development observations were detected."
        )

    iqr_counts = (
        unique_observations
        .groupby(
            [
                "protocol",
                "target_iqr_outlier_class",
            ],
            as_index=False,
        )
        .agg(
            unique_observation_count=(
                "district_id",
                "count",
            ),
            minimum_observed_yield_kg_ha=(
                "observed_yield_kg_ha",
                "min",
            ),
            maximum_observed_yield_kg_ha=(
                "observed_yield_kg_ha",
                "max",
            ),
        )
    )

    tail_counts = (
        unique_observations
        .groupby(
            [
                "protocol",
                "target_yield_tail",
            ],
            as_index=False,
        )
        .agg(
            unique_observation_count=(
                "district_id",
                "count",
            ),
            minimum_observed_yield_kg_ha=(
                "observed_yield_kg_ha",
                "min",
            ),
            maximum_observed_yield_kg_ha=(
                "observed_yield_kg_ha",
                "max",
            ),
        )
    )

    iqr_metrics = (
        predictions
        .groupby(
            [
                "protocol",
                "model_id",
                "model_display_name",
                "target_iqr_outlier_class",
            ],
            as_index=False,
        )
        .agg(
            observation_count=(
                "absolute_error_kg_ha",
                "count",
            ),
            mae_kg_ha=(
                "absolute_error_kg_ha",
                "mean",
            ),
            median_absolute_error_kg_ha=(
                "absolute_error_kg_ha",
                "median",
            ),
            maximum_absolute_error_kg_ha=(
                "absolute_error_kg_ha",
                "max",
            ),
        )
    )

    tail_metrics = (
        predictions
        .groupby(
            [
                "protocol",
                "model_id",
                "model_display_name",
                "target_yield_tail",
            ],
            as_index=False,
        )
        .agg(
            observation_count=(
                "absolute_error_kg_ha",
                "count",
            ),
            mae_kg_ha=(
                "absolute_error_kg_ha",
                "mean",
            ),
            median_absolute_error_kg_ha=(
                "absolute_error_kg_ha",
                "median",
            ),
            maximum_absolute_error_kg_ha=(
                "absolute_error_kg_ha",
                "max",
            ),
        )
    )

    iqr_pivot = iqr_metrics.pivot_table(
        index=[
            "protocol",
            "model_id",
            "model_display_name",
        ],
        columns="target_iqr_outlier_class",
        values="mae_kg_ha",
        aggfunc="first",
    ).reset_index()

    for column in [
        "low_iqr_outlier",
        "high_iqr_outlier",
        "non_outlier",
    ]:
        if column not in iqr_pivot.columns:
            iqr_pivot[column] = np.nan

    iqr_pivot[
        "high_outlier_minus_non_outlier_mae_kg_ha"
    ] = (
        iqr_pivot["high_iqr_outlier"]
        - iqr_pivot["non_outlier"]
    )

    iqr_pivot[
        "high_outlier_to_non_outlier_mae_ratio"
    ] = (
        iqr_pivot["high_iqr_outlier"]
        / iqr_pivot["non_outlier"]
    )

    tail_pivot = tail_metrics.pivot_table(
        index=[
            "protocol",
            "model_id",
            "model_display_name",
        ],
        columns="target_yield_tail",
        values="mae_kg_ha",
        aggfunc="first",
    ).reset_index()

    for column in [
        "bottom_decile",
        "central_eighty_percent",
        "top_decile",
    ]:
        if column not in tail_pivot.columns:
            tail_pivot[column] = np.nan

    tail_pivot[
        "top_decile_minus_central_mae_kg_ha"
    ] = (
        tail_pivot["top_decile"]
        - tail_pivot["central_eighty_percent"]
    )

    tail_pivot[
        "top_decile_to_central_mae_ratio"
    ] = (
        tail_pivot["top_decile"]
        / tail_pivot["central_eighty_percent"]
    )

    order_mapping = {
        model_id: index
        for index, model_id in enumerate(
            MODEL_ORDER,
            start=1,
        )
    }

    for frame in [
        iqr_metrics,
        tail_metrics,
        iqr_pivot,
        tail_pivot,
    ]:
        frame[
            "model_order"
        ] = frame[
            "model_id"
        ].map(order_mapping)

    iqr_metrics = iqr_metrics.sort_values(
        [
            "protocol",
            "model_order",
            "target_iqr_outlier_class",
        ]
    )

    tail_metrics = tail_metrics.sort_values(
        [
            "protocol",
            "model_order",
            "target_yield_tail",
        ]
    )

    iqr_pivot = iqr_pivot.sort_values(
        [
            "protocol",
            "model_order",
        ]
    )

    tail_pivot = tail_pivot.sort_values(
        [
            "protocol",
            "model_order",
        ]
    )

    paths = []

    output_tables = {
        "outlier_sensitivity_unique_observations.csv":
            unique_observations,
        "outlier_sensitivity_iqr_counts.csv":
            iqr_counts,
        "outlier_sensitivity_tail_counts.csv":
            tail_counts,
        "outlier_sensitivity_iqr_model_metrics.csv":
            iqr_metrics,
        "outlier_sensitivity_tail_model_metrics.csv":
            tail_metrics,
        "outlier_sensitivity_iqr_contrasts.csv":
            iqr_pivot,
        "outlier_sensitivity_tail_contrasts.csv":
            tail_pivot,
    }

    for file_name, frame in output_tables.items():
        path = OUTPUT_DIR / file_name

        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        paths.append(path)

    figure = plt.figure(
        figsize=(11.5, 6.3)
    )

    axis = figure.add_subplot(111)

    plot_data = iqr_metrics.loc[
        iqr_metrics[
            "target_iqr_outlier_class"
        ].isin(
            [
                "non_outlier",
                "high_iqr_outlier",
            ]
        )
    ].copy()

    plot_data[
        "category"
    ] = plot_data[
        "target_iqr_outlier_class"
    ].map(
        {
            "non_outlier": "Non-outlier",
            "high_iqr_outlier":
                "High IQR outlier",
        }
    )

    protocols = [
        "district_grouped",
        "rolling_origin",
    ]

    model_names = [
        MODEL_NAMES[model_id]
        for model_id in MODEL_ORDER
    ]

    x = np.arange(
        len(model_names)
    )

    width = 0.19
    offsets = [
        -1.5 * width,
        -0.5 * width,
        0.5 * width,
        1.5 * width,
    ]

    combinations = [
        (
            "district_grouped",
            "Non-outlier",
        ),
        (
            "district_grouped",
            "High IQR outlier",
        ),
        (
            "rolling_origin",
            "Non-outlier",
        ),
        (
            "rolling_origin",
            "High IQR outlier",
        ),
    ]

    for offset, combination in zip(
        offsets,
        combinations,
    ):
        protocol, category = combination

        subset = plot_data.loc[
            plot_data[
                "protocol"
            ].eq(protocol)
            & plot_data[
                "category"
            ].eq(category)
        ].set_index(
            "model_display_name"
        )

        values = [
            subset.loc[
                model_name,
                "mae_kg_ha",
            ]
            for model_name in model_names
        ]

        label = (
            protocol.replace(
                "_",
                " ",
            ).title()
            + " — "
            + category
        )

        axis.bar(
            x + offset,
            values,
            width,
            label=label,
        )

    axis.set_xticks(x)
    axis.set_xticklabels(
        model_names,
        rotation=20,
        ha="right",
    )

    axis.set_ylabel(
        "Mean absolute error (kg/ha)"
    )

    axis.set_title(
        "Model Error on Training-Defined Yield Outliers",
        fontweight="bold",
        pad=12,
    )

    axis.legend(
        frameon=False,
        ncol=2,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    paths.extend(
        save_figure(
            figure,
            "Figure_Outlier_Sensitivity_IQR",
        )
    )

    figure = plt.figure(
        figsize=(11.5, 6.3)
    )

    axis = figure.add_subplot(111)

    categories = [
        "bottom_decile",
        "central_eighty_percent",
        "top_decile",
    ]

    category_names = {
        "bottom_decile": "Bottom decile",
        "central_eighty_percent":
            "Central 80%",
        "top_decile": "Top decile",
    }

    selected = tail_metrics.loc[
        tail_metrics["protocol"]
        .eq("rolling_origin")
    ].copy()

    width = 0.25
    offsets = [
        -width,
        0,
        width,
    ]

    for offset, category in zip(
        offsets,
        categories,
    ):
        subset = selected.loc[
            selected[
                "target_yield_tail"
            ].eq(category)
        ].set_index(
            "model_display_name"
        )

        values = [
            subset.loc[
                model_name,
                "mae_kg_ha",
            ]
            for model_name in model_names
        ]

        axis.bar(
            x + offset,
            values,
            width,
            label=category_names[category],
        )

    axis.set_xticks(x)
    axis.set_xticklabels(
        model_names,
        rotation=20,
        ha="right",
    )

    axis.set_ylabel(
        "Mean absolute error (kg/ha)"
    )

    axis.set_title(
        "Rolling-Origin Error across Yield-Distribution Tails",
        fontweight="bold",
        pad=12,
    )

    axis.legend(
        frameon=False,
        ncol=3,
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    paths.extend(
        save_figure(
            figure,
            "Figure_Outlier_Sensitivity_Yield_Tails",
        )
    )

    captions = pd.DataFrame(
        [
            {
                "figure_file":
                    "Figure_Outlier_Sensitivity_IQR.png",
                "caption":
                    (
                        "Mean absolute error for non-outlier and "
                        "training-defined high-IQR yield observations "
                        "under district-grouped and rolling-origin "
                        "development validation."
                    ),
            },
            {
                "figure_file":
                    "Figure_Outlier_Sensitivity_Yield_Tails.png",
                "caption":
                    (
                        "Rolling-origin mean absolute error across "
                        "bottom-decile, central-eighty-percent and "
                        "top-decile observed-yield strata."
                    ),
            },
        ]
    )

    captions_path = (
        OUTPUT_DIR
        / "outlier_sensitivity_figure_captions.csv"
    )

    captions.to_csv(
        captions_path,
        index=False,
        encoding="utf-8-sig",
    )

    paths.append(captions_path)

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_type":
            "development_set_outlier_sensitivity",
        "prediction_source": str(
            PREDICTION_PATH
        ),
        "prediction_source_sha256":
            sha256(PREDICTION_PATH),
        "threshold_source": str(
            THRESHOLD_PATH
        ),
        "threshold_source_sha256":
            sha256(THRESHOLD_PATH),
        "training_fitted_lower_threshold_kg_ha":
            lower_threshold,
        "training_fitted_upper_threshold_kg_ha":
            upper_threshold,
        "protocols": sorted(
            development_protocols
        ),
        "model_ids": MODEL_ORDER,
        "development_prediction_rows": int(
            len(predictions)
        ),
        "unique_development_observations": int(
            len(unique_observations)
        ),
        "high_iqr_outlier_observation_count_by_protocol": {
            str(row["protocol"]):
                int(
                    row[
                        "unique_observation_count"
                    ]
                )
            for _, row in iqr_counts.loc[
                iqr_counts[
                    "target_iqr_outlier_class"
                ].eq("high_iqr_outlier")
            ].iterrows()
        },
        "interpretation_caveat":
            (
                "Only five high-IQR outlier observations were "
                "available per protocol. Results should therefore "
                "be interpreted descriptively rather than as a "
                "high-powered inferential comparison."
            ),
        "safety": {
            "final_test_accessed": False,
            "final_test_rerun": False,
            "models_retrained": False,
            "predictions_recomputed": False,
            "frozen_notebook_modified": False,
            "source_files_modified": False,
        },
        "generation_script": str(
            SCRIPT_PATH
        ),
        "generation_script_sha256":
            sha256(SCRIPT_PATH),
    }

    metadata_path = (
        OUTPUT_DIR
        / "outlier_sensitivity_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    paths.append(metadata_path)

    notes = """OUTLIER SENSITIVITY REPORTING NOTES

1. This analysis uses development-set predictions only.
2. No final-test predictions or source arrays were accessed.
3. No models were retrained and no predictions were recomputed.
4. The IQR thresholds were fitted using the training split.
5. No observations were deleted or winsorised.
6. High-IQR outlier error is compared descriptively with non-outlier error.
7. Only five high-IQR observations were available per protocol.
8. The target-tail analysis provides a broader sensitivity comparison across the bottom decile, central 80%, and top decile.
9. These results quantify performance sensitivity to extreme observed yields, not the effect of retraining after removing outliers.
"""

    notes_path = (
        OUTPUT_DIR
        / "outlier_sensitivity_reporting_notes.txt"
    )

    notes_path.write_text(
        notes,
        encoding="utf-8",
    )

    paths.append(notes_path)

    manifest_rows = []

    for path in sorted(
        paths,
        key=lambda item: item.name,
    ):
        manifest_rows.append(
            {
                "file_name": path.name,
                "full_path": str(path),
                "file_size_bytes":
                    path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest_path = (
        OUTPUT_DIR
        / "outlier_sensitivity_output_sha256_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 92)
    print("OUTLIER SENSITIVITY PACKAGE COMPLETE")
    print("=" * 92)
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        f"Development prediction rows: {len(predictions):,}"
    )
    print(
        f"Unique development observations: "
        f"{len(unique_observations):,}"
    )
    print(
        f"Training-derived upper IQR threshold: "
        f"{upper_threshold:.6f} kg/ha"
    )
    print("Final-test data accessed: False")
    print("Models retrained: False")
    print("Predictions recomputed: False")
    print("")
    print("HIGH-IQR OUTLIER CONTRASTS")
    print(
        iqr_pivot[
            [
                "protocol",
                "model_display_name",
                "non_outlier",
                "high_iqr_outlier",
                "high_outlier_minus_non_outlier_mae_kg_ha",
                "high_outlier_to_non_outlier_mae_ratio",
            ]
        ].to_string(index=False)
    )
    print("")
    print("GENERATED FILES")

    for path in paths + [manifest_path]:
        print(
            f"- {path.name} "
            f"({path.stat().st_size:,} bytes)"
        )


if __name__ == "__main__":
    main()
