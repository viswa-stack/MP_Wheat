from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path(r"D:\MP_Wheat_Project")

PIPELINE = (
    ROOT
    / "mp_wheat_project_pipeline"
)

OUTPUT_DIR = (
    PIPELINE
    / "reporting"
    / "computational_efficiency"
)

CLASSICAL_SUMMARY_PATH = (
    PIPELINE
    / "validation"
    / "rolling_origin"
    / "classical_models"
    / "rolling_classical_aggregate_metrics.csv"
)

DEEP_SUMMARY_PATH = (
    PIPELINE
    / "validation"
    / "rolling_origin"
    / "deep_learning_models"
    / "rolling_validation"
    / "deep_model_summary.csv"
)

DEEP_AUDIT_PATH = (
    PIPELINE
    / "validation"
    / "rolling_origin"
    / "deep_learning_models"
    / "rolling_validation"
    / "deep_run_audit.csv"
)

SCRIPT_PATH = Path(__file__).resolve()

MODEL_ORDER = [
    "baseline__prior_2year_moving_average",
    "linear__elastic_net",
    "deep_learning__ft_transformer",
    "deep_learning__lstm",
    "deep_learning__simple_rnn",
]

DISPLAY_NAMES = {
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

MODEL_FAMILIES = {
    "baseline__prior_2year_moving_average":
        "Historical baseline",
    "linear__elastic_net":
        "Linear model",
    "deep_learning__ft_transformer":
        "Deep-learning transformer",
    "deep_learning__lstm":
        "Deep-learning recurrent",
    "deep_learning__simple_rnn":
        "Deep-learning recurrent",
}


def calculate_sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def required_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required source file not found: {path}"
        )


def as_float(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    return float(text)


def as_int(value):
    numeric = as_float(value)

    if numeric is None:
        return None

    return int(round(numeric))


def format_seconds(value):
    if value is None or pd.isna(value):
        return "Not recorded"

    value = float(value)

    if value == 0:
        return "0.000000"

    if abs(value) < 0.001:
        return f"{value:.8f}"

    return f"{value:.4f}"


def format_integer(value, missing_text):
    if value is None or pd.isna(value):
        return missing_text

    return f"{int(round(float(value))):,}"


def format_size_kib(value):
    if value is None or pd.isna(value):
        return "Not recorded"

    return f"{float(value) / 1024:.2f}"


def build_source_manifest(paths):
    rows = []

    for path in paths:
        rows.append(
            {
                "file_name": path.name,
                "full_path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        )

    return pd.DataFrame(rows)


def build_output_manifest(paths):
    rows = []

    for path in sorted(
        paths,
        key=lambda item: item.name,
    ):
        rows.append(
            {
                "file_name": path.name,
                "full_path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": calculate_sha256(path),
            }
        )

    return pd.DataFrame(rows)


def main():
    source_paths = [
        CLASSICAL_SUMMARY_PATH,
        DEEP_SUMMARY_PATH,
        DEEP_AUDIT_PATH,
        SCRIPT_PATH,
    ]

    for source_path in source_paths:
        required_file(source_path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classical = pd.read_csv(
        CLASSICAL_SUMMARY_PATH
    )

    deep_summary = pd.read_csv(
        DEEP_SUMMARY_PATH
    )

    deep_audit = pd.read_csv(
        DEEP_AUDIT_PATH
    )

    rows = []

    classical_ids = [
        "baseline__prior_2year_moving_average",
        "linear__elastic_net",
    ]

    for model_id in classical_ids:
        selected = classical.loc[
            classical["model_id"].astype(str).eq(model_id)
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Expected one classical summary row for {model_id}, "
                f"found {len(selected)}."
            )

        row = selected.iloc[0]

        fold_count = as_int(
            row["fold_count"]
        )

        mean_fit_seconds = as_float(
            row["mean_fit_seconds"]
        )

        total_fit_seconds = as_float(
            row["total_fit_seconds"]
        )

        mean_prediction_seconds = as_float(
            row["mean_prediction_seconds"]
        )

        total_prediction_seconds = (
            mean_prediction_seconds * fold_count
            if mean_prediction_seconds is not None
            and fold_count is not None
            else None
        )

        artifact_size = as_float(
            row["mean_model_bundle_size_bytes"]
        )

        parameter_status = (
            "Not applicable"
            if model_id
            == "baseline__prior_2year_moving_average"
            else "Not recorded in source artifacts"
        )

        rows.append(
            {
                "table_order": MODEL_ORDER.index(model_id) + 1,
                "model": DISPLAY_NAMES[model_id],
                "model_id": model_id,
                "model_family": MODEL_FAMILIES[model_id],
                "algorithm": str(row["algorithm"]),
                "evaluation_protocol":
                    "Rolling-origin validation",
                "seed_count": 1,
                "fold_count_per_evaluation": fold_count,
                "pooled_or_mean_pooled_mae_kg_ha":
                    as_float(row["pooled_mae_kg_ha"]),
                "mean_training_or_fit_seconds_per_fold":
                    mean_fit_seconds,
                "total_training_or_fit_seconds_per_9_fold_evaluation":
                    total_fit_seconds,
                "mean_prediction_seconds_per_fold":
                    mean_prediction_seconds,
                "total_prediction_seconds_per_9_fold_evaluation":
                    total_prediction_seconds,
                "prediction_time_status":
                    "Recorded",
                "parameter_count": None,
                "parameter_count_status":
                    parameter_status,
                "mean_model_artifact_size_bytes":
                    artifact_size,
                "mean_model_artifact_size_kib":
                    (
                        artifact_size / 1024
                        if artifact_size is not None
                        else None
                    ),
                "artifact_type":
                    "Serialized model bundle",
                "runtime_scope":
                    "Mean and total across nine rolling-origin folds",
                "runtime_source_file":
                    str(CLASSICAL_SUMMARY_PATH),
            }
        )

    deep_ids = [
        "deep_learning__ft_transformer",
        "deep_learning__lstm",
        "deep_learning__simple_rnn",
    ]

    for model_id in deep_ids:
        selected = deep_summary.loc[
            deep_summary["model_id"].astype(str).eq(model_id)
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Expected one deep summary row for {model_id}, "
                f"found {len(selected)}."
            )

        row = selected.iloc[0]

        fold_count = as_int(
            row["fold_count_per_seed"]
        )

        seed_count = as_int(
            row["seed_count"]
        )

        mean_total_training_seconds = as_float(
            row["mean_total_training_seconds"]
        )

        mean_training_seconds_per_fold = (
            mean_total_training_seconds / fold_count
            if mean_total_training_seconds is not None
            and fold_count is not None
            and fold_count > 0
            else None
        )

        audit_rows = deep_audit.loc[
            deep_audit["model_id"].astype(str).eq(model_id)
            & deep_audit["checkpoint_exists"]
            .astype(str)
            .str.lower()
            .eq("true")
        ].copy()

        if audit_rows.empty:
            raise ValueError(
                f"No saved checkpoint evidence found for {model_id}."
            )

        checkpoint_sizes = pd.to_numeric(
            audit_rows["checkpoint_size_bytes"],
            errors="coerce",
        ).dropna()

        if checkpoint_sizes.empty:
            raise ValueError(
                f"No valid checkpoint sizes found for {model_id}."
            )

        mean_checkpoint_size = float(
            checkpoint_sizes.mean()
        )

        rows.append(
            {
                "table_order": MODEL_ORDER.index(model_id) + 1,
                "model": DISPLAY_NAMES[model_id],
                "model_id": model_id,
                "model_family": MODEL_FAMILIES[model_id],
                "algorithm": str(row["algorithm"]),
                "evaluation_protocol":
                    "Rolling-origin validation",
                "seed_count": seed_count,
                "fold_count_per_evaluation": fold_count,
                "pooled_or_mean_pooled_mae_kg_ha":
                    as_float(row["mean_pooled_mae_kg_ha"]),
                "mean_training_or_fit_seconds_per_fold":
                    mean_training_seconds_per_fold,
                "total_training_or_fit_seconds_per_9_fold_evaluation":
                    mean_total_training_seconds,
                "mean_prediction_seconds_per_fold":
                    None,
                "total_prediction_seconds_per_9_fold_evaluation":
                    None,
                "prediction_time_status":
                    "Not recorded in source artifacts",
                "parameter_count":
                    as_int(row["parameter_count"]),
                "parameter_count_status":
                    "Recorded",
                "mean_model_artifact_size_bytes":
                    mean_checkpoint_size,
                "mean_model_artifact_size_kib":
                    mean_checkpoint_size / 1024,
                "artifact_type":
                    "Saved model checkpoint",
                "runtime_scope":
                    (
                        "Mean across three seed-specific "
                        "nine-fold rolling-origin evaluations"
                    ),
                "runtime_source_file":
                    str(DEEP_SUMMARY_PATH),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        "table_order"
    ).reset_index(drop=True)

    expected_ids = set(MODEL_ORDER)
    actual_ids = set(table["model_id"])

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Final efficiency table does not contain the expected models."
        )

    table_path = (
        OUTPUT_DIR
        / "computational_efficiency_final_table.csv"
    )

    markdown_path = (
        OUTPUT_DIR
        / "computational_efficiency_manuscript_table.md"
    )

    metadata_path = (
        OUTPUT_DIR
        / "computational_efficiency_metadata.json"
    )

    source_manifest_path = (
        OUTPUT_DIR
        / "computational_efficiency_source_manifest.csv"
    )

    notes_path = (
        OUTPUT_DIR
        / "computational_efficiency_reporting_notes.txt"
    )

    output_manifest_path = (
        OUTPUT_DIR
        / "computational_efficiency_output_sha256_manifest.csv"
    )

    table.to_csv(
        table_path,
        index=False,
        encoding="utf-8-sig",
    )

    markdown_lines = [
        "# Computational Efficiency under Rolling-Origin Validation",
        "",
        "| Model | MAE (kg/ha) | Mean training/fit per fold (s) | Total training/fit per 9-fold evaluation (s) | Mean prediction per fold (s) | Parameters | Artifact size (KiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in table.iterrows():
        if row["parameter_count_status"] == "Not applicable":
            parameter_text = "N/A"
        elif pd.isna(row["parameter_count"]):
            parameter_text = "Not recorded"
        else:
            parameter_text = format_integer(
                row["parameter_count"],
                "Not recorded",
            )

        prediction_text = (
            format_seconds(
                row["mean_prediction_seconds_per_fold"]
            )
            if row["prediction_time_status"] == "Recorded"
            else "Not recorded"
        )

        markdown_lines.append(
            "| "
            + str(row["model"])
            + " | "
            + f"{float(row['pooled_or_mean_pooled_mae_kg_ha']):.3f}"
            + " | "
            + format_seconds(
                row["mean_training_or_fit_seconds_per_fold"]
            )
            + " | "
            + format_seconds(
                row[
                    "total_training_or_fit_seconds_per_9_fold_evaluation"
                ]
            )
            + " | "
            + prediction_text
            + " | "
            + parameter_text
            + " | "
            + format_size_kib(
                row["mean_model_artifact_size_bytes"]
            )
            + " |"
        )

    markdown_lines.extend(
        [
            "",
            "**Table note.** Classical-model times are based on "
            "nine rolling-origin folds. Deep-model totals are the "
            "mean total training time for a complete nine-fold "
            "evaluation across three random seeds; their per-fold "
            "values are derived by dividing this mean total by nine. "
            "Prediction time was logged only for the classical models. "
            "Artifact sizes represent mean serialized model-bundle "
            "size for classical methods and mean saved-checkpoint size "
            "for deep models.",
            "",
            "**Hardware note.** The available system comprised an AMD "
            "Ryzen 7 7700X CPU, 31.04 GB RAM and an NVIDIA GeForce RTX "
            "3060 with 12,288 MiB dedicated memory. The runtime source "
            "files do not independently establish which processing "
            "device was used for every original run.",
        ]
    )

    markdown_path.write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    reporting_notes = """COMPUTATIONAL EFFICIENCY REPORTING NOTES

1. The table uses rolling-origin validation evidence only.
2. Classical fit and prediction times were explicitly recorded separately.
3. Deep-model training time was recorded, but deep-model prediction time was not.
4. Missing deep-model inference times must be reported as 'Not recorded'.
5. The prior 2-year moving-average baseline has no trainable parameters.
6. Elastic Net parameter count was not stored in the source efficiency artifacts.
7. Deep-model parameter counts are taken from the saved rolling-origin model summaries.
8. Deep-model artifact sizes are based on saved checkpoint files.
9. Hardware availability does not prove that every original run used the GPU.
10. No final-test data were accessed and no models were retrained while generating this package.
"""

    notes_path.write_text(
        reporting_notes,
        encoding="utf-8",
    )

    source_manifest = build_source_manifest(
        source_paths
    )

    source_manifest.to_csv(
        source_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_type":
            "post_freeze_computational_efficiency_reporting",
        "evaluation_protocol":
            "rolling_origin_validation",
        "model_count": int(len(table)),
        "models": table["model_id"].tolist(),
        "safety": {
            "final_test_accessed": False,
            "final_test_rerun": False,
            "models_retrained": False,
            "frozen_notebook_modified": False,
            "source_results_modified": False,
        },
        "runtime_interpretation": {
            "classical_models":
                "Mean and total fit time across nine rolling-origin folds.",
            "deep_models":
                "Mean total training time for one complete nine-fold evaluation across three seeds.",
            "deep_per_fold_derivation":
                "Mean total training time divided by nine folds.",
            "classical_prediction_time":
                "Recorded directly in the rolling-origin aggregate metrics.",
            "deep_prediction_time":
                "Not recorded and therefore not estimated.",
        },
        "hardware_environment": {
            "operating_system":
                "Microsoft Windows 11 Pro",
            "operating_system_version":
                "10.0.26200",
            "system_type":
                "x64-based PC",
            "processor":
                "AMD Ryzen 7 7700X 8-Core Processor",
            "physical_cpu_cores": 8,
            "logical_processors": 16,
            "total_ram_gb": 31.04,
            "integrated_gpu":
                "AMD Radeon(TM) Graphics",
            "discrete_gpu":
                "NVIDIA GeForce RTX 3060",
            "discrete_gpu_memory_mib": 12288,
            "nvidia_driver_version":
                "591.86",
            "driver_reported_cuda_compatibility":
                "13.1",
            "device_use_caveat":
                (
                    "The runtime source files do not independently "
                    "record CPU/GPU device utilization for every "
                    "original training run."
                ),
        },
        "software_environment": {
            "python": "3.11.0",
            "python_implementation": "CPython",
            "numpy": "1.26.4",
            "pandas": "2.2.3",
            "scikit_learn": "1.8.0",
            "scipy": "1.13.1",
            "matplotlib": "3.10.7",
            "joblib": "1.5.2",
            "shap": "0.46.0",
            "tensorflow": "2.20.0",
            "torch": "2.11.0+cu126",
            "lightgbm": "4.6.0",
            "xgboost": "2.1.4",
            "catboost": "1.2.10",
        },
        "source_files": [
            {
                "path": str(path),
                "sha256": calculate_sha256(path),
            }
            for path in source_paths
        ],
        "output_directory": str(OUTPUT_DIR),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    output_paths = [
        table_path,
        markdown_path,
        metadata_path,
        source_manifest_path,
        notes_path,
    ]

    output_manifest = build_output_manifest(
        output_paths
    )

    output_manifest.to_csv(
        output_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 88)
    print("COMPUTATIONAL EFFICIENCY PACKAGE COMPLETE")
    print("=" * 88)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Models included: {len(table)}")
    print("Final-test data accessed: False")
    print("Models retrained: False")
    print("Frozen notebook modified: False")
    print("")
    print(
        table[
            [
                "model",
                "mean_training_or_fit_seconds_per_fold",
                "total_training_or_fit_seconds_per_9_fold_evaluation",
                "mean_prediction_seconds_per_fold",
                "parameter_count",
                "mean_model_artifact_size_kib",
            ]
        ].to_string(index=False)
    )
    print("")
    print("GENERATED FILES")

    for path in output_paths + [output_manifest_path]:
        print(
            f"- {path.name} "
            f"({path.stat().st_size:,} bytes)"
        )


if __name__ == "__main__":
    main()
