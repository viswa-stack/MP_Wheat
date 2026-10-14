from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn


ROOT = Path(r"D:\MP_Wheat_Project")

MODEL_PATH = (
    ROOT
    / "mp_wheat_project_pipeline"
    / "models"
    / "bagging_models"
    / "fitted_models"
    / "historical_only__extra_trees.joblib"
)

DATA_PATH = (
    ROOT
    / "mp_wheat_project_pipeline"
    / "model_inputs"
    / "tabular"
    / "primary_historical_only_panel.csv"
)

OUTPUT_DIR = (
    ROOT
    / "mp_wheat_project_pipeline"
    / "explainability"
    / "tree_shap"
)

EXPECTED_MODEL_KEY = "historical_only__extra_trees"
EXPECTED_SELECTION_SPLIT = "validation"
TOP_DEPENDENCE_FEATURES = 5
PNG_DPI = 400

DISPLAY_NAMES = {
    "yield_lag_1y_kg_ha": "Previous-year yield",
    "yield_prior_2y_mean_kg_ha": "Prior 2-year mean yield",
    "yield_prior_3y_mean_kg_ha": "Prior 3-year mean yield",
    "yield_prior_2y_std_kg_ha": "Prior 2-year yield SD",
    "yield_prior_2y_cv_pct": "Prior 2-year yield CV",
    "yield_prior_3y_std_kg_ha": "Prior 3-year yield SD",
    "yield_prior_3y_cv_pct": "Prior 3-year yield CV",
    "yield_prior_3y_trend_kg_ha_per_year": "Prior 3-year yield trend",
}


def calculate_sha256(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file_handle:
        for block in iter(
            lambda: file_handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def save_current_figure(
    png_path,
    pdf_path,
):
    figure = plt.gcf()

    figure.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def create_output_manifest(file_paths):
    rows = []

    for file_path in sorted(
        file_paths,
        key=lambda path: path.name,
    ):
        rows.append(
            {
                "file_name": file_path.name,
                "full_path": str(file_path),
                "file_extension": file_path.suffix.lower(),
                "file_size_bytes": file_path.stat().st_size,
                "sha256": calculate_sha256(file_path),
            }
        )

    return pd.DataFrame(rows)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input panel not found: {DATA_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = joblib.load(MODEL_PATH)

    required_bundle_keys = {
        "model_key",
        "model_family",
        "scenario",
        "algorithm",
        "input_variant",
        "numeric_feature_names",
        "model",
        "training_split",
        "selection_split",
        "final_test_status",
        "frozen_split_sha256",
        "tabular_preprocessing_sha256",
    }

    missing_bundle_keys = required_bundle_keys.difference(
        bundle.keys()
    )

    if missing_bundle_keys:
        raise KeyError(
            "Model bundle is missing keys: "
            + ", ".join(sorted(missing_bundle_keys))
        )

    if bundle["model_key"] != EXPECTED_MODEL_KEY:
        raise ValueError(
            "Unexpected model bundle: "
            + str(bundle["model_key"])
        )

    if bundle["selection_split"] != EXPECTED_SELECTION_SPLIT:
        raise ValueError(
            "The model was not selected using the validation split."
        )

    if str(bundle["final_test_status"]).lower() != "locked":
        raise ValueError(
            "The saved model does not report a locked final test."
        )

    model = bundle["model"]
    feature_names = list(bundle["numeric_feature_names"])

    if len(feature_names) != int(model.n_features_in_):
        raise ValueError(
            "Model feature count does not match bundle feature names."
        )

    panel = pd.read_csv(DATA_PATH)

    required_columns = {
        "district_id",
        "crop_year_start",
        "data_split",
        "target_yield_kg_ha",
        *feature_names,
    }

    missing_columns = required_columns.difference(
        panel.columns
    )

    if missing_columns:
        raise ValueError(
            "Input panel is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    split_counts = (
        panel["data_split"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    validation_data = panel.loc[
        panel["data_split"]
        .astype(str)
        .eq("validation")
    ].copy()

    if validation_data.empty:
        raise ValueError(
            "No validation rows were found."
        )

    if validation_data[
        "data_split"
    ].astype(str).eq("test").any():
        raise RuntimeError(
            "Test rows entered the SHAP dataset."
        )

    X_validation = validation_data[
        feature_names
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    y_validation = pd.to_numeric(
        validation_data["target_yield_kg_ha"],
        errors="coerce",
    )

    nonfinite_feature_count = int(
        (
            ~np.isfinite(
                X_validation.to_numpy(dtype=float)
            )
        ).sum()
    )

    missing_target_count = int(
        y_validation.isna().sum()
    )

    if nonfinite_feature_count > 0:
        raise ValueError(
            "Validation features contain "
            f"{nonfinite_feature_count} missing or non-finite values."
        )

    if missing_target_count > 0:
        raise ValueError(
            "Validation targets contain "
            f"{missing_target_count} missing values."
        )

    X_values = X_validation.to_numpy(dtype=float)
    y_values = y_validation.to_numpy(dtype=float)

    predictions = model.predict(X_values)

    explainer = shap.TreeExplainer(model)
    raw_explanation = explainer(X_values)

    shap_values = np.asarray(
        raw_explanation.values,
        dtype=float,
    )

    if shap_values.ndim == 3:
        if shap_values.shape[2] != 1:
            raise ValueError(
                "Unexpected multi-output SHAP array."
            )

        shap_values = shap_values[:, :, 0]

    if shap_values.shape != X_values.shape:
        raise ValueError(
            "SHAP output shape does not match input shape."
        )

    base_values = np.asarray(
        raw_explanation.base_values,
        dtype=float,
    ).reshape(-1)

    if base_values.size == 1:
        base_values = np.repeat(
            base_values,
            len(validation_data),
        )

    if base_values.size != len(validation_data):
        raise ValueError(
            "Unexpected SHAP base-value shape."
        )

    reconstructed_predictions = (
        base_values
        + shap_values.sum(axis=1)
    )

    maximum_additivity_error = float(
        np.max(
            np.abs(
                reconstructed_predictions
                - predictions
            )
        )
    )

    if maximum_additivity_error > 1e-3:
        raise RuntimeError(
            "SHAP additivity check failed. "
            f"Maximum error: {maximum_additivity_error}"
        )

    display_names = [
        DISPLAY_NAMES.get(
            feature_name,
            feature_name,
        )
        for feature_name in feature_names
    ]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_values,
        feature_names=display_names,
    )

    mean_absolute_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    mean_signed_shap = np.mean(
        shap_values,
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "importance_rank": np.arange(
                1,
                len(feature_names) + 1,
            ),
            "feature_name": feature_names,
            "display_name": display_names,
            "mean_absolute_shap_kg_ha": (
                mean_absolute_shap
            ),
            "mean_signed_shap_kg_ha": (
                mean_signed_shap
            ),
        }
    ).sort_values(
        "mean_absolute_shap_kg_ha",
        ascending=False,
    ).reset_index(drop=True)

    importance["importance_rank"] = (
        np.arange(
            1,
            len(importance) + 1,
        )
    )

    top_feature_names = importance[
        "feature_name"
    ].head(
        min(
            TOP_DEPENDENCE_FEATURES,
            len(importance),
        )
    ).tolist()

    identifiers = validation_data[
        [
            "district_id",
            "crop_year_start",
            "data_split",
        ]
    ].reset_index(drop=True)

    observation_explanations = identifiers.copy()

    observation_explanations[
        "target_yield_kg_ha"
    ] = y_values

    observation_explanations[
        "model_prediction_kg_ha"
    ] = predictions

    observation_explanations[
        "prediction_residual_kg_ha"
    ] = y_values - predictions

    observation_explanations[
        "shap_expected_value_kg_ha"
    ] = base_values

    observation_explanations[
        "shap_reconstructed_prediction_kg_ha"
    ] = reconstructed_predictions

    for feature_index, feature_name in enumerate(
        feature_names
    ):
        observation_explanations[
            feature_name
        ] = X_values[:, feature_index]

        observation_explanations[
            "shap__" + feature_name
        ] = shap_values[:, feature_index]

    sample_manifest = identifiers.copy()

    sample_manifest[
        "included_in_tree_shap"
    ] = True

    sample_manifest[
        "model_key"
    ] = bundle["model_key"]

    sample_manifest[
        "source_panel"
    ] = str(DATA_PATH)

    sample_manifest[
        "final_test_used"
    ] = False

    generated_paths = []

    plt.figure(
        figsize=(9.4, 6.2)
    )

    shap.plots.beeswarm(
        explanation,
        max_display=len(feature_names),
        show=False,
    )

    plt.title(
        "SHAP Summary for the Leading Extra Trees Model",
        pad=14,
        fontweight="bold",
    )

    beeswarm_png = (
        OUTPUT_DIR
        / "tree_shap_summary_beeswarm.png"
    )

    beeswarm_pdf = (
        OUTPUT_DIR
        / "tree_shap_summary_beeswarm.pdf"
    )

    save_current_figure(
        beeswarm_png,
        beeswarm_pdf,
    )

    generated_paths.extend(
        [
            beeswarm_png,
            beeswarm_pdf,
        ]
    )

    plt.figure(
        figsize=(9.0, 5.8)
    )

    shap.plots.bar(
        explanation,
        max_display=len(feature_names),
        show=False,
    )

    plt.title(
        "Mean Absolute SHAP Importance",
        pad=14,
        fontweight="bold",
    )

    bar_png = (
        OUTPUT_DIR
        / "tree_shap_importance_bar.png"
    )

    bar_pdf = (
        OUTPUT_DIR
        / "tree_shap_importance_bar.pdf"
    )

    save_current_figure(
        bar_png,
        bar_pdf,
    )

    generated_paths.extend(
        [
            bar_png,
            bar_pdf,
        ]
    )

    for feature_name in top_feature_names:
        feature_index = feature_names.index(
            feature_name
        )

        display_name = display_names[
            feature_index
        ]

        safe_name = (
            feature_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        plt.figure(
            figsize=(7.4, 5.4)
        )

        shap.dependence_plot(
            feature_index,
            shap_values,
            X_values,
            feature_names=display_names,
            interaction_index=None,
            show=False,
        )

        plt.title(
            f"SHAP Dependence: {display_name}",
            pad=12,
            fontweight="bold",
        )

        dependence_png = (
            OUTPUT_DIR
            / f"tree_shap_dependence_{safe_name}.png"
        )

        dependence_pdf = (
            OUTPUT_DIR
            / f"tree_shap_dependence_{safe_name}.pdf"
        )

        save_current_figure(
            dependence_png,
            dependence_pdf,
        )

        generated_paths.extend(
            [
                dependence_png,
                dependence_pdf,
            ]
        )

    importance_path = (
        OUTPUT_DIR
        / "tree_shap_mean_absolute_importance.csv"
    )

    observation_path = (
        OUTPUT_DIR
        / "tree_shap_observation_explanations.csv"
    )

    sample_manifest_path = (
        OUTPUT_DIR
        / "tree_shap_sample_manifest.csv"
    )

    captions_path = (
        OUTPUT_DIR
        / "tree_shap_figure_captions.csv"
    )

    metadata_path = (
        OUTPUT_DIR
        / "tree_shap_analysis_metadata.json"
    )

    importance.to_csv(
        importance_path,
        index=False,
        encoding="utf-8-sig",
    )

    observation_explanations.to_csv(
        observation_path,
        index=False,
        encoding="utf-8-sig",
    )

    sample_manifest.to_csv(
        sample_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    captions = [
        {
            "figure_file": (
                "tree_shap_summary_beeswarm.png"
            ),
            "caption": (
                "SHAP beeswarm summary for the leading "
                "development-selected Extra Trees model using "
                "114 validation observations. Positive SHAP values "
                "increase predicted wheat yield and negative values "
                "decrease predicted wheat yield."
            ),
        },
        {
            "figure_file": (
                "tree_shap_importance_bar.png"
            ),
            "caption": (
                "Global feature importance for the leading Extra "
                "Trees model, expressed as mean absolute SHAP values "
                "across the validation observations."
            ),
        },
    ]

    for feature_name in top_feature_names:
        captions.append(
            {
                "figure_file": (
                    "tree_shap_dependence_"
                    + feature_name
                    + ".png"
                ),
                "caption": (
                    "SHAP dependence relationship for "
                    + DISPLAY_NAMES.get(
                        feature_name,
                        feature_name,
                    )
                    + " in the leading Extra Trees model."
                ),
            }
        )

    pd.DataFrame(
        captions
    ).to_csv(
        captions_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "analysis_type": (
            "tree_model_shap_validation_explainability"
        ),
        "model_key": bundle["model_key"],
        "model_family": bundle["model_family"],
        "algorithm": bundle["algorithm"],
        "scenario": bundle["scenario"],
        "model_type": (
            type(model).__module__
            + "."
            + type(model).__name__
        ),
        "model_bundle_path": str(MODEL_PATH),
        "model_bundle_sha256": calculate_sha256(
            MODEL_PATH
        ),
        "input_panel_path": str(DATA_PATH),
        "input_panel_sha256": calculate_sha256(
            DATA_PATH
        ),
        "input_variant": bundle["input_variant"],
        "training_split": bundle["training_split"],
        "selection_split": bundle["selection_split"],
        "final_test_status": bundle[
            "final_test_status"
        ],
        "final_test_used_in_shap": False,
        "explained_split": "validation",
        "available_split_counts": split_counts,
        "explained_observation_count": int(
            len(validation_data)
        ),
        "feature_count": int(
            len(feature_names)
        ),
        "feature_names": feature_names,
        "top_dependence_features": (
            top_feature_names
        ),
        "shap_explainer": "shap.TreeExplainer",
        "maximum_additivity_error": (
            maximum_additivity_error
        ),
        "model_retrained": False,
        "model_modified": False,
        "input_panel_modified": False,
        "frozen_split_sha256": bundle[
            "frozen_split_sha256"
        ],
        "tabular_preprocessing_sha256": bundle[
            "tabular_preprocessing_sha256"
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
        },
        "generation_script": str(
            Path(__file__).resolve()
        ),
        "generation_script_sha256": (
            calculate_sha256(
                Path(__file__).resolve()
            )
        ),
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
        importance_path,
        observation_path,
        sample_manifest_path,
        captions_path,
        metadata_path,
    ]

    manifest = create_output_manifest(
        generated_paths
        + supporting_paths
    )

    manifest_path = (
        OUTPUT_DIR
        / "tree_shap_output_sha256_manifest.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 88)
    print("TREE-MODEL SHAP EXPLAINABILITY COMPLETE")
    print("=" * 88)
    print(f"Model: {bundle['model_key']}")
    print(
        "Explained split: validation"
    )
    print(
        f"Validation observations: {len(validation_data)}"
    )
    print(
        "Final-test observations used: False"
    )
    print(
        "Model retrained: False"
    )
    print(
        f"Maximum SHAP additivity error: "
        f"{maximum_additivity_error:.12f}"
    )
    print(
        f"Output directory: {OUTPUT_DIR}"
    )
    print("")
    print("TOP SHAP FEATURES")

    print(
        importance[
            [
                "importance_rank",
                "feature_name",
                "mean_absolute_shap_kg_ha",
            ]
        ]
        .head(TOP_DEPENDENCE_FEATURES)
        .to_string(index=False)
    )

    print("")
    print("GENERATED FILES")

    for file_path in (
        generated_paths
        + supporting_paths
        + [manifest_path]
    ):
        print(
            f"- {file_path.name} "
            f"({file_path.stat().st_size:,} bytes)"
        )


if __name__ == "__main__":
    main()