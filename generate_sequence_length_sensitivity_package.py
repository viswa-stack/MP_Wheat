from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(r"D:\MP_Wheat_Project")
PIPELINE = ROOT / "mp_wheat_project_pipeline"

RESOLUTION_ROOT = (
    PIPELINE
    / "validation"
    / "ablation"
    / "unified_synthesis_and_robustness_protocol"
    / "development_set_robustness"
    / "frozen_model_perturbation_preflight"
    / "exact_artifact_resolution"
)

RECONSTRUCTION_ROOT = (
    RESOLUTION_ROOT
    / "dynamic_recurrent_architecture_reconstruction_v3"
)

MAPPING_PATH = (
    RESOLUTION_ROOT
    / "perturbation_preprocessing_mapping_audit.csv"
)

REPLAY_PATH = (
    RECONSTRUCTION_ROOT
    / "reconstruction_v3_full_checkpoint_replay.csv"
)

OUTPUT_DIR = (
    PIPELINE
    / "reporting"
    / "robustness"
    / "sequence_length_sensitivity"
)

SCRIPT_PATH = Path(__file__).resolve()

ALLOWED_MODELS = {
    "deep_learning__lstm",
    "deep_learning__simple_rnn",
}

MODEL_ORDER = [
    "deep_learning__lstm",
    "deep_learning__simple_rnn",
]

MODEL_NAMES = {
    "deep_learning__lstm": "LSTM",
    "deep_learning__simple_rnn": "Simple RNN",
}

PROTOCOL_ORDER = [
    "district_grouped",
    "rolling_origin",
]

PROHIBITED_PATH_MARKERS = (
    "one_time_locked_final_test_execution",
    "final_test_execution",
    "test_execution",
)

SEQUENCE_LENGTHS = list(range(1, 8))
REFERENCE_SEQUENCE_LENGTH = 7
REFERENCE_TOLERANCE = 1e-5
BOOTSTRAP_REPLICATES = 10000
MONTE_CARLO_SIGN_FLIPS = 100000
CONFIDENCE_LEVEL = 0.95


def sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def require_safe_file(path):
    path = Path(path)
    lower = str(path).lower()

    for marker in PROHIBITED_PATH_MARKERS:
        if marker in lower:
            raise RuntimeError(
                f"Blocked prohibited path: {path}"
            )

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_torch_checkpoint(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def get_state_dictionary(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint must be dictionary-like."
        )

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            return candidate

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise KeyError(
        "No model state dictionary was found."
    )


class DynamicReconstructedSequenceRegressorV3(nn.Module):
    def __init__(
        self,
        state_dictionary,
        model_id,
        dropout,
    ):
        super().__init__()

        input_weight = state_dictionary[
            "recurrent_layer.weight_ih_l0"
        ]

        input_size = int(
            input_weight.shape[1]
        )

        recurrent_width = int(
            state_dictionary[
                "recurrent_normalisation.weight"
            ].numel()
        )

        recurrent_layer_indices = sorted(
            {
                int(
                    name.split("_l")[-1]
                )
                for name in state_dictionary
                if name.startswith(
                    "recurrent_layer.weight_ih_l"
                )
            }
        )

        num_layers = len(
            recurrent_layer_indices
        )

        if model_id == "deep_learning__lstm":
            self.recurrent_layer = nn.LSTM(
                input_size=input_size,
                hidden_size=recurrent_width,
                num_layers=num_layers,
                batch_first=True,
                dropout=(
                    float(dropout)
                    if num_layers > 1
                    else 0.0
                ),
                bidirectional=False,
            )
        elif model_id == "deep_learning__simple_rnn":
            self.recurrent_layer = nn.RNN(
                input_size=input_size,
                hidden_size=recurrent_width,
                num_layers=num_layers,
                nonlinearity="tanh",
                batch_first=True,
                dropout=(
                    float(dropout)
                    if num_layers > 1
                    else 0.0
                ),
                bidirectional=False,
            )
        else:
            raise ValueError(
                f"Unsupported model: {model_id}"
            )

        self.recurrent_normalisation = nn.LayerNorm(
            recurrent_width
        )

        static_weight = state_dictionary[
            "static_projection.0.weight"
        ]

        static_output_width = int(
            static_weight.shape[0]
        )

        static_input_width = int(
            static_weight.shape[1]
        )

        self.static_projection = nn.Sequential(
            nn.Linear(
                static_input_width,
                static_output_width,
            ),
            nn.ReLU(),
        )

        output_weight = state_dictionary[
            "output_head.0.weight"
        ]

        head_hidden_width = int(
            output_weight.shape[0]
        )

        fused_input_width = int(
            output_weight.shape[1]
        )

        expected_fused_width = (
            recurrent_width
            + static_output_width
        )

        if fused_input_width != expected_fused_width:
            raise RuntimeError(
                "Checkpoint fusion width is inconsistent: "
                f"{fused_input_width} != "
                f"{expected_fused_width}"
            )

        self.output_head = nn.Sequential(
            nn.Linear(
                fused_input_width,
                head_hidden_width,
            ),
            nn.ReLU(),
            nn.Dropout(
                float(dropout)
            ),
            nn.Linear(
                head_hidden_width,
                1,
            ),
        )

    def forward(
        self,
        sequence_input,
        static_input,
    ):
        recurrent_output, _ = self.recurrent_layer(
            sequence_input
        )

        pooled = recurrent_output[:, -1, :]

        pooled = self.recurrent_normalisation(
            pooled
        )

        static_embedding = self.static_projection(
            static_input
        )

        fused = torch.cat(
            [
                pooled,
                static_embedding,
            ],
            dim=1,
        )

        return self.output_head(
            fused
        ).squeeze(-1)


def construct_model(
    checkpoint,
    model_id,
):
    state_dictionary = get_state_dictionary(
        checkpoint
    )

    parameter_dictionary = checkpoint.get(
        "parameter_dictionary",
        {},
    )

    dropout = float(
        parameter_dictionary.get(
            "dropout",
            0.0,
        )
    )

    model = DynamicReconstructedSequenceRegressorV3(
        state_dictionary=state_dictionary,
        model_id=model_id,
        dropout=dropout,
    )

    load_result = model.load_state_dict(
        state_dictionary,
        strict=True,
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict checkpoint loading failed."
        )

    model.eval()

    return model


def predict_scaled(
    model,
    sequence_array,
    static_array,
):
    sequence_tensor = torch.as_tensor(
        sequence_array,
        dtype=torch.float32,
    )

    static_tensor = torch.as_tensor(
        static_array,
        dtype=torch.float32,
    )

    with torch.no_grad():
        prediction = model(
            sequence_tensor,
            static_tensor,
        )

    result = (
        prediction.detach()
        .cpu()
        .numpy()
        .astype(float)
        .reshape(-1)
    )

    if not np.isfinite(result).all():
        raise RuntimeError(
            "Nonfinite scaled predictions generated."
        )

    return result


def inverse_target(
    scaled_predictions,
    target_scaler,
):
    raw = target_scaler.inverse_transform(
        np.asarray(
            scaled_predictions,
            dtype=float,
        ).reshape(-1, 1)
    ).reshape(-1)

    if not np.isfinite(raw).all():
        raise RuntimeError(
            "Nonfinite raw predictions generated."
        )

    return raw


def calculate_metrics(
    observed,
    predicted,
):
    observed = np.asarray(
        observed,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    residual = predicted - observed
    absolute_error = np.abs(residual)

    mae = float(
        np.mean(absolute_error)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                residual ** 2
            )
        )
    )

    denominator = float(
        np.sum(
            (
                observed
                - np.mean(observed)
            ) ** 2
        )
    )

    r2 = (
        float(
            1.0
            - np.sum(
                residual ** 2
            )
            / denominator
        )
        if denominator > 0
        else float("nan")
    )

    nonzero = np.abs(observed) > 1e-12

    mape = (
        float(
            np.mean(
                np.abs(
                    residual[nonzero]
                    / observed[nonzero]
                )
            )
            * 100.0
        )
        if nonzero.any()
        else float("nan")
    )

    return {
        "observation_count": int(
            len(observed)
        ),
        "mae_kg_ha": mae,
        "rmse_kg_ha": rmse,
        "r2": r2,
        "mape_pct": mape,
        "median_absolute_error_kg_ha": float(
            np.median(
                absolute_error
            )
        ),
        "mean_bias_kg_ha": float(
            np.mean(residual)
        ),
        "maximum_absolute_error_kg_ha": float(
            np.max(
                absolute_error
            )
        ),
    }


def deterministic_seed(text):
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return int(
        digest[:8],
        16,
    )


def cluster_bootstrap_ci(
    cluster_values,
    seed,
):
    values = np.asarray(
        cluster_values,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    index_matrix = rng.integers(
        0,
        len(values),
        size=(
            BOOTSTRAP_REPLICATES,
            len(values),
        ),
    )

    bootstrap_means = values[
        index_matrix
    ].mean(axis=1)

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

    lower = float(
        np.quantile(
            bootstrap_means,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def cluster_sign_flip_test(
    cluster_values,
    seed,
):
    values = np.asarray(
        cluster_values,
        dtype=float,
    )

    observed_statistic = abs(
        float(
            np.mean(values)
        )
    )

    cluster_count = len(values)

    if cluster_count <= 20:
        draw_count = 2 ** cluster_count
        extreme_count = 0

        positions = np.arange(
            cluster_count,
            dtype=np.uint64,
        )

        for mask in range(
            draw_count
        ):
            signs = np.where(
                (
                    (
                        np.uint64(mask)
                        >> positions
                    )
                    & np.uint64(1)
                )
                == 1,
                1.0,
                -1.0,
            )

            statistic = abs(
                float(
                    np.mean(
                        values * signs
                    )
                )
            )

            if (
                statistic
                >= observed_statistic
                - 1e-15
            ):
                extreme_count += 1

        p_value = (
            extreme_count
            / draw_count
        )

        return (
            float(p_value),
            "exact_cluster_sign_flip",
            int(draw_count),
        )

    rng = np.random.default_rng(
        seed
    )

    extreme_count = 0
    completed = 0
    chunk_size = 5000

    while completed < MONTE_CARLO_SIGN_FLIPS:
        current = min(
            chunk_size,
            MONTE_CARLO_SIGN_FLIPS
            - completed,
        )

        signs = rng.choice(
            np.array(
                [-1.0, 1.0]
            ),
            size=(
                current,
                cluster_count,
            ),
        )

        statistics = np.abs(
            signs.dot(values)
            / cluster_count
        )

        extreme_count += int(
            np.sum(
                statistics
                >= observed_statistic
                - 1e-15
            )
        )

        completed += current

    p_value = (
        extreme_count + 1
    ) / (
        MONTE_CARLO_SIGN_FLIPS + 1
    )

    return (
        float(p_value),
        "monte_carlo_cluster_sign_flip",
        int(MONTE_CARLO_SIGN_FLIPS),
    )


def holm_adjust(
    p_values,
):
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty_like(
        p_values
    )

    running_maximum = 0.0
    comparison_count = len(
        p_values
    )

    for rank, index in enumerate(
        order
    ):
        multiplier = (
            comparison_count - rank
        )

        candidate = min(
            1.0,
            p_values[index]
            * multiplier,
        )

        running_maximum = max(
            running_maximum,
            candidate,
        )

        adjusted[index] = (
            running_maximum
        )

    return adjusted


def save_figure(
    figure,
    stem,
):
    png_path = (
        OUTPUT_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{stem}.pdf"
    )

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

    plt.close(
        figure
    )

    return [
        png_path,
        pdf_path,
    ]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260725)
    np.random.seed(20260725)

    require_safe_file(
        MAPPING_PATH
    )

    require_safe_file(
        REPLAY_PATH
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mapping = pd.read_csv(
        MAPPING_PATH
    )

    replay = pd.read_csv(
        REPLAY_PATH
    )

    mapping = mapping.loc[
        mapping[
            "protocol"
        ].isin(PROTOCOL_ORDER)
    ].copy()

    recurrent_replay = replay.loc[
        replay[
            "model_id"
        ].isin(ALLOWED_MODELS)
    ].copy()

    if len(mapping) != 14:
        raise RuntimeError(
            "Expected 14 development preprocessing bundles; "
            f"found {len(mapping)}."
        )

    if len(recurrent_replay) != 84:
        raise RuntimeError(
            "Expected 84 recurrent checkpoints; "
            f"found {len(recurrent_replay)}."
        )

    if recurrent_replay[
        "locked_year_overlap_count"
    ].fillna(0).astype(int).sum() != 0:
        raise RuntimeError(
            "Locked final-test year overlap detected."
        )

    if recurrent_replay.duplicated(
        [
            "protocol",
            "model_id",
            "fold_id",
            "seed",
        ]
    ).any():
        raise RuntimeError(
            "Duplicate recurrent checkpoint rows detected."
        )

    bundle_paths = {
        (
            str(row.protocol),
            str(row.fold_id),
        ): require_safe_file(
            row.bundle_path
        )
        for row in mapping.itertuples(
            index=False
        )
    }

    bundle_cache = {}

    for key, path in bundle_paths.items():
        bundle = joblib.load(
            path
        )

        required_keys = {
            "X_sequence_validation_scaled",
            "X_static_validation_scaled",
            "y_validation_raw",
            "validation_keys",
            "target_scaler",
        }

        missing = (
            required_keys
            - set(bundle.keys())
        )

        if missing:
            raise KeyError(
                f"Bundle {path} is missing: "
                + ", ".join(
                    sorted(missing)
                )
            )

        sequence_array = np.asarray(
            bundle[
                "X_sequence_validation_scaled"
            ],
            dtype=float,
        )

        if (
            sequence_array.ndim != 3
            or sequence_array.shape[1:] != (
                7,
                13,
            )
        ):
            raise RuntimeError(
                f"Unexpected sequence shape in {path}: "
                f"{sequence_array.shape}"
            )

        static_array = np.asarray(
            bundle[
                "X_static_validation_scaled"
            ],
            dtype=float,
        )

        observed = np.asarray(
            bundle[
                "y_validation_raw"
            ],
            dtype=float,
        ).reshape(-1)

        validation_keys = bundle[
            "validation_keys"
        ].copy()

        if len(validation_keys.columns) < 2:
            raise RuntimeError(
                "Validation keys require at least two columns."
            )

        if (
            "district_id"
            not in validation_keys.columns
        ):
            validation_keys = (
                validation_keys.rename(
                    columns={
                        validation_keys.columns[0]:
                            "district_id"
                    }
                )
            )

        if (
            "crop_year_start"
            not in validation_keys.columns
        ):
            remaining = [
                column
                for column
                in validation_keys.columns
                if column != "district_id"
            ]

            validation_keys = (
                validation_keys.rename(
                    columns={
                        remaining[0]:
                            "crop_year_start"
                    }
                )
            )

        validation_keys = (
            validation_keys[
                [
                    "district_id",
                    "crop_year_start",
                ]
            ]
            .reset_index(drop=True)
        )

        row_count = len(
            observed
        )

        if not (
            len(sequence_array)
            == len(static_array)
            == len(validation_keys)
            == row_count
        ):
            raise RuntimeError(
                f"Bundle row alignment failure: {path}"
            )

        if not (
            np.isfinite(
                sequence_array
            ).all()
            and np.isfinite(
                static_array
            ).all()
            and np.isfinite(
                observed
            ).all()
        ):
            raise RuntimeError(
                f"Nonfinite bundle values detected: {path}"
            )

        bundle_cache[key] = {
            "path": path,
            "sequence": sequence_array,
            "static": static_array,
            "observed": observed,
            "validation_keys":
                validation_keys,
            "target_scaler":
                bundle[
                    "target_scaler"
                ],
        }

    seed_prediction_frames = []
    replay_audit_rows = []
    source_paths = {
        MAPPING_PATH,
        REPLAY_PATH,
        SCRIPT_PATH,
    }

    recurrent_replay = (
        recurrent_replay.sort_values(
            [
                "protocol",
                "model_id",
                "fold_id",
                "seed",
            ]
        )
    )

    for row in recurrent_replay.itertuples(
        index=False
    ):
        protocol = str(
            row.protocol
        )

        model_id = str(
            row.model_id
        )

        fold_id = str(
            row.fold_id
        )

        seed = int(
            row.seed
        )

        bundle_key = (
            protocol,
            fold_id,
        )

        if bundle_key not in bundle_cache:
            raise KeyError(
                f"Missing bundle for {bundle_key}"
            )

        bundle = bundle_cache[
            bundle_key
        ]

        checkpoint_path = require_safe_file(
            row.checkpoint_path
        )

        run_artifact_path = require_safe_file(
            row.run_artifact_path
        )

        source_paths.add(
            checkpoint_path
        )

        source_paths.add(
            run_artifact_path
        )

        source_paths.add(
            bundle[
                "path"
            ]
        )

        checkpoint = load_torch_checkpoint(
            checkpoint_path
        )

        checkpoint_status = str(
            checkpoint.get(
                "final_test_status",
                "",
            )
        ).upper()

        if checkpoint_status not in {
            "",
            "LOCKED",
        }:
            raise RuntimeError(
                "Unexpected validation-checkpoint final-test status: "
                f"{checkpoint_status}"
            )

        model = construct_model(
            checkpoint=checkpoint,
            model_id=model_id,
        )

        sequence = bundle[
            "sequence"
        ]

        static = bundle[
            "static"
        ]

        observed = bundle[
            "observed"
        ]

        target_scaler = bundle[
            "target_scaler"
        ]

        validation_keys = bundle[
            "validation_keys"
        ]

        full_scaled_prediction = predict_scaled(
            model=model,
            sequence_array=sequence,
            static_array=static,
        )

        full_raw_prediction = inverse_target(
            full_scaled_prediction,
            target_scaler,
        )

        run_artifact = joblib.load(
            run_artifact_path
        )

        saved_raw_prediction = np.asarray(
            run_artifact[
                "predictions"
            ],
            dtype=float,
        ).reshape(-1)

        saved_scaled_prediction = np.asarray(
            run_artifact[
                "scaled_predictions"
            ],
            dtype=float,
        ).reshape(-1)

        if not (
            len(full_raw_prediction)
            == len(saved_raw_prediction)
            == len(saved_scaled_prediction)
            == len(observed)
        ):
            raise RuntimeError(
                "Reference replay row-count mismatch."
            )

        maximum_raw_replay_error = float(
            np.max(
                np.abs(
                    full_raw_prediction
                    - saved_raw_prediction
                )
            )
        )

        maximum_scaled_replay_error = float(
            np.max(
                np.abs(
                    full_scaled_prediction
                    - saved_scaled_prediction
                )
            )
        )

        replay_audit_rows.append(
            {
                "protocol": protocol,
                "model_id": model_id,
                "fold_id": fold_id,
                "seed": seed,
                "prediction_count":
                    len(observed),
                "maximum_raw_replay_error_kg_ha":
                    maximum_raw_replay_error,
                "maximum_scaled_replay_error":
                    maximum_scaled_replay_error,
                "replay_within_tolerance":
                    (
                        maximum_raw_replay_error
                        <= REFERENCE_TOLERANCE
                        and
                        maximum_scaled_replay_error
                        <= REFERENCE_TOLERANCE
                    ),
            }
        )

        if (
            maximum_raw_replay_error
            > REFERENCE_TOLERANCE
            or maximum_scaled_replay_error
            > REFERENCE_TOLERANCE
        ):
            raise RuntimeError(
                "Frozen reference replay exceeded tolerance: "
                f"{protocol}, {model_id}, "
                f"{fold_id}, {seed}"
            )

        for effective_length in SEQUENCE_LENGTHS:
            neutralised_count = (
                REFERENCE_SEQUENCE_LENGTH
                - effective_length
            )

            perturbed_sequence = (
                sequence.copy()
            )

            if neutralised_count > 0:
                perturbed_sequence[
                    :,
                    :neutralised_count,
                    :,
                ] = 0.0

            scaled_prediction = predict_scaled(
                model=model,
                sequence_array=
                    perturbed_sequence,
                static_array=static,
            )

            raw_prediction = inverse_target(
                scaled_prediction,
                target_scaler,
            )

            frame = (
                validation_keys.copy()
            )

            frame.insert(
                0,
                "protocol",
                protocol,
            )

            frame.insert(
                1,
                "model_id",
                model_id,
            )

            frame.insert(
                2,
                "model_display_name",
                MODEL_NAMES[
                    model_id
                ],
            )

            frame.insert(
                3,
                "fold_id",
                fold_id,
            )

            frame.insert(
                4,
                "seed",
                seed,
            )

            frame.insert(
                5,
                "effective_sequence_length",
                effective_length,
            )

            frame.insert(
                6,
                "neutralised_leading_step_count",
                neutralised_count,
            )

            frame[
                "observed_yield_kg_ha"
            ] = observed

            frame[
                "reference_full_length_prediction_kg_ha"
            ] = full_raw_prediction

            frame[
                "effective_length_prediction_kg_ha"
            ] = raw_prediction

            frame[
                "prediction_shift_from_full_length_kg_ha"
            ] = (
                raw_prediction
                - full_raw_prediction
            )

            seed_prediction_frames.append(
                frame
            )

    seed_predictions = pd.concat(
        seed_prediction_frames,
        ignore_index=True,
    )

    replay_audit = pd.DataFrame(
        replay_audit_rows
    )

    ensemble_group_columns = [
        "protocol",
        "model_id",
        "model_display_name",
        "fold_id",
        "effective_sequence_length",
        "neutralised_leading_step_count",
        "district_id",
        "crop_year_start",
        "observed_yield_kg_ha",
    ]

    ensemble_predictions = (
        seed_predictions
        .groupby(
            ensemble_group_columns,
            as_index=False,
        )
        .agg(
            seed_count=(
                "seed",
                "nunique",
            ),
            reference_full_length_prediction_kg_ha=(
                "reference_full_length_prediction_kg_ha",
                "mean",
            ),
            effective_length_prediction_kg_ha=(
                "effective_length_prediction_kg_ha",
                "mean",
            ),
            effective_length_prediction_seed_std_kg_ha=(
                "effective_length_prediction_kg_ha",
                lambda values: float(
                    np.std(
                        np.asarray(
                            values,
                            dtype=float,
                        ),
                        ddof=0,
                    )
                ),
            ),
        )
    )

    ensemble_predictions[
        "reference_absolute_error_kg_ha"
    ] = np.abs(
        ensemble_predictions[
            "reference_full_length_prediction_kg_ha"
        ]
        - ensemble_predictions[
            "observed_yield_kg_ha"
        ]
    )

    ensemble_predictions[
        "effective_absolute_error_kg_ha"
    ] = np.abs(
        ensemble_predictions[
            "effective_length_prediction_kg_ha"
        ]
        - ensemble_predictions[
            "observed_yield_kg_ha"
        ]
    )

    ensemble_predictions[
        "absolute_error_difference_kg_ha"
    ] = (
        ensemble_predictions[
            "effective_absolute_error_kg_ha"
        ]
        - ensemble_predictions[
            "reference_absolute_error_kg_ha"
        ]
    )

    ensemble_predictions[
        "prediction_shift_from_full_length_kg_ha"
    ] = (
        ensemble_predictions[
            "effective_length_prediction_kg_ha"
        ]
        - ensemble_predictions[
            "reference_full_length_prediction_kg_ha"
        ]
    )

    metric_rows = []

    for (
        protocol,
        model_id,
        model_name,
        effective_length,
    ), group in ensemble_predictions.groupby(
        [
            "protocol",
            "model_id",
            "model_display_name",
            "effective_sequence_length",
        ],
        sort=False,
    ):
        metrics = calculate_metrics(
            group[
                "observed_yield_kg_ha"
            ],
            group[
                "effective_length_prediction_kg_ha"
            ],
        )

        metric_rows.append(
            {
                "protocol": protocol,
                "model_id": model_id,
                "model_display_name":
                    model_name,
                "effective_sequence_length":
                    int(effective_length),
                "neutralised_leading_step_count":
                    int(
                        REFERENCE_SEQUENCE_LENGTH
                        - effective_length
                    ),
                "fold_count": int(
                    group[
                        "fold_id"
                    ].nunique()
                ),
                "district_count": int(
                    group[
                        "district_id"
                    ].nunique()
                ),
                "year_count": int(
                    group[
                        "crop_year_start"
                    ].nunique()
                ),
                **metrics,
                "mean_absolute_prediction_shift_kg_ha":
                    float(
                        np.mean(
                            np.abs(
                                group[
                                    "prediction_shift_from_full_length_kg_ha"
                                ]
                            )
                        )
                    ),
                "mean_signed_prediction_shift_kg_ha":
                    float(
                        np.mean(
                            group[
                                "prediction_shift_from_full_length_kg_ha"
                            ]
                        )
                    ),
                "mean_prediction_seed_std_kg_ha":
                    float(
                        np.mean(
                            group[
                                "effective_length_prediction_seed_std_kg_ha"
                            ]
                        )
                    ),
            }
        )

    metric_summary = pd.DataFrame(
        metric_rows
    )

    reference_metrics = (
        metric_summary.loc[
            metric_summary[
                "effective_sequence_length"
            ].eq(
                REFERENCE_SEQUENCE_LENGTH
            ),
            [
                "protocol",
                "model_id",
                "mae_kg_ha",
                "rmse_kg_ha",
                "r2",
            ],
        ]
        .rename(
            columns={
                "mae_kg_ha":
                    "reference_full_length_mae_kg_ha",
                "rmse_kg_ha":
                    "reference_full_length_rmse_kg_ha",
                "r2":
                    "reference_full_length_r2",
            }
        )
    )

    metric_summary = metric_summary.merge(
        reference_metrics,
        on=[
            "protocol",
            "model_id",
        ],
        how="left",
        validate="many_to_one",
    )

    metric_summary[
        "mae_difference_from_full_length_kg_ha"
    ] = (
        metric_summary[
            "mae_kg_ha"
        ]
        - metric_summary[
            "reference_full_length_mae_kg_ha"
        ]
    )

    metric_summary[
        "relative_mae_change_from_full_length_pct"
    ] = (
        metric_summary[
            "mae_difference_from_full_length_kg_ha"
        ]
        / metric_summary[
            "reference_full_length_mae_kg_ha"
        ]
        * 100.0
    )

    statistical_rows = []

    comparison_data = (
        ensemble_predictions.loc[
            ensemble_predictions[
                "effective_sequence_length"
            ].lt(
                REFERENCE_SEQUENCE_LENGTH
            )
        ]
    )

    for (
        protocol,
        model_id,
        model_name,
        effective_length,
    ), group in comparison_data.groupby(
        [
            "protocol",
            "model_id",
            "model_display_name",
            "effective_sequence_length",
        ],
        sort=False,
    ):
        cluster_variable = (
            "district_id"
            if protocol
            == "district_grouped"
            else "crop_year_start"
        )

        cluster_means = (
            group.groupby(
                cluster_variable
            )[
                "absolute_error_difference_kg_ha"
            ]
            .mean()
            .to_numpy(
                dtype=float
            )
        )

        seed_text = (
            f"{protocol}|{model_id}|"
            f"{effective_length}"
        )

        bootstrap_seed = deterministic_seed(
            "bootstrap|" + seed_text
        )

        sign_flip_seed = deterministic_seed(
            "signflip|" + seed_text
        )

        ci_lower, ci_upper = (
            cluster_bootstrap_ci(
                cluster_values=
                    cluster_means,
                seed=bootstrap_seed,
            )
        )

        (
            p_value,
            sign_flip_method,
            sign_flip_draw_count,
        ) = cluster_sign_flip_test(
            cluster_values=
                cluster_means,
            seed=sign_flip_seed,
        )

        statistical_rows.append(
            {
                "protocol": protocol,
                "model_id": model_id,
                "model_display_name":
                    model_name,
                "effective_sequence_length":
                    int(effective_length),
                "neutralised_leading_step_count":
                    int(
                        REFERENCE_SEQUENCE_LENGTH
                        - effective_length
                    ),
                "observation_count": int(
                    len(group)
                ),
                "cluster_variable":
                    cluster_variable,
                "cluster_count": int(
                    len(cluster_means)
                ),
                "mean_absolute_error_difference_kg_ha":
                    float(
                        group[
                            "absolute_error_difference_kg_ha"
                        ].mean()
                    ),
                "difference_ci_lower_kg_ha":
                    ci_lower,
                "difference_ci_upper_kg_ha":
                    ci_upper,
                "sign_flip_p_value":
                    p_value,
                "sign_flip_method":
                    sign_flip_method,
                "sign_flip_draw_count":
                    sign_flip_draw_count,
                "bootstrap_replicates":
                    BOOTSTRAP_REPLICATES,
                "confidence_level":
                    CONFIDENCE_LEVEL,
            }
        )

    statistical_results = pd.DataFrame(
        statistical_rows
    )

    adjusted_frames = []

    for (
        protocol,
        model_id,
    ), group in statistical_results.groupby(
        [
            "protocol",
            "model_id",
        ],
        sort=False,
    ):
        adjusted = group.copy()

        adjusted[
            "sign_flip_holm_p_value"
        ] = holm_adjust(
            adjusted[
                "sign_flip_p_value"
            ].to_numpy(
                dtype=float
            )
        )

        adjusted_frames.append(
            adjusted
        )

    statistical_results = pd.concat(
        adjusted_frames,
        ignore_index=True,
    )

    statistical_results[
        "statistical_conclusion"
    ] = np.select(
        [
            (
                statistical_results[
                    "sign_flip_holm_p_value"
                ]
                < 0.05
            )
            & (
                statistical_results[
                    "mean_absolute_error_difference_kg_ha"
                ]
                > 0
            ),
            (
                statistical_results[
                    "sign_flip_holm_p_value"
                ]
                < 0.05
            )
            & (
                statistical_results[
                    "mean_absolute_error_difference_kg_ha"
                ]
                < 0
            ),
        ],
        [
            "shorter_effective_sequence_statistically_degrades_accuracy",
            "shorter_effective_sequence_statistically_improves_accuracy",
        ],
        default="difference_not_statistically_resolved",
    )

    protocol_order_map = {
        protocol: index
        for index, protocol in enumerate(
            PROTOCOL_ORDER,
            start=1,
        )
    }

    model_order_map = {
        model_id: index
        for index, model_id in enumerate(
            MODEL_ORDER,
            start=1,
        )
    }

    for frame in (
        seed_predictions,
        ensemble_predictions,
        metric_summary,
        statistical_results,
        replay_audit,
    ):
        frame[
            "protocol_order"
        ] = frame[
            "protocol"
        ].map(
            protocol_order_map
        )

        frame[
            "model_order"
        ] = frame[
            "model_id"
        ].map(
            model_order_map
        )

    seed_predictions = (
        seed_predictions.sort_values(
            [
                "protocol_order",
                "model_order",
                "fold_id",
                "seed",
                "effective_sequence_length",
                "crop_year_start",
                "district_id",
            ]
        )
    )

    ensemble_predictions = (
        ensemble_predictions.sort_values(
            [
                "protocol_order",
                "model_order",
                "fold_id",
                "effective_sequence_length",
                "crop_year_start",
                "district_id",
            ]
        )
    )

    metric_summary = (
        metric_summary.sort_values(
            [
                "protocol_order",
                "model_order",
                "effective_sequence_length",
            ]
        )
    )

    statistical_results = (
        statistical_results.sort_values(
            [
                "protocol_order",
                "model_order",
                "effective_sequence_length",
            ]
        )
    )

    replay_audit = (
        replay_audit.sort_values(
            [
                "protocol_order",
                "model_order",
                "fold_id",
                "seed",
            ]
        )
    )

    generated_paths = []

    output_frames = {
        "sequence_length_seed_predictions.csv":
            seed_predictions,
        "sequence_length_ensemble_predictions.csv":
            ensemble_predictions,
        "sequence_length_metric_summary.csv":
            metric_summary,
        "sequence_length_statistical_results.csv":
            statistical_results,
        "sequence_length_reference_replay_audit.csv":
            replay_audit,
    }

    for file_name, frame in output_frames.items():
        path = OUTPUT_DIR / file_name

        frame.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        generated_paths.append(
            path
        )

    caption_rows = []

    for protocol in PROTOCOL_ORDER:
        figure = plt.figure(
            figsize=(9.5, 5.8)
        )

        axis = figure.add_subplot(
            111
        )

        protocol_metrics = (
            metric_summary.loc[
                metric_summary[
                    "protocol"
                ].eq(protocol)
            ]
        )

        for model_id in MODEL_ORDER:
            model_data = (
                protocol_metrics.loc[
                    protocol_metrics[
                        "model_id"
                    ].eq(model_id)
                ]
                .sort_values(
                    "effective_sequence_length"
                )
            )

            axis.plot(
                model_data[
                    "effective_sequence_length"
                ],
                model_data[
                    "mae_kg_ha"
                ],
                marker="o",
                linewidth=2,
                label=MODEL_NAMES[
                    model_id
                ],
            )

        axis.set_xticks(
            SEQUENCE_LENGTHS
        )

        axis.set_xlabel(
            "Effective monthly sequence length"
        )

        axis.set_ylabel(
            "Mean absolute error (kg/ha)"
        )

        protocol_title = (
            protocol.replace(
                "_",
                " ",
            ).title()
        )

        axis.set_title(
            f"{protocol_title} Effective Sequence-Length Sensitivity",
            fontweight="bold",
            pad=12,
        )

        axis.grid(
            alpha=0.25
        )

        axis.legend(
            frameon=False
        )

        figure.tight_layout()

        stem = (
            "Figure_Sequence_Length_Sensitivity_"
            + protocol
        )

        generated_paths.extend(
            save_figure(
                figure,
                stem,
            )
        )

        caption_rows.append(
            {
                "figure_file":
                    f"{stem}.png",
                "caption":
                    (
                        "Mean absolute error of the frozen LSTM "
                        "and Simple RNN as the effective monthly "
                        "sequence length increases from one to seven "
                        "steps. Earlier steps were neutralised at "
                        "their training-centred scaled value; models "
                        "were not retrained."
                    ),
            }
        )

    captions = pd.DataFrame(
        caption_rows
    )

    captions_path = (
        OUTPUT_DIR
        / "sequence_length_figure_captions.csv"
    )

    captions.to_csv(
        captions_path,
        index=False,
        encoding="utf-8-sig",
    )

    generated_paths.append(
        captions_path
    )

    expected_seed_prediction_count = int(
        sum(
            len(
                bundle[
                    "observed"
                ]
            )
            for bundle
            in bundle_cache.values()
        )
        * len(ALLOWED_MODELS)
        * 3
        * len(SEQUENCE_LENGTHS)
    )

    expected_ensemble_prediction_count = int(
        sum(
            len(
                bundle[
                    "observed"
                ]
            )
            for bundle
            in bundle_cache.values()
        )
        * len(ALLOWED_MODELS)
        * len(SEQUENCE_LENGTHS)
    )

    check_rows = [
        {
            "check":
                "Fourteen development preprocessing bundles resolved",
            "passed":
                len(bundle_cache) == 14,
            "details":
                f"Observed {len(bundle_cache)}; expected 14",
        },
        {
            "check":
                "All eighty-four recurrent checkpoints evaluated",
            "passed":
                len(replay_audit) == 84,
            "details":
                f"Observed {len(replay_audit)}; expected 84",
        },
        {
            "check":
                "Every frozen full-length replay matched saved predictions",
            "passed":
                bool(
                    replay_audit[
                        "replay_within_tolerance"
                    ].all()
                ),
            "details":
                (
                    "Maximum raw replay error "
                    f"{replay_audit['maximum_raw_replay_error_kg_ha'].max():.12g}; "
                    "maximum scaled replay error "
                    f"{replay_audit['maximum_scaled_replay_error'].max():.12g}"
                ),
        },
        {
            "check":
                "Expected seed-level prediction count generated",
            "passed":
                len(seed_predictions)
                == expected_seed_prediction_count,
            "details":
                (
                    f"Observed {len(seed_predictions)}; "
                    f"expected {expected_seed_prediction_count}"
                ),
        },
        {
            "check":
                "Expected ensemble prediction count generated",
            "passed":
                len(ensemble_predictions)
                == expected_ensemble_prediction_count,
            "details":
                (
                    f"Observed {len(ensemble_predictions)}; "
                    f"expected {expected_ensemble_prediction_count}"
                ),
        },
        {
            "check":
                "All effective sequence lengths one through seven evaluated",
            "passed":
                set(
                    metric_summary[
                        "effective_sequence_length"
                    ].unique()
                )
                == set(SEQUENCE_LENGTHS),
            "details":
                (
                    "Observed "
                    + str(
                        sorted(
                            metric_summary[
                                "effective_sequence_length"
                            ].unique()
                        )
                    )
                ),
        },
        {
            "check":
                "All generated predictions and metrics are finite",
            "passed":
                bool(
                    np.isfinite(
                        seed_predictions[
                            [
                                "observed_yield_kg_ha",
                                "reference_full_length_prediction_kg_ha",
                                "effective_length_prediction_kg_ha",
                            ]
                        ].to_numpy(
                            dtype=float
                        )
                    ).all()
                    and np.isfinite(
                        metric_summary[
                            [
                                "mae_kg_ha",
                                "rmse_kg_ha",
                                "r2",
                                "mape_pct",
                            ]
                        ].to_numpy(
                            dtype=float
                        )
                    ).all()
                ),
            "details":
                "Prediction and metric finiteness verified",
        },
        {
            "check":
                "No locked final-test year was evaluated",
            "passed":
                recurrent_replay[
                    "locked_year_overlap_count"
                ].fillna(0).astype(int).sum()
                == 0,
            "details":
                "Locked-year overlap count 0",
        },
        {
            "check":
                "No model was trained",
            "passed": True,
            "details":
                "Only frozen recurrent checkpoints were evaluated",
        },
        {
            "check":
                "No final-test prediction was generated",
            "passed": True,
            "details":
                "Development-validation bundles only",
        },
    ]

    execution_checks = pd.DataFrame(
        check_rows
    )

    if not execution_checks[
        "passed"
    ].all():
        failed = execution_checks.loc[
            ~execution_checks[
                "passed"
            ],
            "check",
        ].tolist()

        raise RuntimeError(
            "Execution checks failed: "
            + "; ".join(failed)
        )

    checks_path = (
        OUTPUT_DIR
        / "sequence_length_execution_checks.csv"
    )

    execution_checks.to_csv(
        checks_path,
        index=False,
        encoding="utf-8-sig",
    )

    generated_paths.append(
        checks_path
    )

    source_manifest_rows = []

    for path in sorted(
        source_paths,
        key=lambda item: str(item),
    ):
        source_manifest_rows.append(
            {
                "file_name":
                    Path(path).name,
                "full_path":
                    str(path),
                "file_size_bytes":
                    Path(path).stat().st_size,
                "sha256":
                    sha256(path),
            }
        )

    source_manifest = pd.DataFrame(
        source_manifest_rows
    )

    source_manifest_path = (
        OUTPUT_DIR
        / "sequence_length_source_manifest.csv"
    )

    source_manifest.to_csv(
        source_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    generated_paths.append(
        source_manifest_path
    )

    metadata = {
        "generated_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "analysis_type":
            "frozen_model_effective_sequence_length_sensitivity",
        "effective_sequence_lengths":
            SEQUENCE_LENGTHS,
        "reference_sequence_length":
            REFERENCE_SEQUENCE_LENGTH,
        "neutralisation_rule":
            (
                "For effective length k, the first 7-k "
                "scaled monthly time steps were replaced "
                "with zero, the training-centred neutral value."
            ),
        "models":
            MODEL_ORDER,
        "protocols":
            PROTOCOL_ORDER,
        "development_fold_count":
            len(bundle_cache),
        "checkpoint_count":
            len(replay_audit),
        "seed_prediction_count":
            len(seed_predictions),
        "ensemble_prediction_count":
            len(ensemble_predictions),
        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,
        "sign_flip_monte_carlo_draws":
            MONTE_CARLO_SIGN_FLIPS,
        "confidence_level":
            CONFIDENCE_LEVEL,
        "reference_replay_tolerance":
            REFERENCE_TOLERANCE,
        "maximum_reference_raw_replay_error_kg_ha":
            float(
                replay_audit[
                    "maximum_raw_replay_error_kg_ha"
                ].max()
            ),
        "maximum_reference_scaled_replay_error":
            float(
                replay_audit[
                    "maximum_scaled_replay_error"
                ].max()
            ),
        "interpretation_caveat":
            (
                "This is an effective sequence-length sensitivity "
                "analysis of frozen seven-step architectures. "
                "Earlier time steps were neutralised while retaining "
                "the original tensor shape. The models were not "
                "retrained on physically shorter sequences."
            ),
        "safety": {
            "final_test_accessed": False,
            "final_test_predictions_generated": False,
            "final_test_metrics_calculated": False,
            "models_retrained": False,
            "frozen_notebook_modified": False,
            "source_files_modified": False,
        },
        "generation_script":
            str(SCRIPT_PATH),
        "generation_script_sha256":
            sha256(SCRIPT_PATH),
    }

    metadata_path = (
        OUTPUT_DIR
        / "sequence_length_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    generated_paths.append(
        metadata_path
    )

    notes = """EFFECTIVE SEQUENCE-LENGTH SENSITIVITY REPORTING NOTES

1. The analysis uses development-validation data only.
2. No final-test arrays, predictions, or metrics were accessed.
3. No model was retrained.
4. Frozen LSTM and Simple RNN checkpoints were replayed.
5. Every full seven-step replay was verified against its saved prediction.
6. Effective sequence lengths from one to seven were evaluated.
7. For length k, the earliest 7-k time steps were neutralised to zero in scaled space.
8. Zero represents the training-centred neutral value after StandardScaler transformation.
9. Static historical-yield predictors were retained unchanged.
10. This analysis measures frozen-model sensitivity to the amount of active temporal context.
11. It is not equivalent to retraining separate architectures on physically shorter sequences.
12. Statistical comparisons use paired absolute-error differences relative to the full seven-step sequence.
13. District-grouped inference clusters by district.
14. Rolling-origin inference clusters by crop year.
15. Holm adjustment is applied across the six shortened-length comparisons within each protocol-model combination.
"""

    notes_path = (
        OUTPUT_DIR
        / "sequence_length_reporting_notes.txt"
    )

    notes_path.write_text(
        notes,
        encoding="utf-8",
    )

    generated_paths.append(
        notes_path
    )

    output_manifest_rows = []

    for path in sorted(
        generated_paths,
        key=lambda item: item.name,
    ):
        output_manifest_rows.append(
            {
                "file_name":
                    path.name,
                "full_path":
                    str(path),
                "file_size_bytes":
                    path.stat().st_size,
                "sha256":
                    sha256(path),
            }
        )

    output_manifest = pd.DataFrame(
        output_manifest_rows
    )

    output_manifest_path = (
        OUTPUT_DIR
        / "sequence_length_output_sha256_manifest.csv"
    )

    output_manifest.to_csv(
        output_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 100)
    print("SEQUENCE-LENGTH SENSITIVITY PACKAGE COMPLETE")
    print("=" * 100)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Development folds: {len(bundle_cache)}")
    print(f"Recurrent checkpoints: {len(replay_audit)}")
    print(
        f"Seed-level predictions: "
        f"{len(seed_predictions):,}"
    )
    print(
        f"Ensemble predictions: "
        f"{len(ensemble_predictions):,}"
    )
    print(
        "Maximum reference raw replay error: "
        f"{replay_audit['maximum_raw_replay_error_kg_ha'].max():.12g}"
    )
    print(
        "Maximum reference scaled replay error: "
        f"{replay_audit['maximum_scaled_replay_error'].max():.12g}"
    )
    print("Models retrained: False")
    print("Final-test data accessed: False")
    print("")
    print("MAE BY EFFECTIVE SEQUENCE LENGTH")
    print(
        metric_summary[
            [
                "protocol",
                "model_display_name",
                "effective_sequence_length",
                "mae_kg_ha",
                "mae_difference_from_full_length_kg_ha",
                "relative_mae_change_from_full_length_pct",
            ]
        ].to_string(
            index=False
        )
    )
    print("")
    print("STATISTICAL COMPARISONS")
    print(
        statistical_results[
            [
                "protocol",
                "model_display_name",
                "effective_sequence_length",
                "mean_absolute_error_difference_kg_ha",
                "difference_ci_lower_kg_ha",
                "difference_ci_upper_kg_ha",
                "sign_flip_holm_p_value",
                "statistical_conclusion",
            ]
        ].to_string(
            index=False
        )
    )
    print("")
    print("GENERATED FILES")

    for path in (
        generated_paths
        + [output_manifest_path]
    ):
        print(
            f"- {path.name} "
            f"({path.stat().st_size:,} bytes)"
        )


if __name__ == "__main__":
    main()
