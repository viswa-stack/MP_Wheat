from pathlib import Path
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd


ROOT = Path(
    r"D:\MP_Wheat_Project\mp_wheat_project_pipeline"
    r"\validation\ablation"
    r"\unified_synthesis_and_robustness_protocol"
    r"\development_set_robustness"
    r"\frozen_model_perturbation_preflight"
    r"\exact_artifact_resolution"
    r"\dynamic_recurrent_architecture_reconstruction_v3"
)

AUDIT_PATH = ROOT / "reconstruction_v3_sample_context_audit.csv"

ALLOWED_MODELS = {
    "deep_learning__lstm",
    "deep_learning__simple_rnn",
}


def describe(value, depth=0, max_depth=3):
    prefix = "  " * depth

    if isinstance(value, np.ndarray):
        print(
            f"{prefix}type=ndarray, "
            f"shape={value.shape}, dtype={value.dtype}"
        )
        return

    if isinstance(value, pd.DataFrame):
        print(
            f"{prefix}type=DataFrame, "
            f"shape={value.shape}, "
            f"columns={list(value.columns)}"
        )
        return

    if isinstance(value, pd.Series):
        print(
            f"{prefix}type=Series, "
            f"shape={value.shape}, "
            f"dtype={value.dtype}, "
            f"name={value.name}"
        )
        return

    if isinstance(value, Mapping):
        keys = list(value.keys())

        print(
            f"{prefix}type={type(value).__name__}, "
            f"key_count={len(keys)}"
        )

        if depth >= max_depth:
            return

        for key in keys:
            item = value[key]

            print(
                f"{prefix}- key={key!r}:",
                end=" "
            )

            if isinstance(
                item,
                (
                    np.ndarray,
                    pd.DataFrame,
                    pd.Series,
                    Mapping,
                    list,
                    tuple,
                ),
            ):
                print()
                describe(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            else:
                print(
                    f"type={type(item).__name__}"
                )

        return

    if isinstance(value, (list, tuple)):
        print(
            f"{prefix}type={type(value).__name__}, "
            f"length={len(value)}"
        )

        if depth < max_depth and len(value) > 0:
            print(
                f"{prefix}first_item:"
            )
            describe(
                value[0],
                depth=depth + 1,
                max_depth=max_depth,
            )

        return

    print(
        f"{prefix}type={type(value).__name__}"
    )


def main():
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(AUDIT_PATH)

    audit = pd.read_csv(AUDIT_PATH)

    recurrent = (
        audit.loc[
            audit["model_id"].isin(ALLOWED_MODELS),
            [
                "protocol",
                "model_id",
                "fold_id",
                "seed",
                "run_artifact_path",
            ],
        ]
        .drop_duplicates(
            [
                "protocol",
                "model_id",
            ]
        )
        .sort_values(
            [
                "protocol",
                "model_id",
            ]
        )
    )

    print("=" * 100)
    print("RECURRENT DEVELOPMENT RUN-ARTIFACT STRUCTURE AUDIT")
    print("=" * 100)
    print(f"Artifact count: {len(recurrent)}")
    print("Final-test paths permitted: False")

    for row in recurrent.itertuples(index=False):
        path = Path(row.run_artifact_path)

        print()
        print("=" * 100)
        print(f"PROTOCOL: {row.protocol}")
        print(f"MODEL: {row.model_id}")
        print(f"FOLD: {row.fold_id}")
        print(f"SEED: {row.seed}")
        print(f"PATH: {path}")
        print("=" * 100)

        path_text = str(path).lower()

        if "final_test" in path_text or "test_execution" in path_text:
            print("BLOCKED: path appears related to final-test execution")
            continue

        if not path.is_file():
            print("FILE NOT FOUND")
            continue

        try:
            artifact = joblib.load(path)
        except Exception as error:
            print(
                f"LOAD ERROR: "
                f"{type(error).__name__}: {error}"
            )
            continue

        describe(
            artifact,
            depth=0,
            max_depth=3,
        )


if __name__ == "__main__":
    main()
