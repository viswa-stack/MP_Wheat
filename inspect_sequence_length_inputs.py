from pathlib import Path

import joblib
import torch


FILES = {
    "district_bundle": Path(
        r"D:\MP_Wheat_Project\mp_wheat_project_pipeline"
        r"\validation\district_grouped\deep_learning_models"
        r"\fold_preprocessing\bundles"
        r"\district_group_fold_01.joblib"
    ),
    "rolling_bundle": Path(
        r"D:\MP_Wheat_Project\mp_wheat_project_pipeline"
        r"\validation\rolling_origin\deep_learning_models"
        r"\fold_preprocessing\bundles"
        r"\rolling_origin_2007__deep_preprocessing.joblib"
    ),
    "lstm_checkpoint": Path(
        r"D:\MP_Wheat_Project\mp_wheat_project_pipeline"
        r"\validation\district_grouped\deep_learning_models"
        r"\spatial_validation\model_checkpoints"
        r"\district_group_fold_01__lstm__seed_20260721.pt"
    ),
    "rnn_checkpoint": Path(
        r"D:\MP_Wheat_Project\mp_wheat_project_pipeline"
        r"\validation\district_grouped\deep_learning_models"
        r"\spatial_validation\model_checkpoints"
        r"\district_group_fold_01__simple_rnn__seed_20260721.pt"
    ),
}


def describe_bundle(path):
    print("=" * 100)
    print(f"BUNDLE: {path}")
    print("=" * 100)

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        print(f"Top-level type: {type(bundle).__name__}")
        return

    print(f"Key count: {len(bundle)}")

    for key in sorted(bundle):
        value = bundle[key]

        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)

        details = [f"type={type(value).__name__}"]

        if shape is not None:
            details.append(f"shape={tuple(shape)}")

        if dtype is not None:
            details.append(f"dtype={dtype}")

        if isinstance(value, (list, tuple)):
            details.append(f"length={len(value)}")

        print(f"- {key}: " + ", ".join(details))


def load_checkpoint(path):
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


def find_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "state_dict",
            "model",
        ):
            candidate = checkpoint.get(key)

            if isinstance(candidate, dict):
                return candidate, key

        if checkpoint and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        ):
            return checkpoint, "<top_level>"

    raise ValueError(
        "No state dictionary was found."
    )


def describe_checkpoint(path):
    print("=" * 100)
    print(f"CHECKPOINT: {path}")
    print("=" * 100)

    checkpoint = load_checkpoint(path)

    if isinstance(checkpoint, dict):
        print(
            "Top-level keys: "
            + ", ".join(
                sorted(str(key) for key in checkpoint)
            )
        )
    else:
        print(
            f"Top-level type: {type(checkpoint).__name__}"
        )

    state_dict, state_key = find_state_dict(
        checkpoint
    )

    print(f"State-dictionary location: {state_key}")
    print(f"Tensor count: {len(state_dict)}")

    total_parameters = 0

    for name, tensor in state_dict.items():
        shape = tuple(tensor.shape)
        count = int(tensor.numel())
        total_parameters += count

        print(
            f"- {name}: shape={shape}, "
            f"dtype={tensor.dtype}, "
            f"parameters={count}"
        )

    print(f"Total parameter count: {total_parameters}")


def main():
    print("=" * 100)
    print("SEQUENCE-LENGTH INPUT AND CHECKPOINT SCHEMA AUDIT")
    print("=" * 100)
    print("Final-test access permitted: False")

    for name, path in FILES.items():
        if (
            "final_test" in str(path).lower()
            or "test_execution" in str(path).lower()
        ):
            raise RuntimeError(
                f"Blocked unsafe path: {path}"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"{name} not found: {path}"
            )

    describe_bundle(FILES["district_bundle"])
    describe_bundle(FILES["rolling_bundle"])

    describe_checkpoint(
        FILES["lstm_checkpoint"]
    )

    describe_checkpoint(
        FILES["rnn_checkpoint"]
    )


if __name__ == "__main__":
    main()
