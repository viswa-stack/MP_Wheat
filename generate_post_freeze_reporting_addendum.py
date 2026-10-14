from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import shutil
import zipfile

import pandas as pd


ROOT = Path(r"D:\MP_Wheat_Project")
PIPELINE = ROOT / "mp_wheat_project_pipeline"

OUTPUT_DIR = (
    PIPELINE
    / "reporting"
    / "post_freeze_reporting_addendum"
)

LEARNING_DIR = (
    PIPELINE
    / "reporting"
    / "figures"
    / "learning_curves"
)

SHAP_DIR = (
    PIPELINE
    / "explainability"
    / "tree_shap"
)

EFFICIENCY_DIR = (
    PIPELINE
    / "reporting"
    / "computational_efficiency"
)

GEE_STAGE_DIR = (
    ROOT
    / "gee_reproducibility"
    / "GEE_Reproducibility_Addendum_v2"
)

GEE_ARCHIVE = (
    ROOT
    / "GEE_Reproducibility_Addendum_v2_20260725.zip"
)

GEE_ARCHIVE_HASH = Path(
    str(GEE_ARCHIVE) + ".sha256"
)

FROZEN_NOTEBOOK = ROOT / "Wheat.ipynb"

FINAL_CHECKPOINT = (
    PIPELINE
    / "checkpoints"
    / "final_reproducibility_freeze"
    / "mp_wheat_final_master_checkpoint_20260724_061332_055712.cloudpickle"
)

EXPECTED_NOTEBOOK_SHA256 = (
    "64890299541d4f67d03070dd480bb2f75d399655ddc9493922e4717b98c498fb"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "b9ad99f248c812718849c06280f12d4a65c7aeb1067e44bd6c0d662d9b28f046"
)

EXPECTED_GEE_ARCHIVE_SHA256 = (
    "edb0c14e262505a8e2acc6ae7ec106c085904c5b983de8573f871715520a3e3d"
)


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


def require_directory(path):
    if not path.is_dir():
        raise FileNotFoundError(
            f"Required directory not found: {path}"
        )


def inventory_directory(
    directory,
    component,
):
    rows = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        rows.append(
            {
                "component": component,
                "file_name": path.name,
                "relative_path": str(
                    path.relative_to(ROOT)
                ),
                "full_path": str(path),
                "file_extension": path.suffix.lower(),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    return rows


def copy_key_file(
    source,
    destination_directory,
):
    require_file(source)

    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        destination_directory
        / source.name
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def main():
    require_directory(LEARNING_DIR)
    require_directory(SHAP_DIR)
    require_directory(EFFICIENCY_DIR)
    require_directory(GEE_STAGE_DIR)

    require_file(GEE_ARCHIVE)
    require_file(GEE_ARCHIVE_HASH)
    require_file(FROZEN_NOTEBOOK)
    require_file(FINAL_CHECKPOINT)

    notebook_hash = sha256(
        FROZEN_NOTEBOOK
    )

    checkpoint_hash = sha256(
        FINAL_CHECKPOINT
    )

    gee_archive_hash = sha256(
        GEE_ARCHIVE
    )

    if notebook_hash != EXPECTED_NOTEBOOK_SHA256:
        raise RuntimeError(
            "Frozen notebook SHA-256 no longer matches."
        )

    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Final checkpoint SHA-256 no longer matches."
        )

    if gee_archive_hash != EXPECTED_GEE_ARCHIVE_SHA256:
        raise RuntimeError(
            "GEE addendum archive SHA-256 no longer matches."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_dir = (
        OUTPUT_DIR
        / "component_manifests"
    )

    evidence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_rows = []

    inventory_rows.extend(
        inventory_directory(
            LEARNING_DIR,
            "learning_curves",
        )
    )

    inventory_rows.extend(
        inventory_directory(
            SHAP_DIR,
            "tree_shap",
        )
    )

    inventory_rows.extend(
        inventory_directory(
            EFFICIENCY_DIR,
            "computational_efficiency",
        )
    )

    inventory_rows.extend(
        inventory_directory(
            GEE_STAGE_DIR,
            "gee_reproducibility_addendum_v2",
        )
    )

    inventory_rows.append(
        {
            "component": "gee_archive",
            "file_name": GEE_ARCHIVE.name,
            "relative_path": str(
                GEE_ARCHIVE.relative_to(ROOT)
            ),
            "full_path": str(GEE_ARCHIVE),
            "file_extension": ".zip",
            "file_size_bytes": GEE_ARCHIVE.stat().st_size,
            "sha256": gee_archive_hash,
        }
    )

    inventory = pd.DataFrame(
        inventory_rows
    ).sort_values(
        [
            "component",
            "relative_path",
        ]
    ).reset_index(drop=True)

    master_inventory_path = (
        OUTPUT_DIR
        / "post_freeze_master_artifact_inventory.csv"
    )

    inventory.to_csv(
        master_inventory_path,
        index=False,
        encoding="utf-8-sig",
    )

    component_summary = (
        inventory
        .groupby(
            "component",
            as_index=False,
        )
        .agg(
            file_count=(
                "file_name",
                "count",
            ),
            total_size_bytes=(
                "file_size_bytes",
                "sum",
            ),
        )
    )

    component_summary[
        "total_size_mib"
    ] = (
        component_summary[
            "total_size_bytes"
        ]
        / (1024 ** 2)
    )

    component_summary_path = (
        OUTPUT_DIR
        / "post_freeze_component_summary.csv"
    )

    component_summary.to_csv(
        component_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    key_manifest_sources = [
        (
            LEARNING_DIR
            / "learning_curve_output_sha256_manifest.csv"
        ),
        (
            SHAP_DIR
            / "tree_shap_output_sha256_manifest.csv"
        ),
        (
            EFFICIENCY_DIR
            / "computational_efficiency_output_sha256_manifest.csv"
        ),
        (
            GEE_STAGE_DIR
            / "GEE_ADDENDUM_SHA256_MANIFEST.csv"
        ),
        GEE_ARCHIVE_HASH,
    ]

    copied_manifests = []

    for source in key_manifest_sources:
        copied_manifests.append(
            copy_key_file(
                source,
                evidence_dir,
            )
        )

    metadata = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "package_name":
            "MP Wheat Post-Freeze Reporting Addendum",
        "purpose":
            (
                "Consolidated reporting inventory for learning curves, "
                "tree-model SHAP explainability, Google Earth Engine "
                "publication maps and computational-efficiency evidence."
            ),
        "safety": {
            "final_test_status": "CONSUMED_AND_LOCKED",
            "final_test_accessed": False,
            "final_test_rerun": False,
            "models_retrained": False,
            "frozen_notebook_modified": False,
            "final_checkpoint_modified": False,
            "original_gee_archive_modified": False,
        },
        "frozen_artifacts": {
            "notebook_path": str(
                FROZEN_NOTEBOOK
            ),
            "notebook_sha256":
                notebook_hash,
            "final_checkpoint_path": str(
                FINAL_CHECKPOINT
            ),
            "final_checkpoint_sha256":
                checkpoint_hash,
        },
        "components": {
            "learning_curves": {
                "directory": str(
                    LEARNING_DIR
                ),
                "file_count": int(
                    (
                        inventory[
                            "component"
                        ]
                        == "learning_curves"
                    ).sum()
                ),
            },
            "tree_shap": {
                "directory": str(
                    SHAP_DIR
                ),
                "file_count": int(
                    (
                        inventory[
                            "component"
                        ]
                        == "tree_shap"
                    ).sum()
                ),
                "explained_split":
                    "validation",
                "explained_observation_count":
                    114,
                "final_test_used":
                    False,
            },
            "computational_efficiency": {
                "directory": str(
                    EFFICIENCY_DIR
                ),
                "file_count": int(
                    (
                        inventory[
                            "component"
                        ]
                        == "computational_efficiency"
                    ).sum()
                ),
                "model_count": 5,
                "protocol":
                    "rolling_origin_validation",
            },
            "gee_publication_maps": {
                "stage_directory": str(
                    GEE_STAGE_DIR
                ),
                "archive_path": str(
                    GEE_ARCHIVE
                ),
                "archive_sha256":
                    gee_archive_hash,
                "visualised_map_count":
                    30,
                "expected_visualised_map_count":
                    30,
                "optional_raw_map_count":
                    4,
                "expected_optional_raw_map_count":
                    6,
                "missing_optional_raw_years": [
                    "2019_2020",
                    "2020_2021",
                ],
            },
        },
        "master_inventory_path": str(
            master_inventory_path
        ),
        "component_summary_path": str(
            component_summary_path
        ),
    }

    metadata_path = (
        OUTPUT_DIR
        / "post_freeze_reporting_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""# MP Wheat Post-Freeze Reporting Addendum

## Purpose

This addendum consolidates the reporting evidence generated after the primary modelling notebook and final checkpoint were frozen.

## Included reporting components

1. Learning-curve figures and supporting data
2. Tree-model SHAP summary and dependence analysis
3. GEE publication-map reproducibility addendum
4. Computational-efficiency table and environment metadata

## Safety status

- Final test: consumed and locked
- Final-test data accessed during addendum generation: no
- Models retrained: no
- Frozen notebook modified: no
- Final checkpoint modified: no
- Original GEE archive modified: no

## Frozen artifact verification

- `Wheat.ipynb` SHA-256:
  `{notebook_hash}`
- Final master checkpoint SHA-256:
  `{checkpoint_hash}`

## GEE publication-map coverage

- Required visualised maps: 30/30
- Optional raw multiband maps: 4/6
- Missing optional raw years: 2019-2020 and 2020-2021
- GEE addendum archive SHA-256:
  `{gee_archive_hash}`

## Master inventory

`post_freeze_master_artifact_inventory.csv` records the file path, size and SHA-256 hash of every indexed reporting artifact.

## Component summary

`post_freeze_component_summary.csv` reports the number and total size of files in each reporting component.
"""

    readme_path = (
        OUTPUT_DIR
        / "README.md"
    )

    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    package_files = [
        master_inventory_path,
        component_summary_path,
        metadata_path,
        readme_path,
        *copied_manifests,
    ]

    output_manifest_rows = []

    for path in sorted(
        package_files,
        key=lambda item: item.name,
    ):
        output_manifest_rows.append(
            {
                "file_name": path.name,
                "relative_path": str(
                    path.relative_to(
                        OUTPUT_DIR
                    )
                ),
                "file_size_bytes":
                    path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    output_manifest = pd.DataFrame(
        output_manifest_rows
    )

    output_manifest_path = (
        OUTPUT_DIR
        / "post_freeze_addendum_sha256_manifest.csv"
    )

    output_manifest.to_csv(
        output_manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    archive_path = (
        ROOT
        / "MP_Wheat_Post_Freeze_Reporting_Addendum_20260725.zip"
    )

    archive_hash_path = Path(
        str(archive_path) + ".sha256"
    )

    if archive_path.exists():
        archive_path.unlink()

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(
            OUTPUT_DIR.rglob("*")
        ):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(
                        path.relative_to(
                            OUTPUT_DIR.parent
                        )
                    ),
                )

    archive_hash = sha256(
        archive_path
    )

    archive_hash_path.write_text(
        f"{archive_hash}  {archive_path.name}\n",
        encoding="ascii",
    )

    print("=" * 92)
    print("POST-FREEZE REPORTING ADDENDUM COMPLETE")
    print("=" * 92)
    print(f"Inventory rows: {len(inventory):,}")
    print("")
    print(component_summary.to_string(index=False))
    print("")
    print(f"Frozen notebook hash verified: True")
    print(f"Final checkpoint hash verified: True")
    print(f"GEE addendum archive hash verified: True")
    print(f"Final-test data accessed: False")
    print(f"Models retrained: False")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Archive: {archive_path}")
    print(f"Archive SHA256: {archive_hash}")
    print(f"Hash sidecar: {archive_hash_path}")


if __name__ == "__main__":
    main()
