
"""Phase 6, Experiment 01: Evaluation protocol verification.

Reconstructs the Phase 5 project-aware folds.
Checks development data, labels and leakage separation.
Compares fold assignments with the saved Phase 5 manifest.

Does not train a model or load the original test partition.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

PHASE4 = THESIS / "promise_advanced_models_phase_4"
PHASE5 = THESIS / "promise_improved_model_phase_5"

MANIFEST = (
    PHASE5 / "outputs" / "tables"
    / "01_fixed_fold_assignments.csv"
)

REPORT = ROOT / "outputs" / "reports" / "01_protocol_audit.json"
FOLD_OUTPUT = ROOT / "outputs" / "tables" / "01_verified_folds.csv"

SEED = 42

EXPECTED_LABELS = {
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
}


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "phase4_reference", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold_column(manifest):
    """Find a single saved held-out fold column."""

    candidates = [
        "HeldOutFold", "held_out_fold",
        "CVFold", "cv_fold", "FoldID",
        "fold_id", "Fold", "fold",
    ]

    for name in candidates:
        if name not in manifest.columns:
            continue

        values = pd.to_numeric(
            manifest[name], errors="coerce"
        )

        if values.isna().any():
            continue

        unique = set(values.astype(int))

        if unique in ({0, 1, 2}, {1, 2, 3}):
            return name, values.astype(int)

    raise ValueError(
        "Cannot identify held-out fold column in "
        f"Phase 5 manifest. Columns: "
        f"{manifest.columns.tolist()}"
    )


def main():
    print("\nPHASE 6: EXPERIMENT 01")
    print("EVALUATION PROTOCOL VERIFICATION")
    print("=" * 60)

    reference = load_module(
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py"
    )

    data = reference.load_development_data().copy()

    required = {
        "RecordID", "ProjectID",
        "NormalizedText", "_class_"
    }

    assert required.issubset(data.columns)

    assert len(data) == 3127, (
        f"Unexpected record count: {len(data)}"
    )

    assert data["RecordID"].is_unique
    assert data["NormalizedText"].notna().all()
    assert data["NormalizedText"].astype(str).str.strip().ne("").all()

    labels = data["_class_"].astype(str).to_numpy()
    groups = np.asarray(
        reference.create_connected_groups(data)
    )

    assert len(groups) == len(data)
    assert set(labels) == EXPECTED_LABELS
    assert len(pd.unique(groups)) == 66

    print("Development records:", len(data))
    print("Requirement classes:", len(set(labels)))
    print("Connected groups:", len(pd.unique(groups)))
    print("Dataset integrity: PASSED")

    splitter = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    assignments = np.full(
        len(data), -1, dtype=int
    )

    fold_statistics = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(data, labels, groups),
        start=1,
    ):
        train = data.iloc[train_idx]
        valid = data.iloc[val_idx]

        train_projects = set(
            train["ProjectID"].astype(str)
        )
        valid_projects = set(
            valid["ProjectID"].astype(str)
        )

        train_texts = set(
            train["NormalizedText"].astype(str)
        )
        valid_texts = set(
            valid["NormalizedText"].astype(str)
        )

        train_groups = set(groups[train_idx])
        valid_groups = set(groups[val_idx])

        assert train_projects.isdisjoint(valid_projects)
        assert train_texts.isdisjoint(valid_texts)
        assert train_groups.isdisjoint(valid_groups)

        assert np.all(assignments[val_idx] == -1)
        assignments[val_idx] = fold

        stats = {
            "fold": fold,
            "training_records": len(train_idx),
            "held_out_records": len(val_idx),
            "project_overlap": 0,
            "normalized_text_overlap": 0,
            "connected_group_overlap": 0,
        }

        fold_statistics.append(stats)

        print(
            f"Fold {fold}: train={len(train_idx)}, "
            f"held_out={len(val_idx)}, "
            "leakage checks=PASSED"
        )

    assert np.all(assignments > 0)

    print("\nAll records assigned exactly once: PASSED")

    # Verify the reconstructed assignments against
    # the saved Phase 5 validation-fold manifest.

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"RecordID": str},
        keep_default_na=False,
    )

    assert len(manifest) == len(data)
    assert "RecordID" in manifest.columns
    assert manifest["RecordID"].is_unique

    column, saved_folds = fold_column(manifest)

    reconstructed = pd.DataFrame({
        "RecordID": data["RecordID"].astype(str),
        "Phase6Fold": assignments,
    })

    comparison = reconstructed.merge(
        pd.DataFrame({
            "RecordID": manifest["RecordID"],
            "SavedFold": saved_folds,
        }),
        on="RecordID",
        validate="one_to_one",
    )

    assert len(comparison) == 3127

    saved = comparison["SavedFold"].to_numpy()
    rebuilt = comparison["Phase6Fold"].to_numpy()

    if set(saved) == {0, 1, 2}:
        saved = saved + 1

    assert np.array_equal(saved, rebuilt), (
        "Phase 6 fold assignments differ from Phase 5"
    )

    print("Phase 5 manifest column:", column)
    print("Exact Phase 5 fold agreement: PASSED")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    FOLD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    reconstructed["ProjectID"] = (
        data["ProjectID"].astype(str).to_numpy()
    )
    reconstructed["ActualClass"] = labels

    reconstructed.to_csv(
        FOLD_OUTPUT, index=False
    )

    report = {
        "phase": 6,
        "experiment": 1,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_records": len(data),
        "classes": len(EXPECTED_LABELS),
        "connected_groups": len(pd.unique(groups)),
        "cv_method": "StratifiedGroupKFold",
        "n_splits": 3,
        "random_seed": SEED,
        "fold_statistics": fold_statistics,
        "matches_phase5_folds": True,
        "phase5_manifest_sha256": checksum(MANIFEST),
        "test_partition_accessed": False,
        "phase5_reference_mean_cv_accuracy": 0.6551,
        "phase5_reference_mean_cv_macro_f1": 0.3200,
        "note": (
            "Evaluation protocol audit only; "
            "no new model performance was measured."
        ),
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", FOLD_OUTPUT)
    print("Saved:", REPORT)

    print("\nEXPERIMENT 01 COMPLETED")
    print("Project-aware fold validation passed.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()

