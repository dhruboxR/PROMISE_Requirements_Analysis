
"""PROMISE+ Phase 5, Experiment 01.

Verify development data, connected groups, and exact
project-aware folds from Phase 4.

Does not train any model or access the test partition.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

PHASE5 = Path(__file__).resolve().parents[1]
THESIS = PHASE5.parent
PHASE4 = THESIS / "promise_advanced_models_phase_4"

SOURCE_SCRIPT = (
    PHASE4 / "scripts"
    / "01_group_aware_feature_comparison.py"
)

SOURCE_METRICS = (
    PHASE4 / "outputs" / "metrics"
    / "01_feature_comparison_folds.csv"
)

OUTPUT_TABLES = PHASE5 / "outputs" / "tables"
OUTPUT_REPORTS = PHASE5 / "outputs" / "reports"

SEED = 42

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def load_phase4_module():
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(SOURCE_SCRIPT)

    spec = importlib.util.spec_from_file_location(
        "phase4_original_cv", SOURCE_SCRIPT
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)

    print("\nPHASE 5: EXPERIMENT 01")
    print("BASELINE AND FOLD VERIFICATION")
    print("=" * 65)

    original = load_phase4_module()

    if original.SEED != SEED:
        raise ValueError("Random seed mismatch")

    if original.LABELS != LABELS:
        raise ValueError("Class order mismatch")

    data = original.load_development_data()

    if len(data) != 3127:
        raise ValueError("Unexpected record count")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs")

    X = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()

    groups = original.create_connected_groups(data)

    print("Development records:", len(data))
    print("Classes:", len(set(y)))
    print("Connected groups:", len(set(groups)))

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(cv.split(X, y, groups))

    fold_rows = []
    fold_details = []

    for fold_number, (tr_idx, va_idx) in enumerate(
        folds, start=1
    ):
        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise ValueError(
                f"Fold {fold_number}: group leakage"
            )

        train_projects = set(
            data.iloc[tr_idx]["ProjectID"]
        )

        val_projects = set(
            data.iloc[va_idx]["ProjectID"]
        )

        if train_projects & val_projects:
            raise ValueError(
                f"Fold {fold_number}: project leakage"
            )

        train_labels = set(y[tr_idx])
        val_labels = set(y[va_idx])

        if train_labels != set(LABELS):
            raise ValueError("Missing training classes")

        if val_labels != set(LABELS):
            raise ValueError("Missing validation classes")

        for index in va_idx:
            fold_rows.append({
                "RecordID": str(data.iloc[index]["RecordID"]),
                "ProjectID": str(data.iloc[index]["ProjectID"]),
                "ActualClass": str(y[index]),
                "ConnectedGroup": int(groups[index]),
                "Fold": fold_number,
            })

        fold_details.append({
            "fold": fold_number,
            "train_records": len(tr_idx),
            "held_out_records": len(va_idx),
            "train_projects": len(train_projects),
            "held_out_projects": len(val_projects),
        })

        print(
            f"Fold {fold_number}: "
            f"train={len(tr_idx)}, "
            f"held_out={len(va_idx)}"
        )

    assignments = pd.DataFrame(fold_rows)

    if len(assignments) != len(data):
        raise ValueError("Incomplete fold assignments")

    if assignments["RecordID"].duplicated().any():
        raise ValueError("Repeated held-out RecordID")

    # Verify compatibility with Phase 4 Experiment 14.
    phase4_oof_path = (
        PHASE4 / "outputs" / "tables"
        / "14_distilbert_oof_predictions.csv"
    )

    if not phase4_oof_path.exists():
        raise FileNotFoundError(phase4_oof_path)

    phase4_oof = pd.read_csv(
        phase4_oof_path,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(phase4_oof) != len(data)
        or phase4_oof["RecordID"].duplicated().any()
    ):
        raise ValueError("Invalid Phase 4 OOF records")

    matched = assignments.merge(
        phase4_oof[
            ["RecordID", "ProjectID", "_class_", "Fold"]
        ],
        on="RecordID",
        suffixes=("_phase5", "_phase4"),
        validate="one_to_one",
    )

    if len(matched) != len(data):
        raise ValueError("Phase 4 OOF records do not match")

    checks = [
        (
            matched["ProjectID_phase5"].astype(str)
            == matched["ProjectID_phase4"].astype(str)
        ).all(),
        (
            matched["ActualClass"].astype(str)
            == matched["_class_"].astype(str)
        ).all(),
        (
            matched["Fold_phase5"].astype(int)
            == matched["Fold_phase4"].astype(int)
        ).all(),
    ]

    if not all(checks):
        raise ValueError("Phase 4 and Phase 5 folds differ")

    print("\nPhase 4 fold alignment: VERIFIED")

    if not SOURCE_METRICS.exists():
        raise FileNotFoundError(SOURCE_METRICS)

    baseline = pd.read_csv(SOURCE_METRICS)

    selected = baseline[
        (baseline["model"] == "Logistic Regression")
        & (baseline["features"] == "word+character")
    ].copy()

    if len(selected) != 3:
        raise ValueError(
            "Expected three LR baseline fold results"
        )

    selected = selected.sort_values("fold")

    for _, row in selected.iterrows():
        fold_number = int(row["fold"])
        expected = fold_details[fold_number - 1]

        if (
            int(row["train_records"])
            != expected["train_records"]
            or int(row["validation_records"])
            != expected["held_out_records"]
        ):
            raise ValueError(
                f"Baseline fold {fold_number} size mismatch"
            )

    print("\nPHASE 4 LOGISTIC REGRESSION BASELINE")
    print("=" * 65)

    metrics = [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "weighted_f1",
    ]

    summary = {}

    for metric in metrics:
        mean = float(selected[metric].mean())
        std = float(selected[metric].std(ddof=1))

        summary[metric] = {
            "mean": mean,
            "std": std,
        }

        print(
            f"{metric:<20}"
            f"mean={mean:.4f} std={std:.4f}"
        )

    assignments_path = (
        OUTPUT_TABLES / "01_fixed_fold_assignments.csv"
    )

    assignments.to_csv(
        assignments_path,
        index=False,
    )

    report = {
        "phase": 5,
        "experiment": 1,
        "seed": SEED,
        "records": len(data),
        "classes": LABELS,
        "connected_groups": len(set(groups)),
        "folds": fold_details,
        "phase4_fold_alignment": "verified",
        "source_script_sha256": sha256(SOURCE_SCRIPT),
        "baseline": summary,
        "test_partition_accessed": False,
    }

    (
        OUTPUT_REPORTS / "01_baseline_manifest.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", assignments_path)
    print("Saved: outputs/reports/01_baseline_manifest.json")
    print("\nEXPERIMENT 01 COMPLETED")
    print("Original test partition was not accessed.")


if __name__ == "__main__":
    main()
