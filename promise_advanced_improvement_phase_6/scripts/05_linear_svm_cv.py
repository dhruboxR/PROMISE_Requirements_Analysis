
"""Phase 6 Experiment 05: Linear SVM cross-validation.

Word + character TF-IDF with balanced LinearSVC.
Uses the same verified project-aware folds as Phase 5.
The original test partition is not accessed.
"""

from pathlib import Path
import importlib.util
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT.parent / "promise_advanced_models_phase_4"

FOLDS = ROOT / "outputs/tables/01_verified_folds.csv"
METRICS = ROOT / "outputs/metrics"
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def load_phase4():
    path = (
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py"
    )
    spec = importlib.util.spec_from_file_location(
        "phase4_svm_reference", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "MacroF1": float(
            f1_score(
                y_true, y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "BalancedAccuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "WeightedF1": float(
            f1_score(
                y_true, y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def main():
    for directory in [METRICS, TABLES, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)

    reference = load_phase4()
    data = reference.load_development_data().copy()

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    folds = pd.read_csv(
        FOLDS,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(data) != 3127 or len(folds) != 3127:
        raise ValueError("Unexpected dataset size")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate development RecordIDs")

    if folds["RecordID"].duplicated().any():
        raise ValueError("Duplicate fold RecordIDs")

    data = data.merge(
        folds[["RecordID", "ProjectID", "Phase6Fold"]],
        on="RecordID",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_verified"),
    )

    if len(data) != 3127:
        raise ValueError("Development/fold join mismatch")

    if not (
        data["ProjectID"] == data["ProjectID_verified"]
    ).all():
        raise ValueError("Project IDs do not match")

    if set(data["_class_"]) != set(LABELS):
        raise ValueError("Unexpected class labels")

    if set(data["Phase6Fold"].astype(int)) != {1, 2, 3}:
        raise ValueError("Unexpected fold assignments")

    print("\nPHASE 6 EXPERIMENT 05")
    print("BALANCED LINEAR SVM")
    print("=" * 60)

    fold_results = []
    predictions = []

    expected_sizes = {
        1: (2159, 968),
        2: (2024, 1103),
        3: (2071, 1056),
    }

    for fold in [1, 2, 3]:
        train = data[data["Phase6Fold"] != fold]
        valid = data[data["Phase6Fold"] == fold]

        if (len(train), len(valid)) != expected_sizes[fold]:
            raise ValueError(f"Fold {fold} size mismatch")

        if not set(train["ProjectID"]).isdisjoint(
            set(valid["ProjectID"])
        ):
            raise ValueError("Project leakage detected")

        if not set(train["NormalizedText"]).isdisjoint(
            set(valid["NormalizedText"])
        ):
            raise ValueError("Duplicate-text leakage detected")

        vectorizer = reference.make_features(
            "word+character"
        )

        X_train = vectorizer.fit_transform(
            train["NormalizedText"].astype(str)
        )

        X_valid = vectorizer.transform(
            valid["NormalizedText"].astype(str)
        )

        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )

        model.fit(X_train, train["_class_"])

        y_true = valid["_class_"].to_numpy()
        y_pred = model.predict(X_valid)

        scores = calculate_metrics(y_true, y_pred)

        fold_results.append({
            "Fold": fold,
            "TrainRecords": len(train),
            "HeldOutRecords": len(valid),
            **scores,
        })

        for record_id, project_id, actual, predicted in zip(
            valid["RecordID"],
            valid["ProjectID"],
            y_true,
            y_pred,
        ):
            predictions.append({
                "RecordID": record_id,
                "ProjectID": project_id,
                "Fold": fold,
                "ActualClass": actual,
                "PredictedLabel": predicted,
            })

        print(
            f"Fold {fold}: "
            f"Accuracy={scores['Accuracy']:.4f}, "
            f"MacroF1={scores['MacroF1']:.4f}, "
            f"BalancedAccuracy={scores['BalancedAccuracy']:.4f}"
        )

    metrics_df = pd.DataFrame(fold_results)
    pred_df = pd.DataFrame(predictions)

    if len(pred_df) != 3127:
        raise ValueError("Incomplete OOF predictions")

    if pred_df["RecordID"].duplicated().any():
        raise ValueError("Duplicate OOF prediction")

    means = metrics_df[
        ["Accuracy", "MacroF1",
         "BalancedAccuracy", "WeightedF1"]
    ].mean().to_dict()

    pooled = calculate_metrics(
        pred_df["ActualClass"],
        pred_df["PredictedLabel"],
    )

    metrics_df.to_csv(
        METRICS / "05_svm_fold_metrics.csv",
        index=False,
    )

    pred_df.to_csv(
        TABLES / "05_svm_oof_predictions.csv",
        index=False,
    )

    report = {
        "phase": 6,
        "experiment": 5,
        "model": "LinearSVC",
        "features": "word+character TF-IDF",
        "C": 1.0,
        "class_weight": "balanced",
        "random_seed": 42,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "mean_fold_metrics": means,
        "pooled_oof_metrics": pooled,
        "per_class_oof": classification_report(
            pred_df["ActualClass"],
            pred_df["PredictedLabel"],
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "original_test_accessed": False,
    }

    (REPORTS / "05_svm_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nMEAN THREE-FOLD RESULTS")
    for metric, value in means.items():
        print(f"{metric}: {value:.4f}")

    print("\nPOOLED OOF RESULTS")
    for metric, value in pooled.items():
        print(f"{metric}: {value:.4f}")

    print("\nEXPERIMENT 05 COMPLETED")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
