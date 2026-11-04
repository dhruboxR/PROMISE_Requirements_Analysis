
"""PROMISE+ Phase 4, Experiment 15.

Compare Logistic Regression and weighted DistilBERT on
the exact same three group-aware cross-validation folds.

Logistic Regression is fitted separately on each training fold.
DistilBERT OOF predictions come from Experiment 14.

Original test partition is never loaded.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

TABLES_DIR = ROOT / "outputs" / "tables"
METRICS_DIR = ROOT / "outputs" / "metrics"
REPORTS_DIR = ROOT / "outputs" / "reports"

DISTIL_OOF = (
    TABLES_DIR / "14_distilbert_oof_predictions.csv"
)

SEED = 42
EXPECTED_RECORDS = 3127

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def load_original_experiment():
    """Reuse Experiment 01 without modifying it."""
    path = SCRIPT_DIR / "01_group_aware_feature_comparison.py"

    spec = importlib.util.spec_from_file_location(
        "original_experiment_01",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def calculate_metrics(actual, predicted):
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def main():
    for folder in [
        TABLES_DIR,
        METRICS_DIR,
        REPORTS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nEXPERIMENT 15: GROUP-AWARE MODEL COMPARISON")
    print("=" * 65)

    original = load_original_experiment()

    if original.SEED != SEED:
        raise ValueError("Random seed mismatch")

    if original.LABELS != LABELS:
        raise ValueError("Label order mismatch")

    data = original.load_development_data()

    if len(data) != EXPECTED_RECORDS:
        raise ValueError("Unexpected development record count")

    X = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()

    groups = original.create_connected_groups(data)

    print("Development records:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Cross-validation folds: 3")

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(cv.split(X, y, groups))

    # Create Logistic Regression OOF predictions.
    lr_frames = []
    fold_metrics = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds, start=1
    ):
        print(f"\nFOLD {fold_number}")
        print("Training records:", len(train_idx))
        print("Held-out records:", len(val_idx))

        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError("Connected-group overlap")

        if set(data.iloc[train_idx]["ProjectID"]) & set(
            data.iloc[val_idx]["ProjectID"]
        ):
            raise ValueError("Project overlap")

        pipeline = Pipeline([
            (
                "features",
                original.make_features("word+character")
            ),
            (
                "classifier",
                original.make_classifier(
                    "Logistic Regression"
                )
            ),
        ])

        # Fit only on the current training fold.
        pipeline.fit(
            X[train_idx],
            y[train_idx],
        )

        predicted = pipeline.predict(X[val_idx])

        frame = data.iloc[val_idx][
            ["RecordID", "ProjectID", "_class_"]
        ].copy()

        frame["Fold"] = fold_number
        frame["LR_Prediction"] = predicted

        lr_frames.append(frame)

        scores = calculate_metrics(
            y[val_idx],
            predicted,
        )

        fold_metrics.append({
            "Model": "Logistic Regression",
            "Fold": fold_number,
            "Records": len(val_idx),
            **scores,
        })

        print(
            f"LR accuracy={scores['accuracy']:.4f} | "
            f"macro-F1={scores['macro_f1']:.4f}"
        )

    lr_oof = pd.concat(
        lr_frames,
        ignore_index=True,
    )

    if len(lr_oof) != EXPECTED_RECORDS:
        raise ValueError("Incomplete LR OOF predictions")

    if lr_oof["RecordID"].duplicated().any():
        raise ValueError("Duplicate LR RecordIDs")

    # Save LR OOF predictions.
    lr_oof.to_csv(
        TABLES_DIR / "15_logistic_regression_oof.csv",
        index=False,
    )

    # Load the existing DistilBERT OOF predictions.
    if not DISTIL_OOF.exists():
        raise FileNotFoundError(
            "Experiment 14 OOF predictions missing: "
            + str(DISTIL_OOF)
        )

    distil_oof = pd.read_csv(
        DISTIL_OOF,
        dtype={
            "RecordID": str,
            "ProjectID": str,
            "_class_": str,
        },
        keep_default_na=False,
    )

    if len(distil_oof) != EXPECTED_RECORDS:
        raise ValueError("Incomplete DistilBERT predictions")

    if distil_oof["RecordID"].duplicated().any():
        raise ValueError("Duplicate DistilBERT RecordIDs")

    required = {
        "RecordID",
        "ProjectID",
        "_class_",
        "Fold",
        "PredictedLabel",
    }

    if not required.issubset(distil_oof.columns):
        raise ValueError(
            "Missing columns in Experiment 14 results"
        )

    # Merge predictions by RecordID, never by row position.
    combined = lr_oof.merge(
        distil_oof[
            [
                "RecordID",
                "ProjectID",
                "_class_",
                "Fold",
                "PredictedLabel",
            ]
        ],
        on="RecordID",
        suffixes=("_lr", "_distil"),
        validate="one_to_one",
    )

    if len(combined) != EXPECTED_RECORDS:
        raise ValueError("Record alignment failed")

    for column in ["ProjectID", "_class_", "Fold"]:
        left = combined[f"{column}_lr"]
        right = combined[f"{column}_distil"]

        if not (left.astype(str) == right.astype(str)).all():
            raise ValueError(
                f"{column} mismatch between model folds"
            )

    combined["Actual"] = combined["_class__lr"]

    combined["DistilBERT_Prediction"] = (
        combined["PredictedLabel"]
    )

    for column in [
        "Actual",
        "LR_Prediction",
        "DistilBERT_Prediction",
    ]:
        if not set(combined[column]).issubset(set(LABELS)):
            raise ValueError(f"Unexpected class in {column}")

    combined["LR_Correct"] = (
        combined["LR_Prediction"]
        == combined["Actual"]
    )

    combined["Distil_Correct"] = (
        combined["DistilBERT_Prediction"]
        == combined["Actual"]
    )

    a = combined["LR_Correct"]
    b = combined["Distil_Correct"]

    overlap = {
        "Both correct": int((a & b).sum()),
        "Only LR correct": int((a & ~b).sum()),
        "Only DistilBERT correct": int((~a & b).sum()),
        "Both wrong": int((~a & ~b).sum()),
    }

    if sum(overlap.values()) != EXPECTED_RECORDS:
        raise ValueError("Overlap counts do not sum correctly")

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION MODEL COMPLEMENTARITY")
    print("=" * 65)

    for name, count in overlap.items():
        print(f"{name:<28}: {count}")

    disagreements = int(
        (
            combined["LR_Prediction"]
            != combined["DistilBERT_Prediction"]
        ).sum()
    )

    oracle_correct = int((a | b).sum())
    oracle_accuracy = oracle_correct / EXPECTED_RECORDS

    print("\nPrediction disagreements:", disagreements)
    print("At least one model correct:", oracle_correct)
    print(f"Oracle accuracy upper bound: {oracle_accuracy:.4f}")

    print("\nOVERALL OUT-OF-FOLD METRICS")

    summary_rows = []

    for model, column in [
        ("Logistic Regression", "LR_Prediction"),
        ("Weighted DistilBERT", "DistilBERT_Prediction"),
    ]:
        scores = calculate_metrics(
            combined["Actual"],
            combined[column],
        )

        summary_rows.append({
            "Model": model,
            **scores,
        })

        print(
            f"{model:<22} "
            f"Accuracy={scores['accuracy']:.4f} "
            f"Macro-F1={scores['macro_f1']:.4f} "
            f"Balanced={scores['balanced_accuracy']:.4f}"
        )

    print("\nDIFFICULT CLASS COMPARISON")

    for label in ["PO", "FT", "SC", "MN", "LF"]:
        subset = combined[
            combined["Actual"] == label
        ]

        print(
            f"{label:<3} | "
            f"total={len(subset):<4} | "
            f"LR correct={int(subset['LR_Correct'].sum()):<4} | "
            f"Distil correct={int(subset['Distil_Correct'].sum()):<4}"
        )

    # Save comparison files.
    combined.to_csv(
        TABLES_DIR / "15_group_cv_model_complementarity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {"Category": key, "Count": value}
            for key, value in overlap.items()
        ]
    ).to_csv(
        TABLES_DIR / "15_group_cv_overlap.csv",
        index=False,
    )

    pd.DataFrame(summary_rows).to_csv(
        METRICS_DIR / "15_group_cv_comparison.csv",
        index=False,
    )

    pd.DataFrame(fold_metrics).to_csv(
        METRICS_DIR / "15_logistic_regression_fold_metrics.csv",
        index=False,
    )

    report = {
        "experiment": 15,
        "development_records": EXPECTED_RECORDS,
        "folds": 3,
        "overlap": overlap,
        "prediction_disagreements": disagreements,
        "oracle_accuracy_upper_bound": oracle_accuracy,
        "models": summary_rows,
        "original_test_accessed": False,
    }

    (
        REPORTS_DIR / "15_group_cv_complementarity.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment 15 completed successfully.")
    print("Original test partition was not accessed.")


if __name__ == "__main__":
    main()
