
"""Phase 4 Experiment 10: Three-model majority voting.

Uses saved validation predictions from Experiment 09.
No retraining. No test partition access.
"""

from pathlib import Path
from collections import Counter
import json

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"
METRICS = ROOT / "outputs" / "metrics"

INPUT_FILE = TABLES / "09_three_model_predictions.csv"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def majority_vote(row):
    predictions = [
        row["LR_Prediction"],
        row["DistilBERT_Prediction"],
        row["RoBERTa_Prediction"],
    ]

    counts = Counter(predictions)
    label, count = counts.most_common(1)[0]

    if count >= 2:
        return label

    # All three disagree: prefer strongest validation model.
    return row["LR_Prediction"]


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true, y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    required = [
        "RecordID", "_class_",
        "LR_Prediction",
        "DistilBERT_Prediction",
        "RoBERTa_Prediction",
    ]

    if any(col not in df.columns for col in required):
        raise ValueError("Missing required prediction columns")

    if len(df) != 553 or df["RecordID"].duplicated().any():
        raise ValueError("Invalid validation records")

    for column in ["_class_"] + required[2:]:
        if not set(df[column]).issubset(set(LABELS)):
            raise ValueError(f"Unknown label in {column}")

    df["Ensemble_Prediction"] = df.apply(
        majority_vote, axis=1
    )

    y_true = df["_class_"]

    model_columns = {
        "Logistic Regression": "LR_Prediction",
        "DistilBERT": "DistilBERT_Prediction",
        "RoBERTa": "RoBERTa_Prediction",
        "Majority Voting": "Ensemble_Prediction",
    }

    results = []

    print("\nEXPERIMENT 10: THREE-MODEL MAJORITY VOTING")
    print("=" * 75)

    for model, column in model_columns.items():
        metrics = calculate_metrics(
            y_true, df[column]
        )

        results.append({
            "Model": model,
            **metrics,
        })

    summary = pd.DataFrame(results)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    all_different = (
        (df["LR_Prediction"] != df["DistilBERT_Prediction"])
        & (df["LR_Prediction"] != df["RoBERTa_Prediction"])
        & (
            df["DistilBERT_Prediction"]
            != df["RoBERTa_Prediction"]
        )
    )

    print("\nAll-three-disagree cases:", all_different.sum())

    print("\nENSEMBLE PER-CLASS REPORT")
    print(
        classification_report(
            y_true,
            df["Ensemble_Prediction"],
            labels=LABELS,
            zero_division=0,
        )
    )

    ensemble_report = classification_report(
        y_true,
        df["Ensemble_Prediction"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    (
        REPORTS / "10_majority_voting_classification_report.json"
    ).write_text(
        json.dumps(ensemble_report, indent=2),
        encoding="utf-8",
    )

    df.to_csv(
        TABLES / "10_majority_voting_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        METRICS / "10_majority_voting_comparison.csv",
        index=False,
    )

    print("\nResults saved successfully.")
    print("No test partition accessed.")


if __name__ == "__main__":
    main()
