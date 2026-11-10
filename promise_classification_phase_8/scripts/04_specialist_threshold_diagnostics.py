
"""
Phase 8 Experiment 04
Binary specialist probability and threshold diagnostics.

Uses existing out-of-fold probabilities from Experiment 03.
No model training or MiniLM encoding.

All threshold comparisons are exploratory.
Thresholds must be selected using inner training-fold
validation before reporting outer-fold performance.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

SOURCE = TABLES / "03_specialist_oof_predictions.csv"

TARGETS = ["O", "SC", "PO"]
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


def main():
    df = pd.read_csv(
        SOURCE,
        dtype={"RecordID": str},
    )

    required = {
        "RecordID", "Fold", "Target",
        "ActualClass", "ActualBinary",
        "TargetProbability",
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    if df.duplicated(
        ["RecordID", "Target"]
    ).any():
        raise ValueError("Duplicate target predictions")

    if set(df["Target"]) != set(TARGETS):
        raise ValueError("Unexpected target classes")

    if not df["ActualBinary"].isin([0, 1]).all():
        raise ValueError("Invalid binary labels")

    if (
        (df["ActualBinary"] == 1)
        != (df["ActualClass"] == df["Target"])
    ).any():
        raise ValueError("Inconsistent target labels")

    if not df["TargetProbability"].between(0, 1).all():
        raise ValueError("Invalid probabilities")

    rows = []
    distribution_rows = []

    for target in TARGETS:
        subset = df[df["Target"] == target].copy()

        print(f"\nTARGET: F vs {target}")
        print("=" * 60)

        for actual_class in ["F", target]:
            scores = subset.loc[
                subset["ActualClass"] == actual_class,
                "TargetProbability",
            ]

            if scores.empty:
                raise ValueError(
                    f"No {actual_class} examples for {target}"
                )

            result = {
                "Target": target,
                "ActualClass": actual_class,
                "Count": len(scores),
                "Minimum": scores.min(),
                "Median": scores.median(),
                "P75": scores.quantile(0.75),
                "P90": scores.quantile(0.90),
                "Maximum": scores.max(),
            }

            distribution_rows.append(result)

            print(
                f"{actual_class}: n={len(scores)}, "
                f"median={result['Median']:.4f}, "
                f"p90={result['P90']:.4f}, "
                f"max={result['Maximum']:.4f}"
            )

        y = subset["ActualBinary"].to_numpy()
        scores = subset["TargetProbability"].to_numpy()

        for threshold in THRESHOLDS:
            pred = (scores >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y, pred, labels=[0, 1]
            ).ravel()

            rows.append({
                "Target": target,
                "Threshold": threshold,
                "Precision": precision_score(
                    y, pred, zero_division=0
                ),
                "Recall": recall_score(
                    y, pred, zero_division=0
                ),
                "F1": f1_score(
                    y, pred, zero_division=0
                ),
                "BalancedAccuracy": balanced_accuracy_score(
                    y, pred
                ),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "FalsePositiveRate": (
                    fp / (fp + tn) if fp + tn else 0
                ),
            })

    results = pd.DataFrame(rows)
    distributions = pd.DataFrame(distribution_rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        TABLES / "04_threshold_diagnostics.csv",
        index=False,
    )

    distributions.to_csv(
        TABLES / "04_probability_distributions.csv",
        index=False,
    )

    selected = results[
        results["Threshold"].isin(
            [0.10, 0.20, 0.30, 0.40, 0.50]
        )
    ][[
        "Target", "Threshold", "Precision",
        "Recall", "F1", "TP", "FP"
    ]].copy()

    report = [
        "PHASE 8 EXPERIMENT 04",
        "SPECIALIST THRESHOLD DIAGNOSTICS",
        "=" * 65,
        "",
        "Exploratory analysis only.",
        "No threshold is approved for final evaluation.",
        "",
        selected.to_string(index=False),
        "",
        "Probabilities are pooled from three different",
        "outer-fold models and are not necessarily calibrated",
        "on the same scale.",
        "",
        "Next step: training-only nested threshold selection.",
        "Original test accessed: No.",
    ]

    report_path = (
        REPORTS / "04_threshold_diagnostics.txt"
    )

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n" + "\n".join(report))
    print("\nSaved:", report_path)
    print("Model retraining: No")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
