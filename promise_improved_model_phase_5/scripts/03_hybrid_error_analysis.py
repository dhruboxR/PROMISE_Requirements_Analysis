
"""Phase 5 Experiment 03: Analyze hybrid classifier predictions."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/tables/02_hybrid_oof_predictions.csv"
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

CONFIGS = [
    "TF-IDF baseline",
    "Hybrid 0.5",
    "Hybrid 1.0",
]


def class_metrics(df, config):
    p, r, f, support = precision_recall_fscore_support(
        df["_class_"],
        df["PredictedLabel"],
        labels=LABELS,
        zero_division=0,
    )

    return pd.DataFrame({
        "Class": LABELS,
        f"{config}_Support": support,
        f"{config}_Precision": p,
        f"{config}_Recall": r,
        f"{config}_F1": f,
    })


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    print("\nPHASE 5: EXPERIMENT 03")
    print("HYBRID CLASS-LEVEL AND ERROR ANALYSIS")
    print("=" * 65)

    subsets = {}

    for config in CONFIGS:
        subset = df[
            df["Configuration"] == config
        ].copy()

        if len(subset) != 3127:
            raise ValueError(
                f"Unexpected record count for {config}"
            )

        if subset["RecordID"].duplicated().any():
            raise ValueError("Duplicate RecordIDs")

        if set(subset["_class_"]) != set(LABELS):
            raise ValueError("Missing class labels")

        subsets[config] = subset

    # Compare per-class precision, recall and F1.
    comparison = class_metrics(
        subsets[CONFIGS[0]], CONFIGS[0]
    )

    for config in CONFIGS[1:]:
        comparison = comparison.merge(
            class_metrics(subsets[config], config),
            on="Class",
            validate="one_to_one",
        )

    comparison["Hybrid1_F1_Gain"] = (
        comparison["Hybrid 1.0_F1"]
        - comparison["TF-IDF baseline_F1"]
    )

    comparison["Hybrid1_Recall_Gain"] = (
        comparison["Hybrid 1.0_Recall"]
        - comparison["TF-IDF baseline_Recall"]
    )

    comparison = comparison.sort_values(
        "Hybrid1_F1_Gain", ascending=False
    )

    comparison.to_csv(
        TABLES / "03_per_class_comparison.csv",
        index=False,
    )

    print("\nPER-CLASS F1 COMPARISON")
    print("-" * 65)

    cols = [
        "Class",
        "TF-IDF baseline_F1",
        "Hybrid 0.5_F1",
        "Hybrid 1.0_F1",
        "Hybrid1_F1_Gain",
    ]

    print(
        comparison[cols].round(4).to_string(
            index=False
        )
    )

    # Align predictions using RecordID and Fold.
    baseline = subsets["TF-IDF baseline"][
        ["RecordID", "ProjectID", "Fold",
         "_class_", "PredictedLabel"]
    ].rename(
        columns={"PredictedLabel": "BaselinePrediction"}
    )

    hybrid = subsets["Hybrid 1.0"][
        ["RecordID", "ProjectID", "Fold",
         "_class_", "PredictedLabel"]
    ].rename(
        columns={"PredictedLabel": "HybridPrediction"}
    )

    joined = baseline.merge(
        hybrid,
        on=["RecordID", "ProjectID", "Fold", "_class_"],
        validate="one_to_one",
    )

    if len(joined) != 3127:
        raise ValueError("Prediction alignment failed")

    joined["BaselineCorrect"] = (
        joined["BaselinePrediction"] == joined["_class_"]
    )
    joined["HybridCorrect"] = (
        joined["HybridPrediction"] == joined["_class_"]
    )

    corrected = joined[
        ~joined["BaselineCorrect"]
        & joined["HybridCorrect"]
    ]

    damaged = joined[
        joined["BaselineCorrect"]
        & ~joined["HybridCorrect"]
    ]

    both_wrong = joined[
        ~joined["BaselineCorrect"]
        & ~joined["HybridCorrect"]
    ]

    both_correct = joined[
        joined["BaselineCorrect"]
        & joined["HybridCorrect"]
    ]

    print("\nBASELINE VS HYBRID 1.0")
    print("-" * 65)
    print("Both correct:", len(both_correct))
    print("Corrected by hybrid:", len(corrected))
    print("Damaged by hybrid:", len(damaged))
    print("Both wrong:", len(both_wrong))
    print("Net correct gain:", len(corrected) - len(damaged))

    joined.to_csv(
        TABLES / "03_prediction_comparison.csv",
        index=False,
    )

    print("\nFOLD-LEVEL COMPARISON")
    print("-" * 65)

    for fold in [1, 2, 3]:
        part = joined[joined["Fold"] == fold]

        print(f"\nFold {fold}:")
        print(
            "Baseline accuracy:",
            round(part["BaselineCorrect"].mean(), 4)
        )
        print(
            "Hybrid accuracy:",
            round(part["HybridCorrect"].mean(), 4)
        )

        print(
            "Corrected:",
            int(
                (~part["BaselineCorrect"]
                 & part["HybridCorrect"]).sum()
            )
        )
        print(
            "Damaged:",
            int(
                (part["BaselineCorrect"]
                 & ~part["HybridCorrect"]).sum()
            )
        )

    print("\nOOF PERFORMANCE CHECK")
    print("-" * 65)

    for config, subset in subsets.items():
        y_true = subset["_class_"]
        y_pred = subset["PredictedLabel"]

        print(
            f"{config}: "
            f"accuracy={accuracy_score(y_true, y_pred):.4f}, "
            f"macro-F1={f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0):.4f}, "
            f"balanced={balanced_accuracy_score(y_true, y_pred):.4f}"
        )

    print("\nExperiment 03 completed.")
    print("No model training or test access.")


if __name__ == "__main__":
    main()
