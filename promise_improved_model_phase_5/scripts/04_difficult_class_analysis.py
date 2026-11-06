
"""Phase 5 Experiment 04: Difficult-class error analysis.

Analyze O, SC and PO using saved out-of-fold predictions.
No model training and no access to the original test set.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/tables/03_prediction_comparison.csv"
OUTPUT = ROOT / "outputs/tables"

TARGET_CLASSES = ["O", "SC", "PO"]
LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    required = {
        "RecordID", "ProjectID", "Fold", "_class_",
        "BaselinePrediction", "HybridPrediction",
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    if len(df) != 3127:
        raise ValueError("Unexpected record count")

    if df["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs")

    print("\nPHASE 5: EXPERIMENT 04")
    print("DIFFICULT-CLASS ERROR INVESTIGATION")
    print("=" * 65)

    df["BaselineCorrect"] = (
        df["BaselinePrediction"] == df["_class_"]
    )

    df["HybridCorrect"] = (
        df["HybridPrediction"] == df["_class_"]
    )

    project_rows = []
    confusion_rows = []

    for label in TARGET_CLASSES:
        subset = df[df["_class_"] == label].copy()

        baseline_correct = int(
            subset["BaselineCorrect"].sum()
        )
        hybrid_correct = int(
            subset["HybridCorrect"].sum()
        )

        print(f"\nCLASS: {label}")
        print("-" * 65)
        print("Actual requirements:", len(subset))
        print("Projects:", subset["ProjectID"].nunique())
        print("Baseline correct:", baseline_correct)
        print("Hybrid correct:", hybrid_correct)

        corrected = int((
            ~subset["BaselineCorrect"]
            & subset["HybridCorrect"]
        ).sum())

        damaged = int((
            subset["BaselineCorrect"]
            & ~subset["HybridCorrect"]
        ).sum())

        print("Corrected by hybrid:", corrected)
        print("Damaged by hybrid:", damaged)

        for column in [
            "BaselinePrediction",
            "HybridPrediction",
        ]:
            print(f"\n{column}:")
            print(
                subset[column]
                .value_counts()
                .head(8)
                .to_string()
            )

        print("\nPROJECT-LEVEL PERFORMANCE:")

        project = (
            subset.groupby("ProjectID")
            .agg(
                Records=("RecordID", "size"),
                BaselineCorrect=(
                    "BaselineCorrect", "sum"
                ),
                HybridCorrect=(
                    "HybridCorrect", "sum"
                ),
            )
            .reset_index()
        )

        project["Class"] = label
        project["HybridRecall"] = (
            project["HybridCorrect"]
            / project["Records"]
        )

        project = project.sort_values(
            ["Records", "ProjectID"],
            ascending=[False, True],
        )

        project_rows.append(project)

        print(project.head(10).to_string(index=False))

        for _, row in subset.iterrows():
            if row["HybridPrediction"] != label:
                confusion_rows.append({
                    "RecordID": row["RecordID"],
                    "ProjectID": row["ProjectID"],
                    "Fold": row["Fold"],
                    "ActualClass": label,
                    "BaselinePrediction": (
                        row["BaselinePrediction"]
                    ),
                    "HybridPrediction": (
                        row["HybridPrediction"]
                    ),
                })

    all_projects = pd.concat(
        project_rows, ignore_index=True
    )

    all_projects.to_csv(
        OUTPUT / "04_difficult_class_project_analysis.csv",
        index=False,
    )

    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT / "04_difficult_class_errors.csv",
        index=False,
    )

    # Complete 12-class hybrid confusion matrix.
    matrix = confusion_matrix(
        df["_class_"],
        df["HybridPrediction"],
        labels=LABELS,
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    )

    confusion_df.index.name = "ActualClass"

    confusion_df.to_csv(
        OUTPUT / "04_hybrid_confusion_matrix.csv"
    )

    print("\nSaved analysis tables.")
    print("Experiment 04 completed.")
    print("No model training or test access.")


if __name__ == "__main__":
    main()
