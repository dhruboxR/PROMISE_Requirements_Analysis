
"""Phase 4 Experiment 06: DistilBERT error analysis.

Analyzes saved validation predictions only.
Does not retrain a model or access the test partition.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def main():
    path = (
        TABLES
        / "05_extended_weighted_distilbert_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(df) != 553:
        raise ValueError("Unexpected validation record count")

    if df["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs")

    if not set(df["_class_"]).issubset(set(LABELS)):
        raise ValueError("Unexpected true labels")

    if not set(df["PredictedLabel"]).issubset(set(LABELS)):
        raise ValueError("Unexpected predictions")

    print("\nEXTENDED DISTILBERT ERROR ANALYSIS")
    print("=" * 65)

    print("\nPREDICTION DISTRIBUTION")
    print(df["PredictedLabel"].value_counts().to_string())

    rows = []

    for label in LABELS:
        subset = df[df["_class_"] == label]
        correct = (
            subset["PredictedLabel"] == label
        ).sum()

        rows.append({
            "Class": label,
            "Total": len(subset),
            "Correct": int(correct),
            "Recall": correct / len(subset) if len(subset) else 0,
            "Projects": subset["ProjectID"].nunique(),
        })

    summary = pd.DataFrame(rows)

    print("\nPER-CLASS SUMMARY")
    print(summary.to_string(index=False))

    summary.to_csv(
        TABLES / "06_distilbert_class_summary.csv",
        index=False,
    )

    for label in ["FT", "PO"]:
        subset = df[df["_class_"] == label]

        print(f"\nACTUAL {label}: PREDICTED LABELS")
        print(
            subset["PredictedLabel"]
            .value_counts()
            .to_string()
        )

        print(f"\n{label} VALIDATION PROJECTS")
        print(subset["ProjectID"].value_counts().to_string())

        subset.to_csv(
            TABLES / f"06_{label.lower()}_error_review.csv",
            index=False,
            encoding="utf-8-sig",
        )

    df["Correct"] = (
        df["_class_"] == df["PredictedLabel"]
    )

    project_summary = (
        df.groupby("ProjectID")
        .agg(
            Records=("RecordID", "size"),
            Correct=("Correct", "sum"),
        )
        .reset_index()
    )

    project_summary["Accuracy"] = (
        project_summary["Correct"]
        / project_summary["Records"]
    )

    project_summary.to_csv(
        TABLES / "06_distilbert_project_performance.csv",
        index=False,
    )

    print("\nPROJECT PERFORMANCE")
    print(
        project_summary
        .sort_values("Records", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\nDiagnostic files saved.")
    print("No test data accessed.")


if __name__ == "__main__":
    main()
