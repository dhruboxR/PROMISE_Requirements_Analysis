
"""Phase 5 Experiment 07: Analyze saved class-weight predictions."""

from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/tables/06_class_weight_oof_predictions.csv"
OUTPUT = ROOT / "outputs/tables"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

MODES = ["none", "tempered", "balanced"]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        INPUT,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    required = {
        "RecordID", "ProjectID", "Fold", "_class_",
        "Configuration", "PredictedLabel",
    }
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    print("\nPHASE 5: EXPERIMENT 07")
    print("CLASS-WEIGHT ERROR ANALYSIS")
    print("=" * 65)

    subsets = {}
    class_reports = {}

    for mode in MODES:
        subset = df[
            df["Configuration"] == mode
        ].copy()

        if (
            len(subset) != 3127
            or subset["RecordID"].duplicated().any()
        ):
            raise ValueError(f"Invalid predictions: {mode}")

        subsets[mode] = subset

        y_true = subset["_class_"]
        y_pred = subset["PredictedLabel"]

        print(f"\nCONFIGURATION: {mode}")
        print("-" * 65)
        print(
            "Accuracy:",
            round(accuracy_score(y_true, y_pred), 4)
        )
        print(
            "Macro-F1:",
            round(
                f1_score(
                    y_true, y_pred,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ), 4
            )
        )
        print(
            "Balanced accuracy:",
            round(
                balanced_accuracy_score(
                    y_true, y_pred
                ), 4
            )
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        class_reports[mode] = report

    print("\nPER-CLASS F1 COMPARISON")
    print("=" * 65)

    rows = []

    for label in LABELS:
        row = {
            "Class": label,
            "Support": class_reports["tempered"][label]["support"],
        }

        for mode in MODES:
            row[f"{mode}_F1"] = (
                class_reports[mode][label]["f1-score"]
            )
            row[f"{mode}_Recall"] = (
                class_reports[mode][label]["recall"]
            )

        row["Tempered_vs_Balanced_F1"] = (
            row["tempered_F1"] - row["balanced_F1"]
        )

        rows.append(row)

    comparison = pd.DataFrame(rows)

    print(
        comparison[
            [
                "Class",
                "none_F1",
                "tempered_F1",
                "balanced_F1",
                "Tempered_vs_Balanced_F1",
            ]
        ].round(4).to_string(index=False)
    )

    comparison.to_csv(
        OUTPUT / "07_class_weight_per_class_comparison.csv",
        index=False,
    )

    # Join tempered and balanced predictions.
    a = subsets["tempered"][
        ["RecordID", "ProjectID", "Fold",
         "_class_", "PredictedLabel"]
    ].rename(
        columns={"PredictedLabel": "TemperedPrediction"}
    )

    b = subsets["balanced"][
        ["RecordID", "ProjectID", "Fold",
         "_class_", "PredictedLabel"]
    ].rename(
        columns={"PredictedLabel": "BalancedPrediction"}
    )

    joined = a.merge(
        b,
        on=["RecordID", "ProjectID", "Fold", "_class_"],
        validate="one_to_one",
    )

    if len(joined) != 3127:
        raise ValueError("Prediction alignment failed")

    tempered_correct = (
        joined["TemperedPrediction"] == joined["_class_"]
    )
    balanced_correct = (
        joined["BalancedPrediction"] == joined["_class_"]
    )

    print("\nTEMPERED VS BALANCED")
    print("=" * 65)
    print(
        "Both correct:",
        int((tempered_correct & balanced_correct).sum())
    )
    print(
        "Correct only by tempered:",
        int((tempered_correct & ~balanced_correct).sum())
    )
    print(
        "Correct only by balanced:",
        int((~tempered_correct & balanced_correct).sum())
    )
    print(
        "Both incorrect:",
        int((~tempered_correct & ~balanced_correct).sum())
    )

    joined.to_csv(
        OUTPUT / "07_tempered_vs_balanced_predictions.csv",
        index=False,
    )

    print("\nExperiment 07 completed.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
