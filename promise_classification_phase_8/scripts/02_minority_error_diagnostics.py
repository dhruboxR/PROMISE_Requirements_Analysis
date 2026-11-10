
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

INPUT = TABLES / "01_hierarchical_oof_predictions.csv"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO",
]

MODELS = [
    "BaselineTempered",
    "HardHierarchy",
    "SoftHierarchy",
]

FOCUS = ["O", "SC", "PO"]

def main():
    df = pd.read_csv(
        INPUT,
        dtype={"RecordID": str},
    )

    required = {
        "RecordID", "Fold", "Model",
        "ActualClass", "PredictedClass"
    }
    if not required.issubset(df.columns):
        raise ValueError("Prediction columns missing")

    if set(df["Model"]) != set(MODELS):
        raise ValueError("Unexpected model names")

    if not df["ActualClass"].isin(LABELS).all():
        raise ValueError("Unexpected true label")

    if not df["PredictedClass"].isin(LABELS).all():
        raise ValueError("Unexpected predicted label")

    if df.duplicated(["RecordID", "Model"]).any():
        raise ValueError("Duplicate model prediction")

    if len(df) != 3127 * len(MODELS):
        raise ValueError("Unexpected prediction count")

    comparison = df.pivot(
        index="RecordID",
        columns="Model",
        values="PredictedClass",
    )

    actual = (
        df.groupby("RecordID")["ActualClass"]
        .agg(lambda x: x.iloc[0] if x.nunique() == 1 else None)
    )

    if actual.isna().any():
        raise ValueError("Conflicting actual labels")

    reports = []
    confusion_rows = []

    for model in MODELS:
        subset = df[df["Model"] == model].copy()

        report = classification_report(
            subset["ActualClass"],
            subset["PredictedClass"],
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        for label in LABELS:
            reports.append({
                "Model": model,
                "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1": report[label]["f1-score"],
                "Support": report[label]["support"],
            })

        matrix = confusion_matrix(
            subset["ActualClass"],
            subset["PredictedClass"],
            labels=LABELS,
        )

        for i, actual_label in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "Model": model,
                    "ActualClass": actual_label,
                    "PredictedClass": predicted_label,
                    "Count": int(matrix[i, j]),
                })

    reports_df = pd.DataFrame(reports)
    confusion_df = pd.DataFrame(confusion_rows)

    reports_df.to_csv(
        TABLES / "02_per_class_metrics.csv",
        index=False
    )
    confusion_df.to_csv(
        TABLES / "02_confusion_counts.csv",
        index=False
    )

    lines = [
        "PHASE 8 EXPERIMENT 02",
        "MINORITY-CLASS ERROR DIAGNOSTICS",
        "=" * 65,
    ]

    for label in FOCUS:
        lines.append(f"\nCLASS: {label}")
        lines.append("-" * 50)

        class_report = reports_df[
            reports_df["Class"] == label
        ][[
            "Model", "Precision", "Recall", "F1", "Support"
        ]]

        lines.append(
            class_report.to_string(index=False)
        )

        for model in MODELS:
            subset = confusion_df[
                (confusion_df["Model"] == model)
                & (confusion_df["ActualClass"] == label)
                & (confusion_df["PredictedClass"] != label)
            ]

            top_errors = subset.sort_values(
                "Count", ascending=False
            ).head(5)

            lines.append(
                f"\n{model}: Most common wrong predictions"
            )
            lines.append(
                top_errors[
                    ["PredictedClass", "Count"]
                ].to_string(index=False)
            )

    baseline = comparison["BaselineTempered"]

    for model in ["HardHierarchy", "SoftHierarchy"]:
        candidate = comparison[model]

        improved = (
            (baseline != actual)
            & (candidate == actual)
        ).sum()

        worsened = (
            (baseline == actual)
            & (candidate != actual)
        ).sum()

        lines.extend([
            f"\n{model} vs BaselineTempered",
            f"Corrected baseline errors: {improved}",
            f"Introduced new errors: {worsened}",
            f"Net corrected records: {improved - worsened}",
        ])

    report_path = REPORTS / "02_minority_diagnostics.txt"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n".join(lines))
    print("\nSaved:", report_path)
    print("Original datasets modified: No")

if __name__ == "__main__":
    main()
