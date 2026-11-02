
"""PROMISE+ Phase 3: Review difficult validation examples."""

import pandas as pd

from common import TABLES_DIR, prepare_outputs, load_split

FOCUS_CLASSES = ["PO", "FT", "L", "SC", "MN", "LF"]


def main():
    prepare_outputs()

    train = load_split("train")
    validation = load_split("validation")

    predictions = pd.read_csv(
        TABLES_DIR
        / "07_tuned_logistic_regression_validation_predictions.csv",
        keep_default_na=False,
    )

    if predictions["RecordID"].duplicated().any():
        raise ValueError("Duplicate validation prediction IDs")

    expected_ids = set(validation["RecordID"])
    observed_ids = set(predictions["RecordID"])

    if expected_ids != observed_ids:
        raise ValueError("Validation records do not match")

    actual = validation[["RecordID", "_class_"]].merge(
        predictions[["RecordID", "_class_"]],
        on="RecordID",
        suffixes=("_verified", "_predicted_file"),
        validate="one_to_one",
    )

    if not (
        actual["_class__verified"]
        == actual["_class__predicted_file"]
    ).all():
        raise ValueError("Validation labels do not match")

    print("\nDIFFICULT CLASS ANALYSIS")
    print("=" * 65)

    rows = []

    for label in FOCUS_CLASSES:
        subset = predictions[predictions["_class_"] == label]
        correct = (
            subset["PredictedLabel"] == label
        ).sum()

        rows.append({
            "Class": label,
            "TrainCount": int((train["_class_"] == label).sum()),
            "ValidationCount": len(subset),
            "Correct": int(correct),
            "Recall": correct / len(subset) if len(subset) else 0,
        })

    summary = pd.DataFrame(rows)

    print(summary.to_string(index=False))

    summary.to_csv(
        TABLES_DIR / "09_difficult_class_summary.csv",
        index=False,
    )

    errors = predictions[
        predictions["_class_"].isin(FOCUS_CLASSES)
        & (predictions["_class_"] != predictions["PredictedLabel"])
    ].copy()

    errors = errors[[
        "RecordID",
        "ProjectID",
        "RequirementText",
        "_class_",
        "PredictedLabel",
    ]]

    errors.to_csv(
        TABLES_DIR / "09_difficult_class_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nPO MISCLASSIFICATIONS (FIRST 10)")
    print("=" * 65)

    po = errors[errors["_class_"] == "PO"]

    for _, row in po.head(10).iterrows():
        print(f"\nRecordID: {row['RecordID']}")
        print(f"ProjectID: {row['ProjectID']}")
        print(f"Actual: PO | Predicted: {row['PredictedLabel']}")
        print(f"Text: {row['RequirementText']}")

    print("\nSaved diagnostic files.")
    print("No training, label changes, or test-set access.")


if __name__ == "__main__":
    main()
