
"""Phase 3: Compare tuned models and analyze validation errors."""

import json

import pandas as pd

from common import (
    LABELS,
    REPORTS_DIR,
    TABLES_DIR,
    prepare_outputs,
)


def read_report(filename):
    path = REPORTS_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    prepare_outputs()

    svm_report = read_report(
        "06_tuned_svm_classification_report.json"
    )
    lr_report = read_report(
        "07_tuned_logistic_regression_classification_report.json"
    )

    rows = []

    for label in LABELS:
        svm = svm_report[label]
        lr = lr_report[label]

        rows.append({
            "Class": label,
            "Support": int(lr["support"]),
            "SVM_Precision": svm["precision"],
            "LR_Precision": lr["precision"],
            "SVM_Recall": svm["recall"],
            "LR_Recall": lr["recall"],
            "SVM_F1": svm["f1-score"],
            "LR_F1": lr["f1-score"],
            "F1_Change": lr["f1-score"] - svm["f1-score"],
        })

    comparison = pd.DataFrame(rows).sort_values("F1_Change")

    comparison.to_csv(
        TABLES_DIR / "08_tuned_models_per_class.csv",
        index=False,
    )

    print("\nTUNED MODEL PER-CLASS COMPARISON")
    print("=" * 85)
    print(comparison.round(4).to_string(index=False))

    print("\nClasses where Logistic Regression performs better:",
          int((comparison["F1_Change"] > 0).sum()))
    print("Classes where SVM performs better:",
          int((comparison["F1_Change"] < 0).sum()))
    print("Classes tied:",
          int((comparison["F1_Change"] == 0).sum()))

    svm = pd.read_csv(
        TABLES_DIR / "06_tuned_svm_validation_predictions.csv"
    )
    lr = pd.read_csv(
        TABLES_DIR /
        "07_tuned_logistic_regression_validation_predictions.csv"
    )

    # Align predictions using the original record identifier.
    if svm["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs in SVM predictions")

    if lr["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs in LR predictions")

    merged = svm[
        ["RecordID", "_class_", "PredictedLabel"]
    ].merge(
        lr[["RecordID", "_class_", "PredictedLabel"]],
        on="RecordID",
        suffixes=("_SVM", "_LR"),
        validate="one_to_one",
    )

    if len(merged) != len(svm) or len(merged) != len(lr):
        raise ValueError("Prediction files are not aligned")

    if not (
        merged["_class__SVM"] == merged["_class__LR"]
    ).all():
        raise ValueError("Actual labels differ between files")

    merged = merged.rename(columns={
        "_class__SVM": "ActualLabel",
        "PredictedLabel_SVM": "SVM_Prediction",
        "PredictedLabel_LR": "LR_Prediction",
    }).drop(columns=["_class__LR"])

    merged["SVM_Correct"] = (
        merged["ActualLabel"] == merged["SVM_Prediction"]
    )
    merged["LR_Correct"] = (
        merged["ActualLabel"] == merged["LR_Prediction"]
    )

    merged["Disagree"] = (
        merged["SVM_Prediction"] != merged["LR_Prediction"]
    )

    merged.to_csv(
        TABLES_DIR / "08_tuned_prediction_comparison.csv",
        index=False,
    )

    print("\nPREDICTION AGREEMENT")
    print("=" * 50)
    print("Total records:", len(merged))
    print("Different predictions:", int(merged["Disagree"].sum()))
    print("Both correct:", int(
        (merged["SVM_Correct"] & merged["LR_Correct"]).sum()
    ))
    print("Only SVM correct:", int(
        (merged["SVM_Correct"] & ~merged["LR_Correct"]).sum()
    ))
    print("Only LR correct:", int(
        (~merged["SVM_Correct"] & merged["LR_Correct"]).sum()
    ))
    print("Both incorrect:", int(
        (~merged["SVM_Correct"] & ~merged["LR_Correct"]).sum()
    ))

    # Analyze errors of the current leading model.
    errors = lr.loc[
        lr["_class_"] != lr["PredictedLabel"]
    ].copy()

    confusion = errors.groupby(
        ["_class_", "PredictedLabel"]
    ).size().reset_index(name="Count")

    confusion = confusion.sort_values(
        "Count", ascending=False
    )

    confusion.to_csv(
        TABLES_DIR / "08_lr_misclassification_pairs.csv",
        index=False,
    )

    print("\nTOP 15 LOGISTIC REGRESSION MISCLASSIFICATIONS")
    print("=" * 65)
    print(confusion.head(15).to_string(index=False))

    # Inspect the previously problematic PO class.
    po = lr[lr["_class_"] == "PO"]

    print("\nPO CLASS PREDICTIONS")
    print(po["PredictedLabel"].value_counts().to_string())

    po.to_csv(
        TABLES_DIR / "08_po_validation_predictions.csv",
        index=False,
    )

    print("\nSaved error-analysis tables.")
    print("No model retraining or test-set access performed.")


if __name__ == "__main__":
    main()
