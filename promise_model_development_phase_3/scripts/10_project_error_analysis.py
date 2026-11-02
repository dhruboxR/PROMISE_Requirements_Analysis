
"""PROMISE+ Phase 3: Project-level validation error analysis."""

import pandas as pd

from common import (
    TABLES_DIR,
    prepare_outputs,
    load_split,
)


def main():
    prepare_outputs()

    train = load_split("train")
    validation = load_split("validation")

    predictions = pd.read_csv(
        TABLES_DIR /
        "07_tuned_logistic_regression_validation_predictions.csv",
        dtype={"ProjectID": str},
        keep_default_na=False,
    )

    if predictions["RecordID"].duplicated().any():
        raise ValueError("Duplicate prediction RecordIDs")

    if set(predictions["RecordID"]) != set(validation["RecordID"]):
        raise ValueError("Validation record mismatch")

    truth = validation[
        ["RecordID", "ProjectID", "_class_"]
    ].copy()

    merged = truth.merge(
        predictions[
            ["RecordID", "PredictedLabel"]
        ],
        on="RecordID",
        validate="one_to_one",
    )

    merged["Correct"] = (
        merged["_class_"] == merged["PredictedLabel"]
    )

    print("\nPO PROJECT DISTRIBUTION")
    print("=" * 65)

    for split_name, df in [
        ("train", train),
        ("validation", validation),
    ]:
        po = df[df["_class_"] == "PO"]

        print(f"\n{split_name.upper()}:")
        print(
            po.groupby("ProjectID")
            .size()
            .sort_values(ascending=False)
            .to_string()
        )

    po_validation = merged[
        merged["_class_"] == "PO"
    ]

    po_project_summary = (
        po_validation
        .groupby("ProjectID")
        .agg(
            PO_Records=("RecordID", "size"),
            Correct_PO=("Correct", "sum"),
        )
        .reset_index()
    )

    po_project_summary["PO_Recall"] = (
        po_project_summary["Correct_PO"]
        / po_project_summary["PO_Records"]
    )

    po_project_summary.to_csv(
        TABLES_DIR / "10_po_project_distribution.csv",
        index=False,
    )

    project_summary = (
        merged.groupby("ProjectID")
        .agg(
            Records=("RecordID", "size"),
            Correct=("Correct", "sum"),
        )
        .reset_index()
    )

    project_summary["Incorrect"] = (
        project_summary["Records"]
        - project_summary["Correct"]
    )

    project_summary["Accuracy"] = (
        project_summary["Correct"]
        / project_summary["Records"]
    )

    project_summary = project_summary.sort_values(
        ["Incorrect", "Records"],
        ascending=False,
    )

    print("\nPROJECTS WITH MOST VALIDATION ERRORS")
    print("=" * 65)

    print(project_summary.head(15).to_string(index=False))

    project_summary.to_csv(
        TABLES_DIR / "10_validation_project_performance.csv",
        index=False,
    )

    print("\nPO VALIDATION PROJECT SUMMARY")
    print(po_project_summary.to_string(index=False))

    print("\nSaved project-level analysis tables.")
    print("Test partition not accessed.")


if __name__ == "__main__":
    main()
