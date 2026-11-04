
"""Phase 4 Experiment 12: Shared error analysis.

Uses saved validation predictions only.
No training and no original test data access.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
TABLES = ROOT / "outputs/tables"

PREDICTIONS = TABLES / "09_three_model_predictions.csv"

VALIDATION = (
    THESIS
    / "promise_preprocessing_phase_2"
    / "outputs/splits/final_validation.csv"
)


def main():
    predictions = pd.read_csv(
        PREDICTIONS,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    validation = pd.read_csv(
        VALIDATION,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(predictions) != 553 or len(validation) != 553:
        raise ValueError("Unexpected validation record count")

    if (
        predictions["RecordID"].duplicated().any()
        or validation["RecordID"].duplicated().any()
    ):
        raise ValueError("Duplicate RecordID found")

    required = [
        "LR_Prediction",
        "DistilBERT_Prediction",
        "RoBERTa_Prediction",
    ]

    for col in required:
        if col not in predictions.columns:
            raise ValueError(f"Missing column: {col}")

    if "RequirementText" not in validation.columns:
        raise ValueError("RequirementText column missing")

    df = predictions.merge(
        validation[
            ["RecordID", "ProjectID", "_class_", "RequirementText"]
        ],
        on="RecordID",
        suffixes=("", "_source"),
        validate="one_to_one",
    )

    if len(df) != 553:
        raise ValueError("Records lost during merge")

    if not (
        df["ProjectID"] == df["ProjectID_source"]
    ).all():
        raise ValueError("Project mismatch")

    if not (
        df["_class_"] == df["_class__source"]
    ).all():
        raise ValueError("Label mismatch")

    wrong = (
        (df["LR_Prediction"] != df["_class_"])
        & (df["DistilBERT_Prediction"] != df["_class_"])
        & (df["RoBERTa_Prediction"] != df["_class_"])
    )

    errors = df.loc[wrong].copy()

    print("\nEXPERIMENT 12: SHARED ERROR ANALYSIS")
    print("=" * 65)
    print("Total validation records:", len(df))
    print("All three models wrong:", len(errors))

    if len(errors) != 126:
        raise ValueError(
            "Shared-error count differs from Experiment 09"
        )

    print("\nSHARED ERRORS BY TRUE CLASS")
    class_counts = (
        errors["_class_"]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="SharedErrors")
    )
    print(class_counts.to_string(index=False))

    print("\nSHARED ERRORS BY PROJECT")
    project_counts = (
        errors.groupby("ProjectID")
        .size()
        .reset_index(name="SharedErrors")
    )

    total_by_project = (
        df.groupby("ProjectID")
        .size()
        .reset_index(name="Total")
    )

    project_counts = project_counts.merge(
        total_by_project,
        on="ProjectID",
        validate="one_to_one",
    )

    project_counts["SharedErrorRate"] = (
        project_counts["SharedErrors"]
        / project_counts["Total"]
    )

    print(
        project_counts
        .sort_values(
            ["SharedErrors", "SharedErrorRate"],
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print("\nSAMPLE SHARED ERRORS")

    columns = [
        "RecordID",
        "ProjectID",
        "_class_",
        "LR_Prediction",
        "DistilBERT_Prediction",
        "RoBERTa_Prediction",
        "RequirementText",
    ]

    for _, row in errors[columns].head(10).iterrows():
        print("\nRecord:", row["RecordID"])
        print("Project:", row["ProjectID"])
        print("Actual:", row["_class_"])
        print("LR:", row["LR_Prediction"])
        print("DistilBERT:", row["DistilBERT_Prediction"])
        print("RoBERTa:", row["RoBERTa_Prediction"])
        print("Text:", str(row["RequirementText"])[:350])

    errors[columns].to_csv(
        TABLES / "12_shared_model_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    class_counts.to_csv(
        TABLES / "12_shared_errors_by_class.csv",
        index=False,
    )

    project_counts.to_csv(
        TABLES / "12_shared_errors_by_project.csv",
        index=False,
    )

    print("\nFiles saved successfully.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
