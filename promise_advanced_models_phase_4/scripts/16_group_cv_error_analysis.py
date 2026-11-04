
"""Experiment 16: Cross-project model error analysis.

Uses Experiment 15 out-of-fold predictions.
No training or access to the original test partition.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

df = pd.read_csv(
    TABLES / "15_group_cv_model_complementarity.csv",
    dtype={"RecordID": str, "ProjectID_lr": str},
    keep_default_na=False,
)

if len(df) != 3127 or df["RecordID"].duplicated().any():
    raise ValueError("Unexpected OOF records")

if not (
    df["ProjectID_lr"].astype(str)
    == df["ProjectID_distil"].astype(str)
).all():
    raise ValueError("Project mismatch")

if not (df["Fold_lr"] == df["Fold_distil"]).all():
    raise ValueError("Fold mismatch")

df["BothWrong"] = (
    ~df["LR_Correct"].astype(bool)
    & ~df["Distil_Correct"].astype(bool)
)

df["LROnly"] = (
    df["LR_Correct"].astype(bool)
    & ~df["Distil_Correct"].astype(bool)
)

df["DistilOnly"] = (
    ~df["LR_Correct"].astype(bool)
    & df["Distil_Correct"].astype(bool)
)

print("\nEXPERIMENT 16: GROUP-AWARE ERROR ANALYSIS")
print("=" * 65)

print("\nERRORS BY CLASS")

class_summary = (
    df.groupby("Actual")
    .agg(
        Total=("RecordID", "size"),
        BothWrong=("BothWrong", "sum"),
        LROnly=("LROnly", "sum"),
        DistilOnly=("DistilOnly", "sum"),
    )
    .reset_index()
)

class_summary["SharedErrorRate"] = (
    class_summary["BothWrong"] / class_summary["Total"]
)

print(
    class_summary
    .sort_values("SharedErrorRate", ascending=False)
    .to_string(index=False)
)

print("\nTOP 15 PROJECTS BY SHARED ERRORS")

project_summary = (
    df.groupby("ProjectID_lr")
    .agg(
        Total=("RecordID", "size"),
        BothWrong=("BothWrong", "sum"),
        LROnly=("LROnly", "sum"),
        DistilOnly=("DistilOnly", "sum"),
    )
    .reset_index()
)

project_summary["SharedErrorRate"] = (
    project_summary["BothWrong"] / project_summary["Total"]
)

print(
    project_summary
    .sort_values(
        ["BothWrong", "SharedErrorRate"],
        ascending=False,
    )
    .head(15)
    .to_string(index=False)
)

print("\nERRORS BY FOLD")

fold_summary = (
    df.groupby("Fold_lr")
    .agg(
        Total=("RecordID", "size"),
        BothWrong=("BothWrong", "sum"),
        LROnly=("LROnly", "sum"),
        DistilOnly=("DistilOnly", "sum"),
    )
    .reset_index()
)

print(fold_summary.to_string(index=False))

class_summary.to_csv(
    TABLES / "16_shared_errors_by_class.csv",
    index=False,
)

project_summary.to_csv(
    TABLES / "16_shared_errors_by_project.csv",
    index=False,
)

fold_summary.to_csv(
    TABLES / "16_shared_errors_by_fold.csv",
    index=False,
)

print("\nExperiment 16 completed.")
print("Original test partition not accessed.")
