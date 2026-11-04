
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

df = pd.read_csv(
    TABLES / "10_majority_voting_predictions.csv",
    dtype={"RecordID": str, "ProjectID": str},
    keep_default_na=False,
)

if len(df) != 553 or df["RecordID"].duplicated().any():
    raise ValueError("Unexpected validation records")

true = df["_class_"]
lr = df["LR_Prediction"]
ensemble = df["Ensemble_Prediction"]

changed = ensemble != lr

improved = changed & (ensemble == true)
damaged = changed & (lr == true)
both_wrong = changed & (lr != true) & (ensemble != true)

print("\nEXPERIMENT 11: VOTING ERROR ANALYSIS")
print("=" * 60)

print("Predictions changed from LR:", int(changed.sum()))
print("Changes that corrected LR:", int(improved.sum()))
print("Changes that damaged LR:", int(damaged.sum()))
print("Changes where both were wrong:", int(both_wrong.sum()))

print("\nNET CORRECT-PREDICTION CHANGE:",
      int(improved.sum() - damaged.sum()))

print("\nCHANGES BY ACTUAL CLASS")

result = df.loc[changed].copy()
result["Improved"] = improved[changed].values
result["Damaged"] = damaged[changed].values

summary = (
    result.groupby("_class_")
    .agg(
        Changed=("RecordID", "size"),
        Improved=("Improved", "sum"),
        Damaged=("Damaged", "sum"),
    )
    .reset_index()
)

summary["NetGain"] = summary["Improved"] - summary["Damaged"]

print(summary.to_string(index=False))

result.to_csv(
    TABLES / "11_voting_changed_predictions.csv",
    index=False,
)

summary.to_csv(
    TABLES / "11_voting_class_changes.csv",
    index=False,
)

print("\nAnalysis completed.")
print("Original test partition not accessed.")
