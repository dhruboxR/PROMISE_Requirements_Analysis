
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
TABLES.mkdir(parents=True, exist_ok=True)

source = TABLES / "24_matched_probabilities_fold1.csv"
df = pd.read_csv(source, dtype={"RecordID": str})

models = {
    "Hybrid": "HybridPrediction",
    "Mixed": "MixedPrediction",
    "Fusion": "FusionPrediction",
}

rows = []

for label, subset in df.groupby("ActualClass"):
    row = {
        "Class": label,
        "Support": len(subset),
    }

    for name, column in models.items():
        row[f"{name}_Correct"] = int(
            (subset[column] == label).sum()
        )
        row[f"{name}_Recall"] = float(
            (subset[column] == label).mean()
        )

    row["FusionCorrectedHybrid"] = int(
        (
            (subset["HybridPrediction"] != label)
            & (subset["FusionPrediction"] == label)
        ).sum()
    )

    row["FusionIntroducedErrors"] = int(
        (
            (subset["HybridPrediction"] == label)
            & (subset["FusionPrediction"] != label)
        ).sum()
    )

    rows.append(row)

summary = pd.DataFrame(rows).sort_values("Class")

summary.to_csv(
    TABLES / "24b_fusion_class_error_summary.csv",
    index=False
)

changed = df[
    df["HybridPrediction"] != df["FusionPrediction"]
].copy()

changed["ChangeOutcome"] = np.select(
    [
        (changed["HybridPrediction"] != changed["ActualClass"])
        & (changed["FusionPrediction"] == changed["ActualClass"]),

        (changed["HybridPrediction"] == changed["ActualClass"])
        & (changed["FusionPrediction"] != changed["ActualClass"]),
    ],
    [
        "Corrected",
        "IntroducedError",
    ],
    default="StillIncorrect"
)

changed[
    [
        "RecordID",
        "ProjectID",
        "ActualClass",
        "HybridPrediction",
        "MixedPrediction",
        "FusionPrediction",
        "ChangeOutcome",
    ]
].to_csv(
    TABLES / "24b_fusion_changed_predictions.csv",
    index=False
)

print("\nPHASE 8 EXPERIMENT 24B")
print("=" * 70)

print("\nPER-CLASS PERFORMANCE")
print(summary.round(4).to_string(index=False))

print("\nFUSION CHANGES")
print(changed["ChangeOutcome"].value_counts().to_string())

print("\nTotal changed predictions:", len(changed))

print("\nSaved:")
print("24b_fusion_class_error_summary.csv")
print("24b_fusion_changed_predictions.csv")

print("\nModel retraining: No")
print("Original test accessed: No")
print("Original labels modified: No")
