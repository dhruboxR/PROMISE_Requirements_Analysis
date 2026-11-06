
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FOLD_FILE = (
    ROOT / "outputs/tables/01_fixed_fold_assignments.csv"
)

OUTPUT_FILE = (
    ROOT / "outputs/tables/05_minority_training_coverage.csv"
)

TARGET_CLASSES = ["O", "SC", "PO"]

def main():
    df = pd.read_csv(
        FOLD_FILE,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False
    )

    required = {
        "RecordID", "ProjectID", "ActualClass", "Fold"
    }
    if not required.issubset(df.columns):
        raise ValueError("Missing required columns")

    if len(df) != 3127 or df["RecordID"].duplicated().any():
        raise ValueError("Invalid fold assignments")

    print("\nPHASE 5: EXPERIMENT 05")
    print("MINORITY-CLASS TRAINING COVERAGE")
    print("=" * 65)

    rows = []

    for fold in [1, 2, 3]:
        train = df[df["Fold"] != fold]
        validation = df[df["Fold"] == fold]

        print(f"\nFOLD {fold}")
        print("-" * 65)

        for label in TARGET_CLASSES:
            tr = train[train["ActualClass"] == label]
            va = validation[
                validation["ActualClass"] == label
            ]

            print(f"\nClass {label}")
            print("Training examples:", len(tr))
            print("Training projects:", tr["ProjectID"].nunique())
            print("Validation examples:", len(va))
            print("Validation projects:", va["ProjectID"].nunique())

            print("Top training projects:")
            print(
                tr["ProjectID"]
                .value_counts()
                .head(5)
                .to_string()
            )

            rows.append({
                "Fold": fold,
                "Class": label,
                "TrainExamples": len(tr),
                "TrainProjects": tr["ProjectID"].nunique(),
                "ValidationExamples": len(va),
                "ValidationProjects": va["ProjectID"].nunique(),
            })

    pd.DataFrame(rows).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nSaved:", OUTPUT_FILE)
    print("Experiment 05 completed.")
    print("Original test partition not accessed.")

if __name__ == "__main__":
    main()
