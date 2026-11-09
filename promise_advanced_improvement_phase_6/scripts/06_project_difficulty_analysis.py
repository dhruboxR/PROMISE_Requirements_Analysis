
"""Phase 6 Experiment 06: Project difficulty and training coverage.

Investigates:
- Class distributions in difficult held-out projects
- Available same-class examples in each training fold
- Number of distinct training projects for each class
- Shared classification failures

Uses verified development folds only.
Does not access the original test partition.
"""

from pathlib import Path
import importlib.util
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT.parent / "promise_advanced_models_phase_4"

FOLDS = ROOT / "outputs/tables/01_verified_folds.csv"
DIFFICULTY = ROOT / "outputs/tables/06_hybrid_svm_difficulty.csv"
OUTPUT = ROOT / "outputs/tables/06_project_class_analysis.csv"
REPORT = ROOT / "outputs/reports/06_project_difficulty_summary.json"

def load_data():
    path = (
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py"
    )

    spec = importlib.util.spec_from_file_location(
        "phase4_reference", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.load_development_data()


def main():
    data = load_data().copy()
    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    folds = pd.read_csv(
        FOLDS,
        dtype={"RecordID": str, "ProjectID": str},
    )

    difficulty = pd.read_csv(
        DIFFICULTY,
        dtype={"ProjectID": str},
    )

    if len(data) != 3127 or len(folds) != 3127:
        raise ValueError("Unexpected development data size")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate development RecordIDs")

    if folds["RecordID"].duplicated().any():
        raise ValueError("Duplicate fold RecordIDs")

    if difficulty["ProjectID"].duplicated().any():
        raise ValueError("Duplicate project difficulty records")

    if not {
        "ProjectID", "Requirements", "BothWrong",
        "BothWrongRate"
    }.issubset(difficulty.columns):
        raise ValueError("Missing difficulty columns")

    data = data.merge(
        folds[["RecordID", "ProjectID", "Phase6Fold"]],
        on="RecordID",
        validate="one_to_one",
        suffixes=("", "_verified"),
    )

    if len(data) != 3127:
        raise ValueError("Fold join mismatch")

    if not (
        data["ProjectID"] == data["ProjectID_verified"]
    ).all():
        raise ValueError("ProjectID mismatch")

    data["Phase6Fold"] = data["Phase6Fold"].astype(int)

    if set(data["Phase6Fold"]) != {1, 2, 3}:
        raise ValueError("Invalid fold assignments")

    # Confirm all records of a project remain in one fold.
    if (
        data.groupby("ProjectID")["Phase6Fold"]
        .nunique().max() != 1
    ):
        raise ValueError("Project split across folds")

    if set(difficulty["ProjectID"]) != set(data["ProjectID"]):
        raise ValueError("Difficulty/project coverage mismatch")

    rows = []

    for project_id, project in data.groupby("ProjectID"):
        fold = int(project["Phase6Fold"].iloc[0])

        train = data[
            data["Phase6Fold"] != fold
        ]

        for label, category in project.groupby("_class_"):
            same_class_train = train[
                train["_class_"] == label
            ]

            rows.append({
                "ProjectID": project_id,
                "Fold": fold,
                "Class": label,
                "HeldOutClassRecords": len(category),
                "ProjectRecords": len(project),
                "TrainingClassRecords": len(same_class_train),
                "TrainingClassProjects": (
                    same_class_train["ProjectID"].nunique()
                ),
            })

    result = pd.DataFrame(rows)

    result = result.merge(
        difficulty[
            ["ProjectID", "BothWrong", "BothWrongRate"]
        ],
        on="ProjectID",
        validate="many_to_one",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT, index=False)

    difficult = result[
        result["ProjectRecords"] >= 10
    ].sort_values(
        ["BothWrongRate", "HeldOutClassRecords"],
        ascending=[False, False],
    )

    print("\nPHASE 6 EXPERIMENT 06")
    print("PROJECT DIFFICULTY AND CLASS COVERAGE")
    print("=" * 70)

    print("\nMOST DIFFICULT PROJECTS: CLASS COMPOSITION")
    print(
        difficult[
            [
                "ProjectID", "Fold", "Class",
                "HeldOutClassRecords",
                "TrainingClassRecords",
                "TrainingClassProjects",
                "BothWrongRate",
            ]
        ].head(35).to_string(index=False)
    )

    summary = {
        "phase": 6,
        "experiment": 6,
        "development_records": 3127,
        "projects_analyzed": int(data["ProjectID"].nunique()),
        "project_class_combinations": len(result),
        "projects_with_all_predictions_wrong": (
            difficulty.loc[
                difficulty["BothWrongRate"] == 1,
                "ProjectID"
            ].tolist()
        ),
        "original_test_accessed": False,
        "note": (
            "Descriptive development-set audit. "
            "No new model evaluation performed."
        ),
    }

    REPORT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT)
    print("Saved:", REPORT)
    print("\nEXPERIMENT 06 PROJECT AUDIT COMPLETED")


if __name__ == "__main__":
    main()
