
"""
PHASE 8 EXPERIMENT 13
Minority-Class Data Feasibility Analysis

Read-only analysis of the existing PROMISE+ development data.

Examines:
  1. Class frequency and project coverage
  2. Connected-group coverage
  3. Concentration by project and connected group
  4. Train/validation class support across fixed outer folds
  5. Group isolation checks

No model training, test access, or dataset modification.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

P4_SCRIPT = (
    THESIS
    / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

MANIFEST = (
    THESIS
    / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

TARGETS = ["O", "SC", "PO", "FT"]
OUTER_FOLDS = [1, 2, 3]


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effective_group_count(counts):
    """
    Inverse Simpson concentration:
    1 / sum(group_share ** 2)

    This is a descriptive diversity measure, not
    an estimate of statistically independent samples.
    """
    counts = np.asarray(counts, dtype=float)

    if len(counts) == 0 or counts.sum() == 0:
        return 0.0

    shares = counts / counts.sum()
    return float(1.0 / np.sum(shares ** 2))


def concentration_stats(frame, column):
    counts = frame[column].value_counts()

    if len(counts) == 0:
        return {
            "Distinct": 0,
            "LargestCount": 0,
            "LargestSharePct": 0.0,
            "EffectiveCount": 0.0
        }

    return {
        "Distinct": int(len(counts)),
        "LargestCount": int(counts.iloc[0]),
        "LargestSharePct": float(
            100.0 * counts.iloc[0] / counts.sum()
        ),
        "EffectiveCount": effective_group_count(
            counts.to_numpy()
        ),
    }


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    phase4 = load_script(
        P4_SCRIPT,
        "phase4_feasibility13"
    )

    if phase4.LABELS != LABELS:
        raise ValueError("Class label configuration changed")

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError(
            f"Unexpected development size: {len(data)}"
        )

    required = {
        "RecordID", "ProjectID",
        "NormalizedText", "_class_"
    }

    if not required.issubset(data.columns):
        raise ValueError(
            f"Missing columns: {required - set(data.columns)}"
        )

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)
    data["_class_"] = data["_class_"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID values")

    if (
        data["RecordID"].isna().any()
        or data["ProjectID"].isna().any()
        or data["_class_"].isna().any()
    ):
        raise ValueError("Missing identifiers or classes")

    if set(data["_class_"]) != set(LABELS):
        raise ValueError("Unexpected class labels")

    # Reuse the same connected-group construction
    # employed by the earlier experiments.
    groups = np.asarray(
        phase4.create_connected_groups(data)
    )

    if len(groups) != len(data):
        raise ValueError("Connected-group length mismatch")

    data["ConnectedGroup"] = groups.astype(str)

    manifest = pd.read_csv(
        MANIFEST,
        dtype={
            "RecordID": str,
            "ProjectID": str,
        },
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"])
            != set(data["RecordID"])
    ):
        raise ValueError("Invalid fixed-fold manifest")

    aligned = (
        manifest.set_index("RecordID")
        .loc[data["RecordID"]]
    )

    if not np.array_equal(
        aligned["ActualClass"].astype(str).to_numpy(),
        data["_class_"].to_numpy()
    ):
        raise ValueError("Manifest class mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        data["ProjectID"].to_numpy()
    ):
        raise ValueError("Manifest project mismatch")

    data["Fold"] = (
        aligned["Fold"].astype(int).to_numpy()
    )

    if set(data["Fold"]) != set(OUTER_FOLDS):
        raise ValueError("Unexpected fold IDs")

    group_fold_counts = (
        data.groupby("ConnectedGroup")["Fold"]
        .nunique()
    )

    if (group_fold_counts > 1).any():
        raise ValueError(
            "Connected-group overlap across outer folds"
        )

    print("\nPHASE 8 EXPERIMENT 13")
    print("=" * 75)
    print("Development records:", len(data))
    print("Projects:", data["ProjectID"].nunique())
    print(
        "Connected groups:",
        data["ConnectedGroup"].nunique()
    )
    print("Classes:", data["_class_"].nunique())
    print("Group separation: PASS")
    print("Original test accessed: No")

    # ---------------------------------------------------
    # 1. Global class distribution and concentration
    # ---------------------------------------------------
    class_rows = []

    for label in LABELS:
        subset = data[data["_class_"] == label]

        project = concentration_stats(
            subset, "ProjectID"
        )
        group = concentration_stats(
            subset, "ConnectedGroup"
        )

        class_rows.append({
            "Class": label,
            "Count": len(subset),
            "Percent": 100.0 * len(subset) / len(data),
            "Projects": project["Distinct"],
            "ConnectedGroups": group["Distinct"],
            "LargestProjectCount": project["LargestCount"],
            "LargestProjectSharePct":
                project["LargestSharePct"],
            "EffectiveProjects":
                project["EffectiveCount"],
            "LargestGroupCount": group["LargestCount"],
            "LargestGroupSharePct":
                group["LargestSharePct"],
            "EffectiveGroups":
                group["EffectiveCount"],
        })

    class_summary = pd.DataFrame(class_rows)

    class_summary.to_csv(
        TABLES / "13_class_project_group_coverage.csv",
        index=False,
    )

    print("\nCLASS COVERAGE")
    print("=" * 75)

    print(
        class_summary[
            [
                "Class", "Count", "Projects",
                "ConnectedGroups",
                "LargestProjectSharePct",
                "EffectiveProjects",
            ]
        ].round(2).to_string(index=False)
    )

    # ---------------------------------------------------
    # 2. Per-project class distribution
    # ---------------------------------------------------
    project_matrix = pd.crosstab(
        data["ProjectID"],
        data["_class_"]
    ).reindex(columns=LABELS, fill_value=0)

    project_matrix["Total"] = project_matrix[
        LABELS
    ].sum(axis=1)

    project_matrix.to_csv(
        TABLES / "13_project_class_matrix.csv"
    )

    # Per-connected-group class distribution
    group_matrix = pd.crosstab(
        data["ConnectedGroup"],
        data["_class_"]
    ).reindex(columns=LABELS, fill_value=0)

    group_matrix["Total"] = group_matrix[
        LABELS
    ].sum(axis=1)

    group_matrix.to_csv(
        TABLES / "13_connected_group_class_matrix.csv"
    )

    # ---------------------------------------------------
    # 3. Outer fold class coverage
    # ---------------------------------------------------
    fold_rows = []

    for fold in OUTER_FOLDS:
        train = data[data["Fold"] != fold]
        validation = data[data["Fold"] == fold]

        if (
            set(train["ConnectedGroup"])
            & set(validation["ConnectedGroup"])
        ):
            raise ValueError(
                f"Group overlap in outer fold {fold}"
            )

        for label in LABELS:
            train_subset = train[
                train["_class_"] == label
            ]

            val_subset = validation[
                validation["_class_"] == label
            ]

            fold_rows.append({
                "OuterFold": fold,
                "Class": label,
                "TrainCount": len(train_subset),
                "ValidationCount": len(val_subset),
                "TrainProjects":
                    train_subset["ProjectID"].nunique(),
                "ValidationProjects":
                    val_subset["ProjectID"].nunique(),
                "TrainConnectedGroups":
                    train_subset["ConnectedGroup"].nunique(),
                "ValidationConnectedGroups":
                    val_subset["ConnectedGroup"].nunique(),
            })

    fold_summary = pd.DataFrame(fold_rows)

    fold_summary.to_csv(
        TABLES / "13_outer_fold_class_coverage.csv",
        index=False,
    )

    print("\nOUTER FOLD SUPPORT: TARGET CLASSES")
    print("=" * 75)

    print(
        fold_summary[
            fold_summary["Class"].isin(TARGETS)
        ][
            [
                "OuterFold",
                "Class",
                "TrainCount",
                "ValidationCount",
                "TrainProjects",
                "ValidationProjects",
                "TrainConnectedGroups",
                "ValidationConnectedGroups",
            ]
        ].to_string(index=False)
    )

    # ---------------------------------------------------
    # 4. Target class project concentration
    # ---------------------------------------------------
    target_rows = []

    for target in TARGETS:
        subset = data[
            data["_class_"] == target
        ]

        counts = (
            subset.groupby("ProjectID")
            .size()
            .sort_values(ascending=False)
        )

        for rank, (project, count) in enumerate(
            counts.items(), start=1
        ):
            target_rows.append({
                "Class": target,
                "ProjectID": project,
                "Rank": rank,
                "Count": int(count),
                "SharePct": float(
                    100.0 * count / len(subset)
                ),
            })

    target_projects = pd.DataFrame(target_rows)

    target_projects.to_csv(
        TABLES / "13_target_project_concentration.csv",
        index=False,
    )

    print("\nTOP PROJECTS PER TARGET CLASS")
    print("=" * 75)

    print(
        target_projects[
            target_projects["Rank"] <= 5
        ].to_string(index=False)
    )

    # ---------------------------------------------------
    # 5. Descriptive flags
    # ---------------------------------------------------
    flags = []

    for _, row in class_summary.iterrows():
        issues = []

        if row["Projects"] < 5:
            issues.append("Fewer than 5 source projects")

        if row["ConnectedGroups"] < 5:
            issues.append("Fewer than 5 connected groups")

        if row["LargestProjectSharePct"] >= 50.0:
            issues.append(
                "At least 50% from one project"
            )

        if row["EffectiveProjects"] < 3:
            issues.append(
                "Effective project diversity below 3"
            )

        fold_subset = fold_summary[
            fold_summary["Class"] == row["Class"]
        ]

        if (fold_subset["ValidationCount"] == 0).any():
            issues.append(
                "Absent from at least one validation fold"
            )

        if (fold_subset["TrainCount"] < 10).any():
            issues.append(
                "Fewer than 10 training records in a fold"
            )

        flags.append({
            "Class": row["Class"],
            "Count": int(row["Count"]),
            "FlagCount": len(issues),
            "Flags": "; ".join(issues) if issues else "None",
        })

    flags_df = pd.DataFrame(flags)

    flags_df.to_csv(
        TABLES / "13_class_feasibility_flags.csv",
        index=False,
    )

    print("\nDESCRIPTIVE FEASIBILITY FLAGS")
    print("=" * 75)

    print(flags_df.to_string(index=False))

    print(
        "\nNote: Flags are descriptive screening rules, "
        "not statistically validated thresholds."
    )

    # ---------------------------------------------------
    # 6. Save audit metadata
    # ---------------------------------------------------
    metadata = {
        "experiment": "Phase 8 Experiment 13",
        "title": "Minority-Class Data Feasibility Analysis",
        "development_records": int(len(data)),
        "project_count": int(
            data["ProjectID"].nunique()
        ),
        "connected_group_count": int(
            data["ConnectedGroup"].nunique()
        ),
        "class_count": int(
            data["_class_"].nunique()
        ),
        "outer_folds": OUTER_FOLDS,
        "group_separation_passed": True,
        "model_retraining": False,
        "original_test_accessed": False,
        "original_datasets_modified": False,
        "effective_diversity_definition":
            "Inverse Simpson concentration of class counts",
        "analysis_limitations": [
            "Project and group counts do not establish model learnability.",
            "Groups may combine multiple projects.",
            "Class concentration is descriptive, not causal.",
            "These development folds have been explored previously.",
        ],
    }

    (
        REPORTS / "13_minority_feasibility.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 13 COMPLETED")
    print("=" * 75)
    print("Class coverage: Saved")
    print("Fold coverage: Saved")
    print("Project concentration: Saved")
    print("Feasibility flags: Saved")
    print("Group separation: PASS")
    print("Model retraining: No")
    print("Original test accessed: No")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
