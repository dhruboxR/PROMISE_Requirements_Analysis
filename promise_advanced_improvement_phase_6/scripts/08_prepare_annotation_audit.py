
"""Phase 6 Experiment 08: Prepare a blind annotation audit.

Samples development requirements without accessing original test data.
Keeps original labels in a separate answer key.
Does not modify any dataset labels.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE4_SCRIPT = (
    ROOT.parent
    / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

FOLD_FILE = ROOT / "outputs/tables/01_verified_folds.csv"

OUT = ROOT / "outputs/annotation_audit"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
PRIORITY_PROJECTS = ["80", "82", "86", "52", "59", "74"]
SAMPLES_PER_PROJECT = 12
CONTROL_SAMPLE_SIZE = 24

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def load_development_data():
    spec = importlib.util.spec_from_file_location(
        "phase4_audit_reference",
        PHASE4_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_development_data().copy()


def main():
    print("\nPHASE 6 EXPERIMENT 08")
    print("ANNOTATION CONSISTENCY AUDIT")
    print("=" * 60)

    data = load_development_data()
    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    folds = pd.read_csv(
        FOLD_FILE,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(data) != 3127 or len(folds) != 3127:
        raise ValueError("Unexpected development dataset size")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate development RecordID")

    if folds["RecordID"].duplicated().any():
        raise ValueError("Duplicate fold RecordID")

    data = data.merge(
        folds[
            ["RecordID", "ProjectID", "Phase6Fold", "ActualClass"]
        ],
        on="RecordID",
        validate="one_to_one",
        suffixes=("", "_verified"),
    )

    if len(data) != 3127:
        raise ValueError("Dataset/fold join mismatch")

    if not (
        data["ProjectID"] == data["ProjectID_verified"]
    ).all():
        raise ValueError("ProjectID mismatch")

    if not (
        data["_class_"] == data["ActualClass"]
    ).all():
        raise ValueError("Class label mismatch")

    if set(data["_class_"]) != set(LABELS):
        raise ValueError("Unexpected class labels")

    data["Phase6Fold"] = data["Phase6Fold"].astype(int)

    if set(data["Phase6Fold"]) != {1, 2, 3}:
        raise ValueError("Unexpected fold assignment")

    selected = []

    for project_id in PRIORITY_PROJECTS:
        subset = data[
            data["ProjectID"] == project_id
        ]

        if len(subset) < SAMPLES_PER_PROJECT:
            raise ValueError(
                f"Insufficient rows for Project {project_id}"
            )

        sample = subset.sample(
            n=SAMPLES_PER_PROJECT,
            random_state=SEED
        ).copy()

        sample["SamplingGroup"] = "Priority"
        selected.append(sample)

    other = data[
        ~data["ProjectID"].isin(PRIORITY_PROJECTS)
    ]

    controls = other.sample(
        n=CONTROL_SAMPLE_SIZE,
        random_state=SEED
    ).copy()

    controls["SamplingGroup"] = "Control"
    selected.append(controls)

    audit = pd.concat(
        selected,
        ignore_index=True
    )

    audit = audit.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    audit["AuditID"] = [
        f"AUDIT-{i:03d}"
        for i in range(1, len(audit) + 1)
    ]

    if len(audit) != 96:
        raise ValueError("Expected 96 audit records")

    if audit["RecordID"].duplicated().any():
        raise ValueError("Duplicate sampled requirements")

    # Hide original labels, projects, and model predictions.
    blind_columns = [
        "AuditID",
        "NormalizedText",
    ]

    review = audit[blind_columns].rename(
        columns={
            "NormalizedText": "RequirementText"
        }
    ).copy()

    review["SuggestedLabel"] = ""
    review["ReviewStatus"] = ""
    review["Confidence"] = ""
    review["ReviewerNotes"] = ""

    # Only consult after completing the blind review.
    answer_key = audit[
        [
            "AuditID",
            "RecordID",
            "ProjectID",
            "Phase6Fold",
            "SamplingGroup",
            "_class_",
        ]
    ].rename(
        columns={"_class_": "OriginalLabel"}
    )

    # Save a fresh second-reviewer template.
    review.to_csv(
        OUT / "08_reviewer_1.csv",
        index=False
    )

    review.to_csv(
        OUT / "08_reviewer_2.csv",
        index=False
    )

    answer_key.to_csv(
        OUT / "08_original_label_key.csv",
        index=False
    )

    summary = {
        "experiment": 8,
        "seed": SEED,
        "sampling_method": (
            "12 random examples from each of six "
            "priority projects plus 24 random "
            "development controls"
        ),
        "total_sampled": len(audit),
        "priority_sampled": 72,
        "control_sampled": 24,
        "priority_projects": PRIORITY_PROJECTS,
        "original_test_accessed": False,
        "original_labels_modified": False,
        "representative_of_full_dataset": False,
    }

    (
        OUT / "08_audit_sampling_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("Development requirements:", len(data))
    print("Priority requirements sampled:", 72)
    print("Control requirements sampled:", 24)
    print("Total audit requirements:", len(audit))

    print("\nSampled original classes:")
    print(
        audit["_class_"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .to_string()
    )

    print("\nSaved files:")
    for name in [
        "08_reviewer_1.csv",
        "08_reviewer_2.csv",
        "08_original_label_key.csv",
        "08_audit_sampling_summary.json",
    ]:
        print(OUT / name)

    print("\nEXPERIMENT 08 PREPARATION COMPLETED")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
