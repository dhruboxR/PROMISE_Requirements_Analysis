
"""Independently validate PROMISE+ Phase 2 data splits."""

import json
import hashlib
from itertools import combinations

import pandas as pd

from common import CLEANED_DIR, SPLITS_DIR, REPORTS_DIR, TABLES_DIR, prepare_outputs

MASTER_PATH = CLEANED_DIR / "04_labeled_requirements.csv"
NAMES = ["train", "validation", "test"]
REQUIRED = [
    "RecordID", "ProjectID", "RequirementText",
    "NormalizedText", "_class_", "LabelID",
    "BinaryLabel", "LabelConflict"
]


def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False
    )


def duplicate_key(value):
    return " ".join(str(value).casefold().split())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prepare_outputs()

    master = read_csv(MASTER_PATH)
    splits = {
        name: read_csv(SPLITS_DIR / f"05_{name}.csv")
        for name in NAMES
    }

    checks = {}
    problems = []

    def check(name, condition):
        passed = bool(condition)
        checks[name] = passed
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
        if not passed:
            problems.append(name)

    print("\nPROMISE+ PHASE 2: SPLIT VALIDATION")
    print("=" * 55)

    for name, part in splits.items():
        missing = set(REQUIRED) - set(part.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

    combined = pd.concat(splits.values(), ignore_index=True)

    check("Total record count preserved", len(combined) == len(master))
    check("RecordIDs are unique", combined["RecordID"].is_unique)
    check(
        "All original RecordIDs included",
        set(combined["RecordID"]) == set(master["RecordID"])
    )

    # Confirm saved records match the prepared master dataset.
    compare_columns = REQUIRED
    original = master[compare_columns].sort_values(
        "RecordID"
    ).reset_index(drop=True)

    reconstructed = combined[compare_columns].sort_values(
        "RecordID"
    ).reset_index(drop=True)

    check(
        "Original record content and labels preserved",
        original.equals(reconstructed)
    )

    check(
        "No missing split assignments",
        all(len(part) > 0 for part in splits.values())
    )

    for a, b in combinations(NAMES, 2):
        first, second = splits[a], splits[b]

        project_overlap = (
            set(first["ProjectID"]) & set(second["ProjectID"])
        )

        text_overlap = (
            set(first["NormalizedText"].map(duplicate_key))
            & set(second["NormalizedText"].map(duplicate_key))
        )

        check(
            f"{a}/{b}: no project overlap",
            len(project_overlap) == 0
        )

        check(
            f"{a}/{b}: no normalized-text overlap",
            len(text_overlap) == 0
        )

    classes = sorted(master["_class_"].unique())
    full_dist = master["_class_"].value_counts(normalize=True)

    distribution_rows = []
    class_rows = []

    for name, part in splits.items():
        counts = part["_class_"].value_counts()
        proportions = part["_class_"].value_counts(normalize=True)

        check(
            f"{name}: all classes represented",
            set(classes).issubset(set(counts.index))
        )

        total_variation = 0.5 * sum(
            abs(
                proportions.get(label, 0)
                - full_dist.get(label, 0)
            )
            for label in classes
        )

        distribution_rows.append({
            "Split": name,
            "Records": len(part),
            "Percentage": round(100 * len(part) / len(master), 2),
            "Projects": part["ProjectID"].nunique(),
            "Classes": part["_class_"].nunique(),
            "DistributionDistance": round(total_variation, 4),
        })

        for label in classes:
            class_rows.append({
                "Split": name,
                "Class": label,
                "Count": int(counts.get(label, 0)),
                "Percentage": round(
                    100 * proportions.get(label, 0), 2
                ),
                "FullDatasetPercentage": round(
                    100 * full_dist.get(label, 0), 2
                ),
            })

    summary = pd.DataFrame(distribution_rows)
    class_table = pd.DataFrame(class_rows)

    summary.to_csv(
        TABLES_DIR / "06_validation_summary.csv", index=False
    )
    class_table.to_csv(
        TABLES_DIR / "06_validation_class_distribution.csv",
        index=False
    )

    report = {
        "all_checks_passed": len(problems) == 0,
        "checks": checks,
        "failed_checks": problems,
        "master_sha256": sha256(MASTER_PATH),
        "split_sha256": {
            name: sha256(SPLITS_DIR / f"05_{name}.csv")
            for name in NAMES
        },
        "split_summary": distribution_rows,
        "distribution_distance_definition": (
            "Total variation distance between split class "
            "proportions and overall dataset class proportions. "
            "Zero indicates identical distributions."
        ),
    }

    path = REPORTS_DIR / "06_split_validation.json"
    path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\nSplit statistics:")
    print(summary.to_string(index=False))

    print("\nClass distribution:")
    print(class_table.to_string(index=False))

    print("\nReport:", path)

    if problems:
        raise SystemExit(
            f"\nValidation failed: {len(problems)} checks."
        )

    print("\nAll split integrity checks passed.")


if __name__ == "__main__":
    main()
