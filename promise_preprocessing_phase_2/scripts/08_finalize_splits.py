
"""Independently validate and freeze the selected PROMISE+ split."""

from pathlib import Path
from itertools import combinations
import hashlib
import json
import shutil

import pandas as pd

from common import (
    CLEANED_DIR, SPLITS_DIR, REPORTS_DIR, prepare_outputs
)

NAMES = ("train", "validation", "test")
MASTER = CLEANED_DIR / "04_labeled_requirements.csv"

COLUMNS = [
    "RecordID", "ProjectID", "RequirementText",
    "NormalizedText", "_class_", "LabelID",
    "IsFunctional", "BinaryLabel", "LabelConflict",
]


def read_data(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False,
    )


def text_key(text):
    return " ".join(str(text).casefold().split())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prepare_outputs()

    master = read_data(MASTER)

    paths = {
        name: SPLITS_DIR / f"07_candidate2_{name}.csv"
        for name in NAMES
    }

    parts = {name: read_data(path) for name, path in paths.items()}

    for name, part in parts.items():
        assert set(COLUMNS).issubset(part.columns), (
            f"Missing columns in {name}"
        )
        assert len(part) > 0, f"Empty {name} split"
        assert set(part["_class_"]) == set(master["_class_"]), (
            f"Missing classes in {name}"
        )

    combined = pd.concat(parts.values(), ignore_index=True)

    # Verify all original records are retained exactly once.
    assert len(combined) == len(master) == 3677
    assert combined["RecordID"].is_unique
    assert set(combined["RecordID"]) == set(master["RecordID"])

    expected = master[COLUMNS].sort_values(
        "RecordID"
    ).reset_index(drop=True)

    actual = combined[COLUMNS].sort_values(
        "RecordID"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, actual)

    # Independently check leakage between partitions.
    for first, second in combinations(NAMES, 2):
        a, b = parts[first], parts[second]

        project_overlap = (
            set(a["ProjectID"]) & set(b["ProjectID"])
        )

        text_overlap = (
            set(a["NormalizedText"].map(text_key))
            & set(b["NormalizedText"].map(text_key))
        )

        assert not project_overlap, (
            f"Project leakage: {first}/{second}"
        )
        assert not text_overlap, (
            f"Text leakage: {first}/{second}"
        )

    # Check the expected selected allocation.
    expected_sizes = {
        "train": 2574,
        "validation": 553,
        "test": 550,
    }

    for name in NAMES:
        assert len(parts[name]) == expected_sizes[name], (
            f"Unexpected size for {name}"
        )

    # Prepare reproducibility manifest.
    manifest = {
        "selected_candidate": "Candidate 2",
        "selection_basis": (
            "Predefined component-isolation and class-balance "
            "criteria; no model performance used"
        ),
        "seed": 42,
        "records": len(master),
        "master_sha256": checksum(MASTER),
        "splits": {},
        "project_overlap": False,
        "normalized_text_overlap": False,
        "original_records_preserved": True,
        "manual_annotation_conflicts_pending": 4,
    }

    # Freeze files, refusing to overwrite a different prior version.
    for name, source in paths.items():
        destination = SPLITS_DIR / f"final_{name}.csv"

        if destination.exists():
            if checksum(destination) != checksum(source):
                raise RuntimeError(
                    f"{destination} already exists with different "
                    "contents. Review before changing a frozen split."
                )
        else:
            shutil.copyfile(source, destination)

        manifest["splits"][name] = {
            "file": destination.name,
            "records": len(parts[name]),
            "projects": int(parts[name]["ProjectID"].nunique()),
            "classes": int(parts[name]["_class_"].nunique()),
            "sha256": checksum(destination),
        }

    output = REPORTS_DIR / "08_final_split_manifest.json"

    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(
                "Existing manifest differs. Review before overwriting."
            )

    output.write_text(
        json.dumps(manifest, indent=4),
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: FINAL SPLIT VALIDATION")
    print("=" * 55)

    for name in NAMES:
        info = manifest["splits"][name]
        print(
            f"{name}: {info['records']} records, "
            f"{info['projects']} projects, "
            f"{info['classes']} classes"
        )

    print("\nAll original records preserved: PASS")
    print("Project overlap: 0")
    print("Normalized-text overlap: 0")
    print("Class coverage: 12/12 in every split")
    print("\nFinal split files frozen successfully.")
    print(f"Manifest: {output}")


if __name__ == "__main__":
    main()
