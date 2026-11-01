
"""Prepare deterministic class labels for PROMISE+."""

import json
import pandas as pd

from common import (
    CLEANED_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    prepare_outputs,
    LABEL_COLUMN,
)

INPUT_FILE = CLEANED_DIR / "02_normalized_requirements.csv"

# Fixed mapping: do not derive IDs from class frequencies.
LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

LABEL_TO_ID = {
    label: idx for idx, label in enumerate(LABELS)
}


def match_key(text):
    return " ".join(str(text).casefold().split())


def main():
    prepare_outputs()

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False,
    )

    assert len(df) == 3677
    assert df["RecordID"].is_unique

    unexpected = set(df[LABEL_COLUMN]) - set(LABELS)
    if unexpected:
        raise ValueError(f"Unknown labels: {unexpected}")

    # Preserve original annotations.
    df["LabelID"] = df[LABEL_COLUMN].map(LABEL_TO_ID)
    df["IsFunctional"] = df[LABEL_COLUMN].eq("F")

    # Binary label: 0 = FR, 1 = NFR.
    df["BinaryLabel"] = (
        ~df["IsFunctional"]
    ).astype(int)

    # Identify normalized texts with multiple original labels.
    keys = df["NormalizedText"].map(match_key)

    label_counts = df.groupby(keys)[LABEL_COLUMN].nunique()
    conflict_keys = set(label_counts[label_counts > 1].index)

    df["LabelConflict"] = keys.isin(conflict_keys)

    assert df["LabelID"].notna().all()
    assert df[LABEL_COLUMN].isin(LABELS).all()
    assert len(df) == 3677

    # Save prepared master dataset.
    output_file = CLEANED_DIR / "04_labeled_requirements.csv"

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    # Save class mapping.
    mapping_file = TABLES_DIR / "04_label_mapping.json"

    mapping_file.write_text(
        json.dumps(
            {
                "label_to_id": LABEL_TO_ID,
                "binary_mapping": {
                    "F": 0,
                    "all_other_classes": 1,
                },
                "note": (
                    "Class IDs are fixed. Original annotations "
                    "are preserved. Binary mapping assumes "
                    "all non-F classes represent NFR categories."
                ),
            },
            indent=4,
        ),
        encoding="utf-8",
    )

    class_table = (
        df.groupby([LABEL_COLUMN, "LabelID"])
        .size()
        .reset_index(name="Count")
        .sort_values("LabelID")
    )

    class_table["Percentage"] = (
        class_table["Count"] / len(df) * 100
    ).round(2)

    class_table.to_csv(
        TABLES_DIR / "04_label_distribution.csv",
        index=False,
    )

    report = {
        "total_records": int(len(df)),
        "unique_labels": int(df[LABEL_COLUMN].nunique()),
        "functional_records": int(df["IsFunctional"].sum()),
        "nonfunctional_records": int((~df["IsFunctional"]).sum()),
        "conflicting_label_records": int(df["LabelConflict"].sum()),
        "conflicting_text_groups": len(conflict_keys),
        "records_removed": 0,
        "original_labels_preserved": True,
        "label_mapping": LABEL_TO_ID,
    }

    report_file = REPORTS_DIR / "04_label_preparation.json"

    report_file.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: LABEL PREPARATION")
    print("=" * 50)

    print(class_table.to_string(index=False))

    print("\nSummary:")
    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    print(output_file)
    print(mapping_file)
    print(report_file)

    print("\nLabel preparation successful.")


if __name__ == "__main__":
    main()
