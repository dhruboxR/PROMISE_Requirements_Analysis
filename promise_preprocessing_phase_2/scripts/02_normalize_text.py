
"""Phase 2: Conservative text normalization for PROMISE+."""

import json
import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd

from common import (
    load_dataset,
    prepare_outputs,
    CLEANED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    TEXT_COLUMN,
    LABEL_COLUMN,
    PROJECT_COLUMN,
)


def normalize_text(value):
    """Normalize Unicode and whitespace while preserving meaning."""

    text = str(value)

    # Normalize Unicode representation without changing
    # capitalization or removing meaningful punctuation.
    text = unicodedata.normalize("NFC", text)

    # Convert consecutive whitespace characters to one space.
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace.
    return text.strip()


def main():
    prepare_outputs()

    df = load_dataset().copy()

    # Stable identifier based on original dataset row order.
    df.insert(0, "RecordID", range(1, len(df) + 1))

    # Preserve original values.
    df[PROJECT_COLUMN] = df[PROJECT_COLUMN].astype(str)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

    # Create a separate normalized-text column.
    df["NormalizedText"] = df[TEXT_COLUMN].apply(normalize_text)

    # Track changes to requirement text.
    changed_mask = df[TEXT_COLUMN] != df["NormalizedText"]

    audit = df.loc[
        changed_mask,
        [
            "RecordID",
            PROJECT_COLUMN,
            TEXT_COLUMN,
            "NormalizedText",
            LABEL_COLUMN,
        ],
    ].copy()

    audit["ChangeType"] = "unicode_or_whitespace_normalization"

    # Save full dataset.
    cleaned_path = CLEANED_DIR / "02_normalized_requirements.csv"

    df.to_csv(
        cleaned_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Save records whose text changed.
    audit_path = TABLES_DIR / "02_normalization_audit.csv"

    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Integrity checks.
    assert len(df) == 3677
    assert df["RecordID"].is_unique
    assert df["NormalizedText"].ne("").all()
    assert df[LABEL_COLUMN].notna().all()

    # Confirm originals remain unchanged.
    original = load_dataset()

    assert df[TEXT_COLUMN].tolist() == (
        original[TEXT_COLUMN].astype(str).tolist()
    )

    assert df[LABEL_COLUMN].tolist() == (
        original[LABEL_COLUMN].astype(str).tolist()
    )

    assert df[PROJECT_COLUMN].tolist() == (
        original[PROJECT_COLUMN].astype(str).tolist()
    )

    # Calculate summary statistics.
    summary = {
        "stage": "Phase 2 - Text Normalization",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_records": int(len(df)),
        "modified_text_records": int(changed_mask.sum()),
        "unchanged_text_records": int((~changed_mask).sum()),
        "unique_original_texts": int(df[TEXT_COLUMN].nunique()),
        "unique_normalized_texts": int(df["NormalizedText"].nunique()),
        "missing_normalized_texts": int(
            df["NormalizedText"].isna().sum()
        ),
        "blank_normalized_texts": int(
            df["NormalizedText"].eq("").sum()
        ),
        "normalization_operations": [
            "Unicode NFC",
            "Whitespace consolidation",
            "Leading and trailing whitespace removal",
        ],
        "original_text_preserved": True,
        "labels_preserved": True,
        "projects_preserved": True,
    }

    report_path = REPORTS_DIR / "02_normalization_summary.json"

    report_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: TEXT NORMALIZATION")
    print("=" * 50)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nGenerated files:")
    print(f"Normalized dataset: {cleaned_path}")
    print(f"Audit table: {audit_path}")
    print(f"Summary report: {report_path}")

    print("\nText normalization completed successfully.")


if __name__ == "__main__":
    main()
