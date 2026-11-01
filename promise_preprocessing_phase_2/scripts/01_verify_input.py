
"""Verify the Phase 1 dataset before preprocessing."""

import hashlib
import json

from common import (
    load_dataset,
    INPUT_PATH,
    REPORTS_DIR,
    prepare_outputs,
    TEXT_COLUMN,
    LABEL_COLUMN,
    PROJECT_COLUMN,
)


def main():
    prepare_outputs()

    df = load_dataset()

    checksum = hashlib.sha256(
        INPUT_PATH.read_bytes()
    ).hexdigest()

    report = {
        "input_path": str(INPUT_PATH),
        "sha256": checksum,
        "total_records": len(df),
        "total_columns": len(df.columns),
        "unique_projects": int(df[PROJECT_COLUMN].nunique()),
        "unique_classes": int(df[LABEL_COLUMN].nunique()),
        "missing_values": {
            col: int(count)
            for col, count in df.isna().sum().items()
        },
        "blank_texts": int(
            df[TEXT_COLUMN]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        ),
    }

    # Compare with the Phase 1 recorded checksum.
    phase1_summary = (
        INPUT_PATH.parent.parent
        / "reports"
        / "07_eda_summary.json"
    )

    if phase1_summary.exists():
        previous = json.loads(
            phase1_summary.read_text(encoding="utf-8")
        )

        expected_hash = previous.get(
            "validated_dataset_sha256"
        )

        if expected_hash and expected_hash != checksum:
            raise ValueError(
                "Dataset checksum differs from Phase 1 report."
            )

        if previous.get("total_records") != len(df):
            raise ValueError(
                "Record count differs from Phase 1 report."
            )

        report["phase1_comparison"] = "passed"
    else:
        report["phase1_comparison"] = "summary unavailable"

    if len(df) != 3677:
        raise ValueError("Unexpected record count.")

    if df[LABEL_COLUMN].nunique() != 12:
        raise ValueError("Unexpected number of classes.")

    if df[[PROJECT_COLUMN, TEXT_COLUMN, LABEL_COLUMN]].isna().any().any():
        raise ValueError("Missing values found.")

    if report["blank_texts"] != 0:
        raise ValueError("Blank requirement texts found.")

    output_path = REPORTS_DIR / "01_input_verification.json"

    output_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: INPUT VERIFICATION")
    print("=" * 45)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nInput verification successful.")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
