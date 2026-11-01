
"""Phase 2: Review duplicates and annotation conflicts without deleting data."""

from pathlib import Path
import json
import pandas as pd

from common import (
    CLEANED_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    prepare_outputs,
    TEXT_COLUMN,
    LABEL_COLUMN,
    PROJECT_COLUMN,
)


INPUT_FILE = CLEANED_DIR / "02_normalized_requirements.csv"


def create_match_key(text):
    """Use case-insensitive, whitespace-normalized matching."""
    return " ".join(str(text).casefold().split())


def main():
    prepare_outputs()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run 02_normalize_text.py before this script."
        )

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        dtype={
            PROJECT_COLUMN: str,
            LABEL_COLUMN: str,
        },
        keep_default_na=False,
    )

    required = {
        "RecordID",
        PROJECT_COLUMN,
        TEXT_COLUMN,
        "NormalizedText",
        LABEL_COLUMN,
    }

    if not required.issubset(df.columns):
        raise ValueError("Required columns are missing.")

    if len(df) != 3677 or not df["RecordID"].is_unique:
        raise ValueError("Unexpected records or duplicate RecordIDs.")

    # Preserve the normalized text from Stage 2.
    # This key is only for detecting potential duplicates.
    df["DuplicateKey"] = df["NormalizedText"].apply(
        create_match_key
    )

    original_cols = [
        "RecordID",
        PROJECT_COLUMN,
        TEXT_COLUMN,
        "NormalizedText",
        LABEL_COLUMN,
        "DuplicateKey",
    ]

    # 1. Exact duplicates across original attributes
    exact_cols = [
        PROJECT_COLUMN,
        TEXT_COLUMN,
        LABEL_COLUMN,
    ]

    exact_mask = df.duplicated(
        subset=exact_cols,
        keep=False,
    )

    exact_excess = int(
        df.duplicated(subset=exact_cols).sum()
    )

    exact_rows = df.loc[exact_mask, original_cols]

    # 2. Repeated normalized texts
    repeated_mask = df.duplicated(
        subset=["DuplicateKey"],
        keep=False,
    )

    repeated_excess = int(
        df.duplicated(subset=["DuplicateKey"]).sum()
    )

    repeated_rows = df.loc[repeated_mask, original_cols]

    # 3. Within-project repetitions
    within_mask = df.duplicated(
        subset=[PROJECT_COLUMN, "DuplicateKey"],
        keep=False,
    )

    within_rows = df.loc[within_mask, original_cols]

    # 4. Cross-project repetitions
    project_counts = df.groupby("DuplicateKey")[
        PROJECT_COLUMN
    ].nunique()

    cross_keys = project_counts[
        project_counts > 1
    ].index

    cross_rows = df.loc[
        df["DuplicateKey"].isin(cross_keys),
        original_cols,
    ]

    # 5. Conflicting classification labels
    label_counts = df.groupby("DuplicateKey")[
        LABEL_COLUMN
    ].nunique()

    conflict_keys = label_counts[
        label_counts > 1
    ].index

    conflict_rows = df.loc[
        df["DuplicateKey"].isin(conflict_keys),
        original_cols,
    ].copy()

    # No automatic label correction
    conflict_rows["ReviewStatus"] = "Pending manual review"
    conflict_rows["Decision"] = "Preserve original annotation"

    # 6. Group-level duplicate summary
    group_summary = (
        df.groupby("DuplicateKey")
        .agg(
            Occurrences=("RecordID", "size"),
            ProjectCount=(PROJECT_COLUMN, "nunique"),
            LabelCount=(LABEL_COLUMN, "nunique"),
            Labels=(
                LABEL_COLUMN,
                lambda x: ", ".join(sorted(set(x))),
            ),
            ExampleText=(TEXT_COLUMN, "first"),
        )
        .reset_index()
    )

    group_summary = group_summary[
        group_summary["Occurrences"] > 1
    ].copy()

    group_summary["HasLabelConflict"] = (
        group_summary["LabelCount"] > 1
    )

    group_summary["CrossProject"] = (
        group_summary["ProjectCount"] > 1
    )

    group_summary = group_summary.sort_values(
        ["HasLabelConflict", "Occurrences"],
        ascending=False,
    )

    # 7. Save review tables
    output_tables = {
        "03_exact_duplicates.csv": exact_rows,
        "03_repeated_texts.csv": repeated_rows,
        "03_within_project_repeats.csv": within_rows,
        "03_cross_project_repeats.csv": cross_rows,
        "03_label_conflicts.csv": conflict_rows,
        "03_duplicate_group_summary.csv": group_summary,
    }

    for filename, table in output_tables.items():
        table.to_csv(
            TABLES_DIR / filename,
            index=False,
            encoding="utf-8-sig",
        )

    # 8. Summary statistics
    summary = {
        "total_records": int(len(df)),
        "exact_duplicate_excess": exact_excess,
        "exact_duplicate_involved_rows": int(len(exact_rows)),
        "normalized_duplicate_excess": repeated_excess,
        "normalized_duplicate_involved_rows": int(
            len(repeated_rows)
        ),
        "unique_duplicate_groups": int(
            len(group_summary)
        ),
        "within_project_repeated_rows": int(
            len(within_rows)
        ),
        "cross_project_repeated_groups": int(
            len(cross_keys)
        ),
        "cross_project_repeated_rows": int(
            len(cross_rows)
        ),
        "conflicting_label_groups": int(
            len(conflict_keys)
        ),
        "conflicting_label_rows": int(
            len(conflict_rows)
        ),
        "records_removed": 0,
        "original_labels_preserved": True,
        "duplicate_matching": "casefold and whitespace normalization",
    }

    report = REPORTS_DIR / "03_duplicate_review_summary.json"

    report.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: DUPLICATE REVIEW")
    print("=" * 52)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nLabel-conflict groups:")
    for _, group in conflict_rows.groupby("DuplicateKey"):
        print("-" * 52)
        for _, row in group.iterrows():
            print(
                f"Record {row['RecordID']} | "
                f"Project {row[PROJECT_COLUMN]} | "
                f"Label {row[LABEL_COLUMN]}"
            )

    print("\nFiles saved:")
    for filename in output_tables:
        print(TABLES_DIR / filename)

    print(f"\nReport: {report}")
    print("\nReview completed. No records were removed.")


if __name__ == "__main__":
    main()
