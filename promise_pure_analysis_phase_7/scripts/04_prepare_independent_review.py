
"""Phase 7 Experiment 04: Prepare independent blind reviewer sheets."""

from pathlib import Path
import csv
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "outputs/tables/03_source_context.csv"
OUT = ROOT / "outputs/annotation_review"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

REVIEW_FIELDS = [
    "AuditID",
    "RequirementTextForReview",
    "TextSourceType",
    "ProposedLabel",
    "Confidence",
    "NeedsMoreContext",
    "ReviewerReason",
]

KEY_FIELDS = [
    "AuditID",
    "RecordID",
    "ProjectID",
    "OriginalLabel",
    "AISuggestedLabel",
    "SourceDocument",
    "SourceLocation",
    "SourceRequirementID",
    "SourceSection",
]


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    with SOURCE.open(encoding="utf-8", newline="") as f:
        source_rows = list(csv.DictReader(f))

    ids = [row["AuditID"] for row in source_rows]

    if len(ids) != 11 or len(set(ids)) != 11:
        raise ValueError(
            "Expected exactly 11 unique priority audit records"
        )

    review_rows = []
    key_rows = []
    context_windows = []

    for row in sorted(source_rows, key=lambda r: r["AuditID"]):
        exact_block = row["OriginalRequirement"].strip()
        context = row["SurroundingContext"].strip()

        # PDF records currently have context windows rather
        # than verified complete requirement boundaries.
        if exact_block:
            text = exact_block
            source_type = "Extracted requirement block"
        else:
            text = context
            source_type = "Context window - verify full text"
            context_windows.append(row["AuditID"])

        if not text:
            raise ValueError(
                f"No review text available: {row['AuditID']}"
            )

        review_rows.append({
            "AuditID": row["AuditID"],
            "RequirementTextForReview": text,
            "TextSourceType": source_type,
            "ProposedLabel": "",
            "Confidence": "",
            "NeedsMoreContext": "",
            "ReviewerReason": "",
        })

        key_rows.append({
            field: row.get(field, "")
            for field in KEY_FIELDS
        })

    for reviewer in ("reviewer_1", "reviewer_2"):
        save_csv(
            OUT / f"04_{reviewer}_blind.csv",
            review_rows,
            REVIEW_FIELDS,
        )

    save_csv(
        OUT / "04_original_label_key_private.csv",
        key_rows,
        KEY_FIELDS,
    )

    summary = {
        "experiment": "04",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "requirements": len(review_rows),
        "reviewer_sheets": 2,
        "allowed_labels": LABELS,
        "pdf_context_windows_requiring_check": context_windows,
        "independent_reviews_completed": 0,
        "dataset_labels_modified": False,
    }

    (OUT / "04_review_preparation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("PHASE 7 EXPERIMENT 04")
    print("=" * 50)
    print("Requirements:", len(review_rows))
    print("Blind reviewer sheets created: 2")
    print("PDF context windows to verify:", len(context_windows))
    print("Original labels revealed to reviewers: No")
    print("Reviewer decisions recorded: 0")
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()
