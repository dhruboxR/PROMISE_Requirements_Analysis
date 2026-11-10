
"""Phase 7 Experiment 04B: Prepare verified blind review sheets."""

from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs/tables/03_source_context.csv"
OUT = ROOT / "outputs/annotation_review"

OUT.mkdir(parents=True, exist_ok=True)

# Manually checked against the PDF passages from Experiment 03.
# Original PROMISE+ and AI-suggested labels are deliberately absent.
PDF_REQUIREMENTS = {
    "AUDIT-005": (
        "Each file should have the suffix of .xml, "
        "e.g. A1234520090623000001.xml."
    ),
    "AUDIT-012": (
        "All xml documents attached to emails to the UK "
        "fisheries administrations' ERS system must be "
        "encrypted using PGP."
    ),
    "AUDIT-015": (
        "The ELSS must provide facilities to transmit "
        "deletions to previously transmitted data."
    ),
    "AUDIT-021": (
        "Status display update must be within 4 sec at "
        "the local stations (certain functions, such as "
        "telescope position, may have tighter constraints)."
    ),
    "AUDIT-026": (
        "System Administration shall be accessible with "
        "screen-reading software, screen-magnification "
        "software, and other software programs designed "
        "to increase accessibility."
    ),
    "AUDIT-027": (
        "For focusing and related activities, maximum "
        "acceptable detector readout time is about 0.1 sec, "
        "though only a portion of the detector may be read "
        "during that time."
    ),
    "AUDIT-031": (
        "Correction operation to capture and deliver a "
        "formatted correction request for transmission by "
        "a vessel's communications system(s) to the UK "
        "fisheries administrations' ERS system to correct "
        "previously sent data (COR)."
    ),
}

LABEL_DEFINITIONS = {
    "F": "Functional: system behavior or capability",
    "A": "Availability: required service availability",
    "L": "Legal: legal or regulatory constraints",
    "LF": "Look and Feel: visual appearance or presentation",
    "MN": "Maintainability: ease of maintenance or modification",
    "O": "Operational: operating environment or procedures",
    "PE": "Performance: timing, throughput, or resource performance",
    "SC": "Scalability: capacity to accommodate increasing load",
    "SE": "Security: protection and authorized access",
    "US": "Usability: accessibility and ease of use",
    "FT": "Fault Tolerance: continued operation despite failures",
    "PO": "Portability: operation across platforms or environments",
}

REVIEW_FIELDS = [
    "AuditID",
    "RequirementText",
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
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def extract_doc_description(block):
    """Extract description, excluding source metadata."""
    match = re.search(
        r"\bDescription:\s*(.*?)(?=\bSource:|\bReq ID:|$)",
        block,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        raise ValueError(
            "DOC description could not be extracted: "
            + block[:100]
        )

    return re.sub(r"\s+", " ", match.group(1)).strip()


def main():
    with INPUT.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    if len(records) != 11:
        raise ValueError("Expected exactly 11 records")

    ids = [r["AuditID"] for r in records]

    if len(set(ids)) != 11:
        raise ValueError("Duplicate audit identifiers")

    review_rows = []
    private_rows = []

    for row in sorted(records, key=lambda x: x["AuditID"]):
        audit_id = row["AuditID"]

        if row["SourceFormat"] == "PDF":
            requirement = PDF_REQUIREMENTS[audit_id]
        elif row["SourceFormat"] == "DOC":
            requirement = extract_doc_description(
                row["OriginalRequirement"]
            )
        else:
            raise ValueError("Unsupported source format")

        if not requirement.strip():
            raise ValueError(f"Empty requirement: {audit_id}")

        review_rows.append({
            "AuditID": audit_id,
            "RequirementText": requirement,
            "ProposedLabel": "",
            "Confidence": "",
            "NeedsMoreContext": "",
            "ReviewerReason": "",
        })

        private_rows.append({
            field: row.get(field, "")
            for field in KEY_FIELDS
        })

    for name in ("reviewer_1", "reviewer_2"):
        save_csv(
            OUT / f"04b_{name}_blind.csv",
            review_rows,
            REVIEW_FIELDS,
        )

    save_csv(
        OUT / "04b_original_label_key_private.csv",
        private_rows,
        KEY_FIELDS,
    )

    (OUT / "04b_label_definitions.json").write_text(
        json.dumps(LABEL_DEFINITIONS, indent=2),
        encoding="utf-8",
    )

    summary = {
        "experiment": "04B",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "requirements": 11,
        "pdf_requirements": 7,
        "doc_requirements": 4,
        "blind_reviewers": 2,
        "reviews_completed": 0,
        "original_labels_modified": False,
    }

    (OUT / "04b_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("PHASE 7 EXPERIMENT 04B")
    print("=" * 50)
    print("Requirements prepared:", len(review_rows))
    print("PDF statements:", 7)
    print("DOC statements:", 4)
    print("Blind reviewer sheets:", 2)
    print("Original labels hidden: Yes")
    print("Independent reviews completed: 0")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
