
"""
Phase 7 - Experiment 03
PURE Source Context Extraction

Inputs:
    outputs/tables/02_source_matches.csv
    dataset/Pure/req/

Outputs:
    outputs/tables/03_source_context.csv
    outputs/reports/03_source_context_summary.json

Notes:
- Original PROMISE+ labels are preserved.
- Every distinct match is retained for review.
- PDF page numbers refer to physical PDF pages.
- DOC line numbers refer to antiword-extracted text.
- No automatic label corrections are performed.
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PURE_DIR = ROOT.parent / "dataset" / "Pure" / "req"

INPUT_CSV = ROOT / "outputs/tables/02_source_matches.csv"
OUTPUT_CSV = ROOT / "outputs/tables/03_source_context.csv"
REPORT_JSON = ROOT / "outputs/reports/03_source_context_summary.json"

# Matching fragments confirmed during Experiment 02.
FRAGMENTS = {
    "AUDIT-005": "Each file should have the suffix of",
    "AUDIT-009": "generate a report of the value of items",
    "AUDIT-010": "generates a list of patrons with no circulation",
    "AUDIT-012": "must be encrypted using PGP",
    "AUDIT-015": "transmit deletions to previously transmitted data",
    "AUDIT-021": "Status display update must be within 4 sec",
    "AUDIT-026": "accessible with screen-reading software",
    "AUDIT-027": "maximum acceptable detector readout time",
    "AUDIT-028": (
        "statistics that can be used to understand "
        "and predict patron behavior"
    ),
    "AUDIT-031": "Correction operation to capture and deliver",
    "AUDIT-034": "interface showing capacity of all branches",
}

FIELDS = [
    "AuditID",
    "RecordID",
    "ProjectID",
    "OriginalLabel",
    "AISuggestedLabel",
    "SourceDocument",
    "SourceFormat",
    "SourceLocation",
    "SourceRequirementID",
    "SourceSection",
    "OriginalRequirement",
    "SurroundingContext",
    "OccurrenceCount",
    "MatchStatus",
    "ContextVerification",
    "ReviewerDecision",
    "ReviewerNotes",
]


def normalize_positions(text):
    """Normalize punctuation/spacing but retain source offsets."""
    chars = []
    offsets = []

    for index, character in enumerate(text):
        if character.isalnum():
            folded = character.casefold()
            for item in folded:
                chars.append(item)
                offsets.append(index)

    return "".join(chars), offsets


def find_occurrences(text, fragment):
    """Return all distinct locations of a normalized fragment."""
    normalized, offsets = normalize_positions(text)
    needle, _ = normalize_positions(fragment)

    if not needle:
        return []

    results = []
    start_at = 0

    while True:
        pos = normalized.find(needle, start_at)

        if pos == -1:
            break

        start = offsets[pos]
        end = offsets[pos + len(needle) - 1] + 1

        results.append((start, end))
        start_at = pos + len(needle)

    return results


def extract_pdf_pages(path):
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )

    return [
        (index, page)
        for index, page in enumerate(
            completed.stdout.split("\f"), start=1
        )
        if page.strip()
    ]


def extract_doc_text(path):
    completed = subprocess.run(
        ["antiword", str(path)],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )

    return completed.stdout


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def get_nearby_context(text, start, end, width=450):
    """Return exact-match neighborhood, without classifying it."""
    left = max(0, start - width)
    right = min(len(text), end + width)
    return clean(text[left:right])


def find_doc_requirement(text, start):
    """
    Find the nearest preceding Req ID and its requirement block.

    End the block at the next Req ID or Category heading.
    This is a document-structure heuristic, not expert verification.
    """
    preceding = list(
        re.finditer(
            r"(?m)^\s*Req ID:\s*(\S+)",
            text[:start],
        )
    )

    if not preceding:
        return "", "", ""

    req_match = preceding[-1]
    req_id = req_match.group(1)
    block_start = req_match.start()

    remainder = text[start:]

    boundary = re.search(
        r"(?m)^\s*(?:Req ID:|Category:)",
        remainder,
    )

    block_end = (
        start + boundary.start()
        if boundary
        else len(text)
    )

    requirement = clean(text[block_start:block_end])

    # Find the nearest Category heading preceding the Req ID.
    category_matches = list(
        re.finditer(
            r"(?m)^\s*Category:\s*(.+)$",
            text[:block_start],
        )
    )

    section = (
        clean(category_matches[-1].group(1))
        if category_matches
        else ""
    )

    return req_id, section, requirement


def context_rows_for_pdf(record, filename, fragment):
    path = PURE_DIR / filename
    rows = []

    for page_number, page in extract_pdf_pages(path):
        for start, end in find_occurrences(page, fragment):
            line_number = page.count("\n", 0, start) + 1

            rows.append({
                "SourceDocument": filename,
                "SourceFormat": "PDF",
                "SourceLocation": (
                    f"PDF page {page_number}, "
                    f"extracted line {line_number}"
                ),
                "SourceRequirementID": "",
                "SourceSection": "",
                "OriginalRequirement": "",
                "SurroundingContext": get_nearby_context(
                    page, start, end, width=650
                ),
            })

    return rows


def context_rows_for_doc(record, filename, fragment):
    path = PURE_DIR / filename
    text = extract_doc_text(path)
    rows = []

    for start, end in find_occurrences(text, fragment):
        line_number = text.count("\n", 0, start) + 1

        req_id, section, requirement = find_doc_requirement(
            text, start
        )

        rows.append({
            "SourceDocument": filename,
            "SourceFormat": "DOC",
            "SourceLocation": (
                f"DOC extracted line {line_number}"
            ),
            "SourceRequirementID": req_id,
            "SourceSection": section,
            "OriginalRequirement": requirement,
            "SurroundingContext": get_nearby_context(
                text, start, end, width=450
            ),
        })

    return rows


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    with INPUT_CSV.open(
        encoding="utf-8", newline=""
    ) as file:
        records = list(csv.DictReader(file))

    if len(records) != 11:
        raise ValueError(
            f"Expected 11 records, found {len(records)}"
        )

    output_rows = []
    unmatched = []
    ambiguous = []

    for record in records:
        audit_id = record["AuditID"]
        filename = record["SourceDocument"]
        fragment = FRAGMENTS[audit_id]

        path = PURE_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        if path.suffix.lower() == ".pdf":
            candidates = context_rows_for_pdf(
                record, filename, fragment
            )
        elif path.suffix.lower() == ".doc":
            candidates = context_rows_for_doc(
                record, filename, fragment
            )
        else:
            raise ValueError(
                f"Unsupported file type: {filename}"
            )

        count = len(candidates)

        if count == 0:
            unmatched.append(audit_id)
        elif count > 1:
            ambiguous.append(audit_id)

        status = (
            "No match"
            if count == 0
            else "Multiple occurrences - review required"
            if count > 1
            else "Source fragment located"
        )

        # Preserve unmatched records in the output too.
        if not candidates:
            candidates = [{
                "SourceDocument": filename,
                "SourceFormat": path.suffix.lower()[1:].upper(),
                "SourceLocation": "",
                "SourceRequirementID": "",
                "SourceSection": "",
                "OriginalRequirement": "",
                "SurroundingContext": "",
            }]

        for candidate in candidates:
            output_rows.append({
                "AuditID": audit_id,
                "RecordID": record["RecordID"],
                "ProjectID": record["ProjectID"],
                "OriginalLabel": record["OriginalLabel"],
                "AISuggestedLabel": record["AISuggestedLabel"],
                **candidate,
                "OccurrenceCount": count,
                "MatchStatus": status,
                "ContextVerification": "Pending manual review",
                "ReviewerDecision": "",
                "ReviewerNotes": "",
            })

        print(
            f"{audit_id} | {filename} | "
            f"{count} occurrence(s) | {status}"
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "experiment": "03",
        "name": "PURE Source Context Extraction",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "priority_records": len(records),
        "records_with_source_occurrences": (
            len(records) - len(unmatched)
        ),
        "total_candidate_occurrences": sum(
            row["OccurrenceCount"]
            for row in output_rows
            if row["OccurrenceCount"] == 1
        ) + sum(
            output_rows[i]["OccurrenceCount"]
            for i in range(len(output_rows))
            if output_rows[i]["AuditID"] in ambiguous
            and (
                i == 0
                or output_rows[i - 1]["AuditID"]
                != output_rows[i]["AuditID"]
            )
        ),
        "ambiguous_record_ids": ambiguous,
        "unmatched_record_ids": unmatched,
        "independent_review_completed": False,
        "dataset_labels_modified": False,
    }

    REPORT_JSON.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("PHASE 7 EXPERIMENT 03")
    print("=" * 55)
    print(
        "Requirements with context:",
        len(records) - len(unmatched),
        "/", len(records),
    )
    print("Ambiguous requirements:", ambiguous)
    print("Unmatched requirements:", unmatched)
    print("CSV:", OUTPUT_CSV)
    print("Report:", REPORT_JSON)
    print("Original labels modified: No")
    print("Independent verification: Pending")


if __name__ == "__main__":
    main()
