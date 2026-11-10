
"""Phase 7 Experiment 02: Match 11 priority records to PURE documents."""

from pathlib import Path
import csv
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PURE_DIR = ROOT.parent / "dataset" / "Pure" / "req"
PHASE6_DIR = ROOT.parent / "promise_advanced_improvement_phase_6"

INPUT_FILE = (
    PHASE6_DIR
    / "outputs"
    / "annotation_audit"
    / "08_priority_source_verification.csv"
)

OUTPUT_FILE = (
    ROOT / "outputs" / "tables" / "02_source_matches.csv"
)

# Search the PDF and DOC versions independently.
CANDIDATES = {
    "52": ["1995 - gemini.pdf"],
    "80": ["2009 - library.pdf", "2009 - library.doc"],
    "86": ["2010 - fishing.pdf"],
}

# Distinctive phrases for source matching.
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


def normalize_with_positions(text):
    result = []
    positions = []

    for index, char in enumerate(text):
        if char.isalnum():
            for normalized_char in char.casefold():
                result.append(normalized_char)
                positions.append(index)

    return "".join(result), positions


def run_command(command):
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return result.stdout


def extract_document(path):
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = run_command([
            "pdftotext", "-layout", str(path), "-"
        ])
        return [
            ("PDF page", page_number, page)
            for page_number, page in enumerate(
                text.split("\f"), start=1
            )
            if page.strip()
        ]

    if suffix == ".doc":
        text = run_command(["antiword", str(path)])
        # Treat the DOC as one continuous text segment.
        return [("DOC line", None, text)]

    raise ValueError(f"Unsupported format: {suffix}")


def locate_fragment(text, fragment):
    normalized_text, positions = normalize_with_positions(text)
    normalized_fragment, _ = normalize_with_positions(fragment)

    if not normalized_fragment:
        return None

    position = normalized_text.find(normalized_fragment)

    if position < 0:
        return None

    raw_start = positions[position]
    raw_end = positions[position + len(normalized_fragment) - 1] + 1

    excerpt = re.sub(
        r"\s+",
        " ",
        text[max(0, raw_start - 200):raw_end + 200],
    ).strip()

    # Line numbering is based on extracted plain text.
    start_line = text.count("\n", 0, raw_start) + 1
    end_line = text.count("\n", 0, raw_end) + 1

    return excerpt, start_line, end_line


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with INPUT_FILE.open(
        encoding="utf-8", newline=""
    ) as file:
        records = list(csv.DictReader(file))

    if len(records) != 11:
        raise ValueError(
            f"Expected 11 priority records, found {len(records)}"
        )

    cache = {}
    results = []

    for record in records:
        audit_id = record["AuditID"]
        project_id = str(record["ProjectID"]).strip()

        if project_id not in CANDIDATES:
            raise ValueError(f"Unexpected ProjectID: {project_id}")

        if audit_id not in FRAGMENTS:
            raise ValueError(f"Missing fragment: {audit_id}")

        matches = []

        for filename in CANDIDATES[project_id]:
            path = PURE_DIR / filename

            if filename not in cache:
                if not path.exists():
                    raise FileNotFoundError(path)

                cache[filename] = extract_document(path)

            for location_type, page, text in cache[filename]:
                found = locate_fragment(
                    text,
                    FRAGMENTS[audit_id]
                )

                if found is None:
                    continue

                excerpt, start_line, end_line = found

                if location_type == "PDF page":
                    location = f"PDF page {page}"
                else:
                    location = (
                        f"DOC extracted lines "
                        f"{start_line}-{end_line}"
                    )

                matches.append({
                    "filename": filename,
                    "location": location,
                    "excerpt": excerpt,
                })

        if matches:
            first = matches[0]
            status = (
                "Fragment matched"
                if len(matches) == 1
                else "Multiple fragment matches"
            )
        else:
            first = {
                "filename": "",
                "location": "",
                "excerpt": "",
            }
            status = "Not found - manual review"

        results.append({
            "AuditID": audit_id,
            "RecordID": record["RecordID"],
            "ProjectID": project_id,
            "OriginalLabel": record["OriginalLabel"],
            "AISuggestedLabel": record["SuggestedLabel"],
            "SourceDocument": first["filename"],
            "SourceLocation": first["location"],
            "MatchStatus": status,
            "MatchCount": len(matches),
            "SourceExcerpt": first["excerpt"],
            "SourceLabelEvidence": "",
            "IndependentReviewerDecision": "",
            "VerificationStatus": "Pending contextual review",
        })

        print(
            f"{audit_id} | Project {project_id} | "
            f"{first['filename'] or 'Unknown'} | "
            f"{first['location'] or 'Unknown'} | "
            f"{status}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    matched = sum(
        result["MatchCount"] > 0
        for result in results
    )

    print("\n" + "=" * 55)
    print("PHASE 7 EXPERIMENT 02 RESULTS")
    print("=" * 55)
    print(f"Matched records: {matched}/{len(results)}")
    print(f"Unmatched records: {len(results) - matched}")
    print(f"Saved: {OUTPUT_FILE}")
    print("Dataset labels modified: No")
    print("Independent annotation verification: Pending")


if __name__ == "__main__":
    main()
