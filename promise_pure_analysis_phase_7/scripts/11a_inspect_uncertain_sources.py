
"""
Phase 7 Experiment 11A
Inspect original PURE text around four uncertain fuzzy matches.

Uses existing extracted source texts from Experiment 05.

Outputs:
    outputs/reports/11a_source_context.txt
    outputs/tables/11a_source_context_review.csv

Does not automatically change fuzzy-match decisions.
"""

from pathlib import Path
import csv
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "outputs/extracted_text"
REPORT = ROOT / "outputs/reports/11a_source_context.txt"
TABLE = ROOT / "outputs/tables/11a_source_context_review.csv"

CASES = [
    {
        "ReviewID": "FUZZY-035",
        "Document": "1999 - multi-mahjong.html",
        "Terms": [
            "choose to pick up",
            "choose to discard",
            "join information",
            "pick up",
            "join",
        ],
    },
    {
        "ReviewID": "FUZZY-042",
        "Document": "1999 - multi-mahjong.html",
        "Terms": [
            "choose to discard",
            "choose to pick up",
            "join information",
            "discard",
            "join",
        ],
    },
    {
        "ReviewID": "FUZZY-115",
        "Document": "2004 - jse.doc",
        "Terms": [
            "This should not have any contents",
            "should not have any contents",
            "any contents",
        ],
    },
    {
        "ReviewID": "FUZZY-119",
        "Document": "2005 - clarus low.pdf",
        "Terms": [
            "P-025",
            "F-501",
            "QEDS",
        ],
    },
]


def normalize_name(value):
    value = unicodedata.normalize("NFKC", value)
    return re.sub(
        r"[^a-z0-9]+", " ", value.lower()
    ).strip()


def find_document(document):
    if not TEXT_DIR.is_dir():
        raise FileNotFoundError(
            f"Extracted text directory not found: {TEXT_DIR}"
        )

    files = sorted(
        path for path in TEXT_DIR.rglob("*.txt")
        if path.is_file()
    )

    if not files:
        raise FileNotFoundError(
            "No extracted TXT files found"
        )

    target = normalize_name(
        Path(document).stem
    )

    matches = [
        path for path in files
        if target in normalize_name(path.stem)
    ]

    # Permit exactly one source text, never guess
    # between ambiguous extraction files.
    if len(matches) != 1:
        print(
            "\nAvailable extracted files "
            "matching document keywords:"
        )

        keywords = target.split()
        for path in files:
            name = normalize_name(path.stem)
            if any(word in name for word in keywords):
                print(" ", path.name)

        raise ValueError(
            f"Expected one extracted file for {document}, "
            f"found {len(matches)}"
        )

    return matches[0]


def search_context(text, term, radius=450, limit=3):
    # Whitespace-tolerant search supports line wrapping.
    tokens = term.split()

    expression = r"\s+".join(
        re.escape(token)
        for token in tokens
    )

    results = []

    for match in re.finditer(
        expression,
        text,
        flags=re.IGNORECASE,
    ):
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)

        snippet = text[start:end]

        results.append({
            "Start": match.start(),
            "Snippet": snippet.strip(),
        })

        if len(results) >= limit:
            break

    return results


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    TABLE.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "PHASE 7 EXPERIMENT 11A",
        "Original PURE Source-Context Inspection",
        "=" * 65,
        "",
        "Automatic extraction only.",
        "No match decisions have been changed.",
        "",
    ]

    records = []

    for case in CASES:
        review_id = case["ReviewID"]
        document = case["Document"]

        path = find_document(document)

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        lines.extend([
            "=" * 65,
            f"REVIEW ID: {review_id}",
            f"DOCUMENT: {document}",
            f"TEXT SOURCE: {path}",
            "",
        ])

        found_any = False

        for term in case["Terms"]:
            matches = search_context(text, term)

            lines.append(
                f"SEARCH TERM: {term}"
            )
            lines.append(
                f"Matches displayed: {len(matches)}"
            )

            for number, match in enumerate(
                matches, 1
            ):
                found_any = True

                lines.extend([
                    f"\n--- Match {number} ---",
                    match["Snippet"],
                    "",
                ])

            if not matches:
                lines.append("No match found.\n")

        records.append({
            "ReviewID": review_id,
            "DocumentName": document,
            "ExtractedTextFile": str(path),
            "ContextFound": (
                "Yes" if found_any else "No"
            ),
            "UpdatedDecision": "",
            "EvidenceNotes": "",
            "VerificationMethod": "",
            "ResearcherConfirmation": "",
        })

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    with TABLE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nPHASE 7 EXPERIMENT 11A")
    print("=" * 55)
    print("Cases inspected:", len(records))
    print(
        "Cases with matching text:",
        sum(
            row["ContextFound"] == "Yes"
            for row in records
        ),
    )
    print("Review decisions changed: 0")
    print("Original datasets modified: No")
    print("\nContext report:", REPORT)
    print("Review template:", TABLE)


if __name__ == "__main__":
    main()

