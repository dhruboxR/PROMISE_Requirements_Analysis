
"""
Phase 7 - Experiment 06
PURE Requirement Candidate Extraction

Purpose:
Identify requirement-like passages using linguistic patterns.

This experiment does NOT:
- Establish a verified requirement count.
- Assign PROMISE+ classification labels.
- Modify the original datasets.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]

STATS_CSV = (
    ROOT / "outputs/tables/05_pure_document_statistics.csv"
)

OUTPUT_CSV = (
    ROOT / "outputs/tables/06_requirement_candidates.csv"
)

SUMMARY_JSON = (
    ROOT / "outputs/reports/06_candidate_summary.json"
)

# Patterns are deliberately limited to common requirement wording.
PATTERNS = {
    "shall": re.compile(r"\bshall\b", re.I),
    "must": re.compile(r"\bmust\b", re.I),
    "should": re.compile(r"\bshould\b", re.I),
    "required_to": re.compile(
        r"\brequired\s+to\b", re.I
    ),
    "will_be_able_to": re.compile(
        r"\bwill\s+be\s+able\s+to\b", re.I
    ),
    "system_can": re.compile(
        r"\b(?:system|application|software)\s+can\b",
        re.I
    ),
}

FIELDS = [
    "CandidateID",
    "DocumentName",
    "Format",
    "ExtractedLine",
    "CandidateText",
    "MatchedPatterns",
    "CharacterCount",
    "WordCount",
    "ReviewStatus",
]


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_candidates(text, document_name, file_format):
    """
    Join adjacent nonempty lines into paragraphs, then split
    into sentence-like units.

    Source locations refer to the first line of a paragraph.
    They are approximate when a paragraph has many sentences.
    """
    paragraphs = []
    buffer = []
    first_line = None

    for line_number, line in enumerate(
        text.splitlines(), start=1
    ):
        stripped = line.strip()

        if not stripped:
            if buffer:
                paragraphs.append(
                    (first_line, normalize(" ".join(buffer)))
                )
                buffer = []
                first_line = None
            continue

        if first_line is None:
            first_line = line_number

        buffer.append(stripped)

    if buffer:
        paragraphs.append(
            (first_line, normalize(" ".join(buffer)))
        )

    candidates = []

    for first_line, paragraph in paragraphs:
        # A preliminary sentence splitter. It does not fully
        # handle abbreviations, decimal numbers, or list items.
        sentences = re.split(
            r"(?<=[.!?])\s+(?=[A-Z])",
            paragraph
        )

        for sentence in sentences:
            sentence = normalize(sentence)

            if len(sentence) < 20:
                continue

            if len(sentence) > 1500:
                continue

            matched = [
                name
                for name, pattern in PATTERNS.items()
                if pattern.search(sentence)
            ]

            if not matched:
                continue

            candidates.append({
                "DocumentName": document_name,
                "Format": file_format,
                "ExtractedLine": first_line,
                "CandidateText": sentence,
                "MatchedPatterns": ";".join(matched),
                "CharacterCount": len(sentence),
                "WordCount": len(sentence.split()),
                "ReviewStatus": "Unverified candidate",
            })

    return candidates


def main():
    if not STATS_CSV.exists():
        raise FileNotFoundError(STATS_CSV)

    with STATS_CSV.open(
        encoding="utf-8", newline=""
    ) as file:
        documents = list(csv.DictReader(file))

    results = []
    document_counts = {}
    pattern_counts = Counter()
    skipped = []

    for doc in documents:
        name = doc["DocumentName"]

        if doc["ExtractionStatus"] != "Success":
            skipped.append(name)
            continue

        relative_path = doc["ExtractedTextFile"]
        text_file = ROOT / relative_path

        if not text_file.exists():
            skipped.append(name)
            continue

        text = text_file.read_text(
            encoding="utf-8", errors="replace"
        )

        candidates = extract_candidates(
            text,
            name,
            doc["Format"],
        )

        document_counts[name] = len(candidates)

        for candidate in candidates:
            candidate["CandidateID"] = (
                f"PURE-CAND-{len(results) + 1:06d}"
            )

            for pattern in candidate[
                "MatchedPatterns"
            ].split(";"):
                pattern_counts[pattern] += 1

            results.append(candidate)

        print(
            f"{name}: "
            f"{len(candidates)} candidate passages"
        )

    OUTPUT_CSV.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "experiment": "06",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "documents_in_inventory": len(documents),
        "documents_processed": len(document_counts),
        "documents_skipped": skipped,
        "total_candidate_passages": len(results),
        "documents_with_candidates": sum(
            count > 0
            for count in document_counts.values()
        ),
        "pattern_frequencies": dict(
            sorted(pattern_counts.items())
        ),
        "top_10_documents": [
            {
                "document": name,
                "candidates": count,
            }
            for name, count in sorted(
                document_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        ],
        "verified_requirement_count": None,
        "human_review_completed": False,
        "dataset_labels_modified": False,
    }

    SUMMARY_JSON.parent.mkdir(
        parents=True, exist_ok=True
    )

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("PHASE 7 EXPERIMENT 06 SUMMARY")
    print("=" * 55)
    print("Documents processed:", len(document_counts))
    print("Total candidate passages:", len(results))
    print("Documents skipped:", len(skipped))
    print("Verified requirements: Not established")
    print("Labels assigned: No")
    print("Candidate CSV:", OUTPUT_CSV)
    print("Summary:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
