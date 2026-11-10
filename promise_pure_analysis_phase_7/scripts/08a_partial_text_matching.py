
"""
Phase 7, Experiment 08A
PURE-PROMISE+ normalized substring matching.

Uses a word n-gram index to identify candidate pairs,
then verifies complete normalized-text containment.

This does not identify semantic paraphrases.
No original datasets are modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import json
import re
import unicodedata

import arff

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

PURE_CSV = ROOT / "outputs/tables/06_requirement_candidates.csv"

ARFF_OPTIONS = [
    THESIS / "promise_exploration_phase_1"
    / "outputs/validated/Promise+_repaired.arff",
    THESIS / "promise_exploration_p1"
    / "outputs/validated/Promise+_repaired.arff",
]

OUT_TABLES = ROOT / "outputs/tables"
OUT_REPORTS = ROOT / "outputs/reports"

OUTPUT_CSV = OUT_TABLES / "08a_partial_matches.csv"
DOC_CSV = OUT_TABLES / "08a_document_summary.csv"
SUMMARY_JSON = OUT_REPORTS / "08a_partial_summary.json"

NGRAM_SIZE = 7
MIN_WORDS = 8
MIN_CHARACTERS = 50


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_promise(path):
    with path.open(encoding="utf-8") as f:
        data = arff.load(f)

    columns = [attr[0] for attr in data["attributes"]]

    for required in ("ProjectID", "RequirementText", "_class_"):
        if required not in columns:
            raise ValueError(f"Missing column: {required}")

    return [
        dict(zip(columns, values))
        for values in data["data"]
    ]


def word_ngrams(words):
    return {
        tuple(words[i:i + NGRAM_SIZE])
        for i in range(len(words) - NGRAM_SIZE + 1)
    }


def sufficient_shared_text(shorter):
    return (
        len(shorter) >= MIN_CHARACTERS
        and len(shorter.split()) >= MIN_WORDS
    )


def classify_pair(pure_text, promise_text):
    if pure_text == promise_text:
        return "Normalized equal", pure_text

    if (
        sufficient_shared_text(promise_text)
        and f" {promise_text} " in f" {pure_text} "
    ):
        return "PROMISE inside PURE", promise_text

    if (
        sufficient_shared_text(pure_text)
        and f" {pure_text} " in f" {promise_text} "
    ):
        return "PURE inside PROMISE", pure_text

    return None, None


def main():
    if not PURE_CSV.exists():
        raise FileNotFoundError(PURE_CSV)

    found = [p for p in ARFF_OPTIONS if p.is_file()]

    if len(found) != 1:
        raise RuntimeError(
            "Expected one repaired PROMISE+ ARFF. "
            f"Found: {[str(p) for p in found]}. "
            "Edit ARFF_OPTIONS if necessary."
        )

    pure = read_csv(PURE_CSV)
    promise = load_promise(found[0])

    if len(promise) != 3677:
        raise ValueError(
            f"Expected 3677 PROMISE+ records; got {len(promise)}"
        )

    if len(pure) != 8462:
        print(
            "WARNING: PURE candidate count differs "
            f"from prior result: {len(pure)}"
        )

    # Map 7-word sequences to PROMISE+ ARFF row positions.
    index = defaultdict(set)
    promise_normalized = []

    for i, record in enumerate(promise):
        normalized = normalize(record["RequirementText"])
        promise_normalized.append(normalized)

        for gram in word_ngrams(normalized.split()):
            index[gram].add(i)

    results = []
    documents = defaultdict(Counter)
    matched_promise_ids = set()
    matched_candidate_ids = set()

    for candidate in pure:
        candidate_id = candidate["CandidateID"]
        document = candidate["DocumentName"]
        pure_text = normalize(candidate["CandidateText"])

        possible_ids = set()

        for gram in word_ngrams(pure_text.split()):
            possible_ids.update(index.get(gram, ()))

        candidate_has_match = False

        for i in sorted(possible_ids):
            promise_text = promise_normalized[i]

            match_type, shared = classify_pair(
                pure_text, promise_text
            )

            if match_type is None:
                continue

            # A match is text-containment evidence,
            # not confirmation of source-project identity.
            results.append({
                "CandidateID": candidate_id,
                "DocumentName": document,
                "PROMISERecordID": i + 1,
                "PROMISEProjectID": promise[i]["ProjectID"],
                "PROMISELabel": promise[i]["_class_"],
                "MatchType": match_type,
                "SharedText": shared,
                "PUREText": candidate["CandidateText"],
                "PROMISEText": promise[i]["RequirementText"],
                "ReviewStatus": "Unverified text match",
            })

            candidate_has_match = True
            documents[document][match_type] += 1
            matched_promise_ids.add(i + 1)

        if candidate_has_match:
            matched_candidate_ids.add(candidate_id)
            documents[document]["MatchedCandidates"] += 1
        else:
            documents[document]["UnmatchedCandidates"] += 1

    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)

    fields = [
        "CandidateID", "DocumentName", "PROMISERecordID",
        "PROMISEProjectID", "PROMISELabel", "MatchType",
        "SharedText", "PUREText", "PROMISEText",
        "ReviewStatus",
    ]

    with OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    doc_fields = [
        "DocumentName",
        "MatchedCandidates",
        "UnmatchedCandidates",
        "MatchedPairs",
        "NormalizedEqualPairs",
        "PartialPairs",
    ]

    doc_rows = []

    for document, counts in sorted(documents.items()):
        doc_rows.append({
            "DocumentName": document,
            "MatchedCandidates": counts["MatchedCandidates"],
            "UnmatchedCandidates": counts["UnmatchedCandidates"],
            "MatchedPairs": (
                counts["Normalized equal"]
                + counts["PROMISE inside PURE"]
                + counts["PURE inside PROMISE"]
            ),
            "NormalizedEqualPairs": counts["Normalized equal"],
            "PartialPairs": (
                counts["PROMISE inside PURE"]
                + counts["PURE inside PROMISE"]
            ),
        })

    with DOC_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=doc_fields)
        writer.writeheader()
        writer.writerows(doc_rows)

    types = Counter(r["MatchType"] for r in results)

    summary = {
        "experiment": "08A",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pure_candidates": len(pure),
        "promise_records": len(promise),
        "total_matching_pairs": len(results),
        "normalized_equal_pairs": types["Normalized equal"],
        "promise_inside_pure_pairs": types["PROMISE inside PURE"],
        "pure_inside_promise_pairs": types["PURE inside PROMISE"],
        "unique_pure_candidates_matched": len(
            matched_candidate_ids
        ),
        "unique_promise_records_matched": len(
            matched_promise_ids
        ),
        "documents_with_matches": sum(
            row["MatchedCandidates"] > 0 for row in doc_rows
        ),
        "minimum_partial_words": MIN_WORDS,
        "minimum_partial_characters": MIN_CHARACTERS,
        "near_duplicates_verified": False,
        "source_projects_verified": False,
        "training_dataset_created": False,
        "original_datasets_modified": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 08A")
    print("=" * 55)
    print("PURE candidates:", len(pure))
    print("PROMISE+ requirements:", len(promise))
    print("Matching text pairs:", len(results))
    print("Normalized-equal pairs:", types["Normalized equal"])
    print("PROMISE inside PURE:", types["PROMISE inside PURE"])
    print("PURE inside PROMISE:", types["PURE inside PROMISE"])
    print("Unique PURE candidates matched:", len(matched_candidate_ids))
    print("Unique PROMISE+ records matched:", len(matched_promise_ids))
    print("Documents with matches:", summary["documents_with_matches"])
    print("CSV:", OUTPUT_CSV)
    print("Document summary:", DOC_CSV)
    print("JSON:", SUMMARY_JSON)
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
