
"""
Phase 7 - Experiment 07
PURE vs PROMISE+ Exact and Normalized Text Overlap

Inputs:
    outputs/tables/06_requirement_candidates.csv
    Phase 1 repaired PROMISE+ ARFF

Outputs:
    outputs/tables/07_pure_promise_overlap.csv
    outputs/tables/07_document_overlap_summary.csv
    outputs/reports/07_overlap_summary.json

This analysis does not modify any dataset or create
a training dataset.
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

PURE_CSV = (
    ROOT / "outputs/tables/06_requirement_candidates.csv"
)

ARFF_OPTIONS = [
    THESIS / "promise_exploration_phase_1"
    / "outputs/validated/Promise+_repaired.arff",

    THESIS / "promise_exploration_p1"
    / "outputs/validated/Promise+_repaired.arff",
]

OUTPUT = ROOT / "outputs"
TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

FIELDS = [
    "CandidateID",
    "DocumentName",
    "CandidateText",
    "MatchType",
    "MatchedPROMISERecordIDs",
    "MatchedProjectIDs",
    "MatchedLabels",
    "MatchedRecordCount",
]


def exact_key(text):
    return str(text).strip()


def normalized_key(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()

    # Remove punctuation and normalize all whitespace.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def read_csv(path):
    with path.open(
        encoding="utf-8", newline=""
    ) as file:
        return list(csv.DictReader(file))


def load_promise(path):
    with path.open(encoding="utf-8") as file:
        dataset = arff.load(file)

    columns = [
        item[0] for item in dataset["attributes"]
    ]

    required = [
        "ProjectID",
        "RequirementText",
        "_class_",
    ]

    for column in required:
        if column not in columns:
            raise ValueError(
                f"Missing PROMISE+ column: {column}"
            )

    return [
        dict(zip(columns, values))
        for values in dataset["data"]
    ]


def main():
    if not PURE_CSV.exists():
        raise FileNotFoundError(PURE_CSV)

    available = [
        path for path in ARFF_OPTIONS
        if path.is_file()
    ]

    if not available:
        raise FileNotFoundError(
            "Repaired PROMISE+ ARFF not found. "
            "Update ARFF_OPTIONS with its actual location."
        )

    if len(available) > 1:
        raise ValueError(
            "Multiple repaired ARFF files found. "
            "Select the intended version explicitly: "
            + ", ".join(map(str, available))
        )

    promise_path = available[0]

    candidates = read_csv(PURE_CSV)
    promise = load_promise(promise_path)

    if len(promise) != 3677:
        raise ValueError(
            f"Expected 3677 PROMISE+ records, "
            f"found {len(promise)}"
        )

    exact_index = defaultdict(list)
    normalized_index = defaultdict(list)

    for index, record in enumerate(promise):
        text = record["RequirementText"]

        # RecordID is the 1-based ARFF row position.
        # It is not an identifier supplied by the ARFF.
        entry = {
            "RecordID": index + 1,
            "ProjectID": str(record["ProjectID"]),
            "Label": str(record["_class_"]),
        }

        exact_index[exact_key(text)].append(entry)
        normalized_index[
            normalized_key(text)
        ].append(entry)

    results = []
    document_counts = defaultdict(Counter)
    matched_record_ids = set()

    for candidate in candidates:
        text = candidate["CandidateText"]

        exact = exact_index.get(
            exact_key(text), []
        )

        if exact:
            matches = exact
            match_type = "Exact"
        else:
            matches = normalized_index.get(
                normalized_key(text), []
            )
            match_type = (
                "Normalized"
                if matches else "Unmatched"
            )

        for match in matches:
            matched_record_ids.add(match["RecordID"])

        document = candidate["DocumentName"]
        document_counts[document][match_type] += 1

        results.append({
            "CandidateID": candidate["CandidateID"],
            "DocumentName": document,
            "CandidateText": text,
            "MatchType": match_type,
            "MatchedPROMISERecordIDs": ";".join(
                str(m["RecordID"]) for m in matches
            ),
            "MatchedProjectIDs": ";".join(
                sorted({
                    m["ProjectID"] for m in matches
                })
            ),
            "MatchedLabels": ";".join(
                sorted({
                    m["Label"] for m in matches
                })
            ),
            "MatchedRecordCount": len(matches),
        })

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    overlap_csv = (
        TABLES / "07_pure_promise_overlap.csv"
    )

    with overlap_csv.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(results)

    document_rows = []

    for document, counts in sorted(
        document_counts.items()
    ):
        document_rows.append({
            "DocumentName": document,
            "Exact": counts["Exact"],
            "Normalized": counts["Normalized"],
            "Unmatched": counts["Unmatched"],
            "Total": sum(counts.values()),
        })

    doc_csv = (
        TABLES / "07_document_overlap_summary.csv"
    )

    with doc_csv.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "DocumentName",
                "Exact",
                "Normalized",
                "Unmatched",
                "Total",
            ],
        )
        writer.writeheader()
        writer.writerows(document_rows)

    counts = Counter(
        row["MatchType"] for row in results
    )

    matched_candidates = (
        counts["Exact"] + counts["Normalized"]
    )

    summary = {
        "experiment": "07",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "promise_source": str(
            promise_path.relative_to(THESIS)
        ),
        "promise_records": len(promise),
        "pure_candidates": len(candidates),
        "exact_matches": counts["Exact"],
        "normalized_only_matches": counts["Normalized"],
        "unmatched_candidates": counts["Unmatched"],
        "matched_candidates": matched_candidates,
        "matched_candidate_percentage": round(
            100 * matched_candidates / len(candidates),
            2
        ) if candidates else 0,
        "distinct_promise_records_matched": len(
            matched_record_ids
        ),
        "documents_with_matches": sum(
            counts["Exact"] + counts["Normalized"] > 0
            for counts in document_counts.values()
        ),
        "near_duplicate_analysis_completed": False,
        "new_requirements_verified": False,
        "training_dataset_created": False,
        "original_datasets_modified": False,
    }

    report_path = (
        REPORTS / "07_overlap_summary.json"
    )

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 07")
    print("=" * 55)
    print("PROMISE+ records:", len(promise))
    print("PURE candidates:", len(candidates))
    print("Exact matches:", counts["Exact"])
    print("Normalized-only matches:", counts["Normalized"])
    print("Unmatched candidates:", counts["Unmatched"])
    print("Matched candidates:", matched_candidates)
    print(
        "Distinct PROMISE+ records matched:",
        len(matched_record_ids),
    )
    print(
        "Documents with matches:",
        summary["documents_with_matches"],
    )
    print("Candidate CSV:", overlap_csv)
    print("Document summary:", doc_csv)
    print("JSON report:", report_path)
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
