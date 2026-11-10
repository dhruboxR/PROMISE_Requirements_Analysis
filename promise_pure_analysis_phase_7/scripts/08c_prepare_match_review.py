
"""Phase 7 Experiment 08C: Prepare fuzzy-match validation samples."""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import random

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "outputs/tables/08b_fuzzy_candidates.csv"
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

HIGH_FILE = TABLES / "08c_high_similarity_pairs.csv"
REVIEW_FILE = TABLES / "08c_manual_review_sample.csv"
SUMMARY_FILE = REPORTS / "08c_review_preparation_summary.json"

SEED = 42

# Sample counts are targets, not assumptions about bin sizes.
SAMPLE_TARGETS = {
    "0.90-1.00": 30,
    "0.80-0.90": 30,
    "0.60-0.80": 20,
    "0.40-0.60": 20,
    "0.00-0.40": 20,
}

REVIEW_FIELDS = [
    "CandidateID",
    "DocumentName",
    "ExtractedLine",
    "PROMISERecordID",
    "PROMISEProjectID",
    "PROMISELabel",
    "Similarity",
    "SimilarityGroup",
    "PUREText",
    "PROMISEText",
    "ReviewerDecision",
    "ReviewerConfidence",
    "ReviewerNotes",
]

VALID_DECISIONS = [
    "Same requirement",
    "Partial overlap",
    "Different requirement",
    "Uncertain",
]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REVIEW_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def similarity_group(score):
    if score >= 0.90:
        return "0.90-1.00"
    if score >= 0.80:
        return "0.80-0.90"
    if score >= 0.60:
        return "0.60-0.80"
    if score >= 0.40:
        return "0.40-0.60"
    return "0.00-0.40"


def add_review_fields(row):
    result = dict(row)
    result["SimilarityGroup"] = similarity_group(
        float(row["Similarity"])
    )
    result["ReviewerDecision"] = ""
    result["ReviewerConfidence"] = ""
    result["ReviewerNotes"] = ""
    return result


def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    rows = read_csv(SOURCE)

    # Select the highest-scoring PROMISE+ record for each
    # PURE candidate. Keep all ties in the source CSV.
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["Similarity"]),
            int(row["PROMISERecordID"]),
        ),
    )

    best_by_candidate = {}

    for row in ranked:
        best_by_candidate.setdefault(
            row["CandidateID"], row
        )

    best = [
        add_review_fields(row)
        for row in best_by_candidate.values()
    ]

    groups = {
        group: []
        for group in SAMPLE_TARGETS
    }

    for row in best:
        groups[row["SimilarityGroup"]].append(row)

    rng = random.Random(SEED)
    sample = []

    for group, target in SAMPLE_TARGETS.items():
        population = sorted(
            groups[group],
            key=lambda row: row["CandidateID"],
        )

        selected = rng.sample(
            population,
            min(target, len(population)),
        )

        sample.extend(selected)

    sample.sort(
        key=lambda row: (
            -float(row["Similarity"]),
            row["CandidateID"],
        )
    )

    high = [
        row for row in best
        if float(row["Similarity"]) >= 0.80
    ]

    high.sort(
        key=lambda row: -float(row["Similarity"])
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    save_csv(HIGH_FILE, high)
    save_csv(REVIEW_FILE, sample)

    summary = {
        "experiment": "08C",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "random_seed": SEED,
        "source_candidate_pairs": len(rows),
        "unique_candidates": len(best),
        "high_similarity_candidates": len(high),
        "manual_review_sample_size": len(sample),
        "similarity_group_population": {
            group: len(items)
            for group, items in groups.items()
        },
        "similarity_group_sample": dict(
            Counter(
                row["SimilarityGroup"]
                for row in sample
            )
        ),
        "valid_reviewer_decisions": VALID_DECISIONS,
        "manual_reviews_completed": 0,
        "training_dataset_created": False,
        "original_datasets_modified": False,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 08C")
    print("=" * 55)
    print("Candidate pairs read:", len(rows))
    print("Unique PURE candidates:", len(best))
    print("High similarity candidates:", len(high))
    print("Manual review sample:", len(sample))

    print("\nSimilarity groups:")
    for group in SAMPLE_TARGETS:
        print(
            group,
            "| Population:", len(groups[group]),
            "| Sample:",
            summary["similarity_group_sample"].get(group, 0),
        )

    print("\nHigh-similarity CSV:", HIGH_FILE)
    print("Manual review CSV:", REVIEW_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Completed manual reviews: 0")


if __name__ == "__main__":
    main()
