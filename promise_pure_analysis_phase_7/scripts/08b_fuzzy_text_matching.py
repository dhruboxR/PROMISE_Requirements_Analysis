
"""
Phase 7 Experiment 08B
Fuzzy PURE-PROMISE+ matching using character TF-IDF.

Only candidates unmatched in Experiment 08A are searched.
Similarity scores are provisional evidence, not verified matches.
No original dataset or classification label is modified.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import json
import re
import unicodedata

import arff
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

CANDIDATE_FILE = (
    ROOT / "outputs/tables/06_requirement_candidates.csv"
)
PARTIAL_FILE = (
    ROOT / "outputs/tables/08a_partial_matches.csv"
)

ARFF_OPTIONS = [
    THESIS / "promise_exploration_phase_1"
    / "outputs/validated/Promise+_repaired.arff",
    THESIS / "promise_exploration_p1"
    / "outputs/validated/Promise+_repaired.arff",
]

TABLE_DIR = ROOT / "outputs/tables"
REPORT_DIR = ROOT / "outputs/reports"

RESULT_FILE = TABLE_DIR / "08b_fuzzy_candidates.csv"
SUMMARY_FILE = REPORT_DIR / "08b_fuzzy_summary.json"

TOP_K = 3
REVIEW_THRESHOLD = 0.80
BATCH_SIZE = 50

FIELDS = [
    "CandidateID",
    "DocumentName",
    "ExtractedLine",
    "PROMISERecordID",
    "PROMISEProjectID",
    "PROMISELabel",
    "Similarity",
    "ReviewPriority",
    "PUREText",
    "PROMISEText",
    "ManualReviewStatus",
]


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def read_csv(path):
    with path.open(
        encoding="utf-8", newline=""
    ) as file:
        return list(csv.DictReader(file))


def load_promise(path):
    with path.open(encoding="utf-8") as file:
        data = arff.load(file)

    columns = [
        attribute[0]
        for attribute in data["attributes"]
    ]

    required = (
        "ProjectID", "RequirementText", "_class_"
    )

    if not all(field in columns for field in required):
        raise ValueError("Required ARFF columns are missing")

    return [
        dict(zip(columns, values))
        for values in data["data"]
    ]


def main():
    for path in (CANDIDATE_FILE, PARTIAL_FILE):
        if not path.exists():
            raise FileNotFoundError(path)

    found = [p for p in ARFF_OPTIONS if p.is_file()]

    if len(found) != 1:
        raise RuntimeError(
            "Select exactly one repaired PROMISE+ ARFF. "
            f"Found: {[str(p) for p in found]}"
        )

    pure = read_csv(CANDIDATE_FILE)
    partial_matches = read_csv(PARTIAL_FILE)
    promise = load_promise(found[0])

    if len(pure) != 8462 or len(promise) != 3677:
        raise ValueError(
            "Dataset counts differ from Experiments 06/07"
        )

    matched_ids = {
        row["CandidateID"]
        for row in partial_matches
    }

    unmatched = [
        row for row in pure
        if row["CandidateID"] not in matched_ids
    ]

    if len(unmatched) != 7264:
        print(
            "WARNING: Expected 7264 unmatched candidates, "
            f"found {len(unmatched)}"
        )

    print("PURE candidates:", len(pure))
    print("Already matched in 08A:", len(matched_ids))
    print("Candidates for fuzzy search:", len(unmatched))
    print("PROMISE+ requirements:", len(promise))

    pure_texts = [
        normalize(row["CandidateText"])
        for row in unmatched
    ]

    promise_texts = [
        normalize(row["RequirementText"])
        for row in promise
    ]

    print("\nBuilding character TF-IDF representations...")

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        dtype=np.float32,
    )

    # Fit on both corpora for a shared comparison space.
    combined = promise_texts + pure_texts
    matrix = vectorizer.fit_transform(combined)

    n_promise = len(promise)
    promise_matrix = matrix[:n_promise]
    pure_matrix = matrix[n_promise:]

    del matrix

    results = []
    priority_counts = Counter()
    best_scores = []

    print("\nComparing candidate passages...")

    for start in range(0, len(unmatched), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(unmatched))

        # Sparse matrix multiplication keeps memory usage
        # manageable for this dataset.
        similarities = (
            pure_matrix[start:stop]
            @ promise_matrix.T
        ).tocsr()

        for local_index in range(stop - start):
            pure_index = start + local_index
            row = unmatched[pure_index]

            similarity_row = similarities.getrow(local_index)

            indices = similarity_row.indices
            scores = similarity_row.data

            if len(scores) == 0:
                best_scores.append(0.0)
                continue

            # Sort deterministically by descending score,
            # then by ascending PROMISE+ row index.
            order = np.lexsort((indices, -scores))
            chosen = order[:TOP_K]

            best_scores.append(
                float(scores[chosen[0]])
            )

            for rank in chosen:
                index = int(indices[rank])
                score = float(scores[rank])

                priority = (
                    "High - manual review"
                    if score >= REVIEW_THRESHOLD
                    else "Lower similarity - unverified"
                )

                priority_counts[priority] += 1

                results.append({
                    "CandidateID": row["CandidateID"],
                    "DocumentName": row["DocumentName"],
                    "ExtractedLine": row["ExtractedLine"],
                    "PROMISERecordID": index + 1,
                    "PROMISEProjectID": promise[index][
                        "ProjectID"
                    ],
                    "PROMISELabel": promise[index][
                        "_class_"
                    ],
                    "Similarity": round(score, 6),
                    "ReviewPriority": priority,
                    "PUREText": row["CandidateText"],
                    "PROMISEText": promise[index][
                        "RequirementText"
                    ],
                    "ManualReviewStatus": "Not reviewed",
                })

        print(
            f"Compared {stop}/{len(unmatched)}",
            end="\r",
            flush=True,
        )

    print()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with RESULT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(results)

    best = np.array(best_scores, dtype=float)

    summary = {
        "experiment": "08B",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pure_total_candidates": len(pure),
        "promise_records": len(promise),
        "previously_matched_candidates": len(matched_ids),
        "unmatched_candidates_searched": len(unmatched),
        "top_k": TOP_K,
        "review_threshold": REVIEW_THRESHOLD,
        "candidate_pairs_saved": len(results),
        "candidates_with_best_similarity_at_least_0_80": int(
            np.sum(best >= 0.80)
        ),
        "candidates_with_best_similarity_at_least_0_90": int(
            np.sum(best >= 0.90)
        ),
        "mean_best_similarity": (
            round(float(best.mean()), 4)
            if len(best) else 0
        ),
        "priority_pair_counts": dict(priority_counts),
        "human_matches_verified": 0,
        "new_requirements_confirmed": 0,
        "training_dataset_created": False,
        "original_datasets_modified": False,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 08B")
    print("=" * 55)
    print("Candidates searched:", len(unmatched))
    print("Candidate pairs saved:", len(results))
    print(
        "Best similarity >= 0.80:",
        summary[
            "candidates_with_best_similarity_at_least_0_80"
        ],
    )
    print(
        "Best similarity >= 0.90:",
        summary[
            "candidates_with_best_similarity_at_least_0_90"
        ],
    )
    print("Mean best similarity:", summary["mean_best_similarity"])
    print("CSV:", RESULT_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Verified fuzzy matches: 0")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
