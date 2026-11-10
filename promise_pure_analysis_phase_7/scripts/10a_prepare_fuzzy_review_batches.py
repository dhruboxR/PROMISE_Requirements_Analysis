
"""
Phase 7 Experiment 10A
Prepare PURE-PROMISE+ fuzzy-match pairs for AI-assisted review.

- Uses the existing 120-pair stratified review sample
- Creates 12 review batches containing 10 pairs each
- Preserves full source text
- Creates a blank decision template
- Does not generate or invent review decisions
- Does not modify the original datasets
"""

from pathlib import Path
from datetime import datetime, timezone
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "outputs/tables/08c_manual_review_sample.csv"
)

BATCH_DIR = (
    ROOT / "outputs/fuzzy_review_batches"
)

DECISION_FILE = (
    ROOT / "outputs/tables/10a_fuzzy_review_decisions.csv"
)

REPORT_FILE = (
    ROOT / "outputs/reports/10a_review_batch_summary.json"
)

BATCH_SIZE = 10

REQUIRED_COLUMNS = [
    "CandidateID",
    "DocumentName",
    "PROMISERecordID",
    "PROMISEProjectID",
    "Similarity",
    "SimilarityGroup",
    "PUREText",
    "PROMISEText",
]

DECISION_COLUMNS = [
    "ReviewID",
    "Batch",
    "CandidateID",
    "DocumentName",
    "PROMISERecordID",
    "PROMISEProjectID",
    "Similarity",
    "SimilarityGroup",
    "ReviewerDecision",
    "ReviewerConfidence",
    "ReviewerNotes",
    "ReviewMethod",
]


def load_rows():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    with INPUT.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        missing = set(REQUIRED_COLUMNS) - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        rows = list(reader)

    if len(rows) != 120:
        raise ValueError(
            f"Expected 120 pairs, found {len(rows)}"
        )

    ids = [r["CandidateID"] for r in rows]

    if len(set(ids)) != 120:
        raise ValueError(
            "Expected 120 unique PURE candidates"
        )

    for row in rows:
        float(row["Similarity"])

        if not row["PUREText"].strip():
            raise ValueError("Empty PURE text")

        if not row["PROMISEText"].strip():
            raise ValueError("Empty PROMISE+ text")

    return rows


def main():
    rows = load_rows()

    # Preserve the existing stratified sample.
    # Sorting changes order, not the selected records.
    rows.sort(
        key=lambda r: (
            -float(r["Similarity"]),
            r["CandidateID"]
        )
    )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    DECISION_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    REPORT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    if DECISION_FILE.exists():
        raise FileExistsError(
            "Decision template already exists. "
            "Preserve completed reviews before rerunning."
        )

    decisions = []
    batch_count = 0

    for start in range(0, len(rows), BATCH_SIZE):
        batch_count += 1
        batch = rows[start:start + BATCH_SIZE]

        lines = [
            f"# Fuzzy Review Batch {batch_count:02d}",
            "",
            "Assess whether each PURE passage and "
            "PROMISE+ passage express the same requirement.",
            "",
            "Decisions: Same requirement | Partial overlap "
            "| Different requirement | Uncertain",
            "",
            "Record a confidence level and a brief reason. "
            "Do not treat similarity as proof.",
            "",
        ]

        for offset, row in enumerate(batch):
            index = start + offset + 1
            review_id = f"FUZZY-{index:03d}"

            lines.extend([
                "---",
                "",
                f"## {review_id}",
                "",
                f"CandidateID: {row['CandidateID']}",
                f"Document: {row['DocumentName']}",
                f"PROMISE+ RecordID: "
                f"{row['PROMISERecordID']}",
                f"PROMISE+ ProjectID: "
                f"{row['PROMISEProjectID']}",
                f"Similarity: {row['Similarity']}",
                f"Similarity group: "
                f"{row['SimilarityGroup']}",
                "",
                "### PURE passage",
                "",
                row["PUREText"],
                "",
                "### PROMISE+ requirement",
                "",
                row["PROMISEText"],
                "",
            ])

            decisions.append({
                "ReviewID": review_id,
                "Batch": batch_count,
                "CandidateID": row["CandidateID"],
                "DocumentName": row["DocumentName"],
                "PROMISERecordID": row["PROMISERecordID"],
                "PROMISEProjectID": row["PROMISEProjectID"],
                "Similarity": row["Similarity"],
                "SimilarityGroup": row[
                    "SimilarityGroup"
                ],
                "ReviewerDecision": "",
                "ReviewerConfidence": "",
                "ReviewerNotes": "",
                "ReviewMethod": "",
            })

        path = (
            BATCH_DIR
            / f"10a_review_batch_{batch_count:02d}.txt"
        )

        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

    with DECISION_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=DECISION_COLUMNS
        )
        writer.writeheader()
        writer.writerows(decisions)

    report = {
        "experiment": "10A",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "review_pairs": len(rows),
        "batch_size": BATCH_SIZE,
        "batches": batch_count,
        "review_decisions_completed": 0,
        "review_method": (
            "Pending AI-assisted contextual review"
        ),
        "independent_expert_review": False,
        "original_datasets_modified": False,
    }

    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nPHASE 7 EXPERIMENT 10A")
    print("=" * 55)
    print("Review pairs:", len(rows))
    print("Batch size:", BATCH_SIZE)
    print("Batches created:", batch_count)
    print("Completed decisions: 0")
    print("Independent expert reviews: 0")
    print("Original datasets modified: No")
    print("\nBatch directory:", BATCH_DIR)
    print("Decision template:", DECISION_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
