
"""
Phase 7 Experiment 08D
PURE to PROMISE+ Source-Project Mapping

Primary evidence:
    08a_partial_matches.csv

Supplementary evidence:
    08b_fuzzy_candidates.csv (similarity >= 0.90)

Outputs:
    08d_project_mapping_candidates.csv
    08d_document_mapping_summary.csv
    08d_mapping_summary.json

Source-project relationships are candidates, not verified facts.
No original dataset is modified.
"""

from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

PARTIAL = TABLES / "08a_partial_matches.csv"
FUZZY = TABLES / "08b_fuzzy_candidates.csv"
INVENTORY = TABLES / "05_pure_document_statistics.csv"

PROJECT_OUTPUT = TABLES / "08d_project_mapping_candidates.csv"
DOCUMENT_OUTPUT = TABLES / "08d_document_mapping_summary.csv"
SUMMARY_OUTPUT = REPORTS / "08d_mapping_summary.json"

FUZZY_THRESHOLD = 0.90


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    partial = read_csv(PARTIAL)
    fuzzy = read_csv(FUZZY)
    inventory = read_csv(INVENTORY)

    # Group evidence by PURE document and PROMISE+ project.
    evidence = defaultdict(
        lambda: {
            "equal_candidates": set(),
            "partial_candidates": set(),
            "fuzzy_candidates": set(),
            "promise_records": set(),
        }
    )

    for row in partial:
        key = (
            row["DocumentName"],
            str(row["PROMISEProjectID"]).strip(),
        )

        candidate = row["CandidateID"]

        if row["MatchType"] == "Normalized equal":
            evidence[key]["equal_candidates"].add(candidate)
        else:
            evidence[key]["partial_candidates"].add(candidate)

        evidence[key]["promise_records"].add(
            str(row["PROMISERecordID"])
        )

    for row in fuzzy:
        score = float(row["Similarity"])

        if score < FUZZY_THRESHOLD:
            continue

        key = (
            row["DocumentName"],
            str(row["PROMISEProjectID"]).strip(),
        )

        evidence[key]["fuzzy_candidates"].add(
            row["CandidateID"]
        )

        # Fuzzy records do not count as confirmed matches.

    project_rows = []

    for (document, project), data in sorted(evidence.items()):
        direct_candidates = (
            data["equal_candidates"]
            | data["partial_candidates"]
        )

        project_rows.append({
            "DocumentName": document,
            "PROMISEProjectID": project,
            "EqualCandidates": len(
                data["equal_candidates"]
            ),
            "PartialCandidates": len(
                data["partial_candidates"]
            ),
            "UniqueDirectCandidates": len(
                direct_candidates
            ),
            "UniqueDirectPROMISERecords": len(
                data["promise_records"]
            ),
            "HighFuzzyCandidates": len(
                data["fuzzy_candidates"]
            ),
            "MappingStatus": "Unverified",
            "ReviewerNotes": "",
        })

    # Summarize candidate project mappings per document.
    by_document = defaultdict(list)

    for row in project_rows:
        by_document[row["DocumentName"]].append(row)

    document_rows = []

    for document in sorted(
        row["DocumentName"] for row in inventory
    ):
        candidates = by_document.get(document, [])

        ranked = sorted(
            candidates,
            key=lambda row: (
                -row["UniqueDirectCandidates"],
                -row["EqualCandidates"],
                -row["HighFuzzyCandidates"],
                row["PROMISEProjectID"],
            ),
        )

        top = ranked[0] if ranked else None

        direct_projects = [
            row for row in candidates
            if row["UniqueDirectCandidates"] > 0
        ]

        document_rows.append({
            "DocumentName": document,
            "CandidateProjectCount": len(candidates),
            "ProjectsWithDirectEvidence": len(
                direct_projects
            ),
            "TopCandidateProjectID": (
                top["PROMISEProjectID"] if top else ""
            ),
            "TopDirectCandidateCount": (
                top["UniqueDirectCandidates"] if top else 0
            ),
            "TopHighFuzzyCount": (
                top["HighFuzzyCandidates"] if top else 0
            ),
            "MappingStatus": "Unverified",
        })

    project_fields = [
        "DocumentName",
        "PROMISEProjectID",
        "EqualCandidates",
        "PartialCandidates",
        "UniqueDirectCandidates",
        "UniqueDirectPROMISERecords",
        "HighFuzzyCandidates",
        "MappingStatus",
        "ReviewerNotes",
    ]

    document_fields = [
        "DocumentName",
        "CandidateProjectCount",
        "ProjectsWithDirectEvidence",
        "TopCandidateProjectID",
        "TopDirectCandidateCount",
        "TopHighFuzzyCount",
        "MappingStatus",
    ]

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    save_csv(
        PROJECT_OUTPUT, project_rows, project_fields
    )

    save_csv(
        DOCUMENT_OUTPUT, document_rows, document_fields
    )

    summary = {
        "experiment": "08D",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pure_documents": len(document_rows),
        "candidate_document_project_relationships": len(
            project_rows
        ),
        "documents_with_direct_evidence": sum(
            row["ProjectsWithDirectEvidence"] > 0
            for row in document_rows
        ),
        "documents_without_direct_evidence": sum(
            row["ProjectsWithDirectEvidence"] == 0
            for row in document_rows
        ),
        "fuzzy_similarity_threshold": FUZZY_THRESHOLD,
        "project_mappings_verified": 0,
        "evaluation_partition_leakage_checked": False,
        "original_datasets_modified": False,
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 08D")
    print("=" * 55)
    print("PURE documents:", len(document_rows))
    print(
        "Candidate document-project relationships:",
        len(project_rows),
    )
    print(
        "Documents with direct evidence:",
        summary["documents_with_direct_evidence"],
    )
    print(
        "Documents without direct evidence:",
        summary["documents_without_direct_evidence"],
    )
    print("Verified mappings: 0")
    print("Project mapping:", PROJECT_OUTPUT)
    print("Document summary:", DOCUMENT_OUTPUT)
    print("Report:", SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
