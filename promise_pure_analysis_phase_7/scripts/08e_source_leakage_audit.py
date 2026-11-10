
"""
Phase 7 Experiment 08E
PURE Source-Project Leakage Audit

Inputs:
- Phase 2 final_train.csv
- Phase 2 final_validation.csv
- Phase 2 final_test.csv
- Phase 7 08d_project_mapping_candidates.csv
- Phase 7 08d_document_mapping_summary.csv

Outputs:
- outputs/tables/08e_project_partition_mapping.csv
- outputs/tables/08e_document_leakage_audit.csv
- outputs/reports/08e_leakage_summary.json

No datasets, partitions, or labels are modified.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import json
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

SPLITS = THESIS / (
    "promise_preprocessing_phase_2/outputs/splits"
)

MAPPING = ROOT / (
    "outputs/tables/08d_project_mapping_candidates.csv"
)

DOCUMENTS = ROOT / (
    "outputs/tables/08d_document_mapping_summary.csv"
)

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

PROJECT_OUT = TABLES / "08e_project_partition_mapping.csv"
DOCUMENT_OUT = TABLES / "08e_document_leakage_audit.csv"
SUMMARY_OUT = REPORTS / "08e_leakage_summary.json"

EXPECTED_COUNTS = {
    "Train": 2574,
    "Validation": 553,
    "Test": 550,
}

TEXT_COLUMNS = (
    "NormalizedText",
    "NormalizedRequirementText",
    "RequirementText",
)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def project_id(value):
    return str(value).strip()


def load_splits():
    files = {
        "Train": SPLITS / "final_train.csv",
        "Validation": SPLITS / "final_validation.csv",
        "Test": SPLITS / "final_test.csv",
    }

    project_partition = {}
    partition_projects = {}
    record_partitions = defaultdict(set)
    text_partitions = defaultdict(set)
    text_columns_used = {}

    for partition, path in files.items():
        rows = read_csv(path)

        if len(rows) != EXPECTED_COUNTS[partition]:
            raise ValueError(
                f"{partition} expected "
                f"{EXPECTED_COUNTS[partition]} records, "
                f"found {len(rows)}"
            )

        if not rows:
            raise ValueError(f"Empty partition: {partition}")

        required = {"RecordID", "ProjectID"}

        if not required.issubset(rows[0]):
            raise ValueError(
                f"Missing required columns in {path}"
            )

        projects = set()

        text_column = next(
            (
                col for col in TEXT_COLUMNS
                if col in rows[0]
            ),
            None,
        )
        text_columns_used[partition] = text_column

        for row in rows:
            pid = project_id(row["ProjectID"])
            rid = str(row["RecordID"]).strip()

            projects.add(pid)
            record_partitions[rid].add(partition)

            if text_column:
                normalized = normalize_text(
                    row[text_column]
                )
                if normalized:
                    text_partitions[normalized].add(
                        partition
                    )

            old = project_partition.get(pid)

            if old is not None and old != partition:
                raise ValueError(
                    f"Project {pid} appears in "
                    f"both {old} and {partition}"
                )

            project_partition[pid] = partition

        partition_projects[partition] = projects

    repeated_record_ids = [
        rid for rid, partitions
        in record_partitions.items()
        if len(partitions) > 1
    ]

    if repeated_record_ids:
        raise ValueError(
            "RecordIDs repeated across partitions: "
            f"{repeated_record_ids[:10]}"
        )

    cross_split_texts = sum(
        len(partitions) > 1
        for partitions in text_partitions.values()
    )

    return (
        project_partition,
        partition_projects,
        cross_split_texts,
        text_columns_used,
    )


def document_status(partitions, has_direct, ambiguous):
    """
    Conservative screening statuses.

    These are provisional source-risk indicators,
    not confirmed leakage findings.
    """
    if "Test" in partitions:
        return "Test-project overlap candidate"

    if "Validation" in partitions:
        return "Validation-project overlap candidate"

    if ambiguous:
        return "Ambiguous project mapping"

    if "Train" in partitions:
        return "Train-project overlap candidate"

    if has_direct:
        return "Mapping unresolved"

    return "No direct evidence - provenance unresolved"


def main():
    (
        project_partitions,
        partition_projects,
        cross_split_texts,
        text_columns_used,
    ) = load_splits()

    mappings = read_csv(MAPPING)
    documents = read_csv(DOCUMENTS)

    by_document = defaultdict(list)
    project_rows = []

    for row in mappings:
        document = row["DocumentName"]
        pid = project_id(row["PROMISEProjectID"])
        partition = project_partitions.get(
            pid, "Unknown"
        )

        direct = int(row["UniqueDirectCandidates"])
        fuzzy = int(row["HighFuzzyCandidates"])

        output = {
            "DocumentName": document,
            "PROMISEProjectID": pid,
            "Partition": partition,
            "UniqueDirectCandidates": direct,
            "UniqueDirectPROMISERecords": int(
                row["UniqueDirectPROMISERecords"]
            ),
            "HighFuzzyCandidates": fuzzy,
            "EvidenceType": (
                "Direct and fuzzy"
                if direct and fuzzy
                else "Direct"
                if direct
                else "High fuzzy only"
            ),
            "VerificationStatus": "Pending",
        }

        project_rows.append(output)
        by_document[document].append(output)

    document_rows = []

    for row in documents:
        document = row["DocumentName"]
        candidates = by_document.get(document, [])

        direct = [
            item for item in candidates
            if item["UniqueDirectCandidates"] > 0
        ]

        direct_projects = {
            item["PROMISEProjectID"]
            for item in direct
        }

        partitions = {
            item["Partition"]
            for item in direct
            if item["Partition"] != "Unknown"
        }

        ambiguous = len(direct_projects) > 1

        status = document_status(
            partitions,
            bool(direct),
            ambiguous,
        )

        # Conservative rule:
        # Do not approve any document for Phase 8
        # based solely on automatic matching.
        document_rows.append({
            "DocumentName": document,
            "DirectProjectCount": len(direct_projects),
            "DirectProjectIDs": ";".join(
                sorted(direct_projects)
            ),
            "LinkedPartitions": ";".join(
                sorted(partitions)
            ),
            "HighFuzzyProjectCount": sum(
                item["HighFuzzyCandidates"] > 0
                for item in candidates
            ),
            "AmbiguousMapping": (
                "Yes" if ambiguous else "No"
            ),
            "RiskStatus": status,
            "SafeForNewTraining": "Not established",
            "ManualVerification": "Pending",
            "ReviewerNotes": "",
        })

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    project_fields = [
        "DocumentName",
        "PROMISEProjectID",
        "Partition",
        "UniqueDirectCandidates",
        "UniqueDirectPROMISERecords",
        "HighFuzzyCandidates",
        "EvidenceType",
        "VerificationStatus",
    ]

    document_fields = [
        "DocumentName",
        "DirectProjectCount",
        "DirectProjectIDs",
        "LinkedPartitions",
        "HighFuzzyProjectCount",
        "AmbiguousMapping",
        "RiskStatus",
        "SafeForNewTraining",
        "ManualVerification",
        "ReviewerNotes",
    ]

    write_csv(
        PROJECT_OUT, project_rows, project_fields
    )

    write_csv(
        DOCUMENT_OUT, document_rows, document_fields
    )

    risk_counts = Counter(
        row["RiskStatus"]
        for row in document_rows
    )

    summary = {
        "experiment": "08E",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "split_record_counts": EXPECTED_COUNTS,
        "split_project_counts": {
            name: len(projects)
            for name, projects
            in partition_projects.items()
        },
        "split_text_columns_used": text_columns_used,
        "cross_split_text_keys_detected": cross_split_texts,
        "pure_documents_checked": len(document_rows),
        "candidate_project_links": len(project_rows),
        "risk_status_counts": dict(risk_counts),
        "ambiguous_source_documents": sum(
            row["AmbiguousMapping"] == "Yes"
            for row in document_rows
        ),
        "documents_approved_for_training": 0,
        "manual_source_verification_completed": False,
        "phase4_cv_fold_audit_completed": False,
        "original_datasets_modified": False,
    }

    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 08E")
    print("=" * 60)

    print("Split record counts:")
    for name, count in EXPECTED_COUNTS.items():
        print(
            f"  {name}: {count} records, "
            f"{len(partition_projects[name])} projects"
        )

    print(
        "Cross-split text keys detected:",
        cross_split_texts,
    )

    print("\nPURE documents:", len(document_rows))
    print("Candidate project links:", len(project_rows))

    print("\nRisk classifications:")
    for name, count in sorted(
        risk_counts.items()
    ):
        print(f"  {name}: {count}")

    print(
        "\nAmbiguous source documents:",
        summary["ambiguous_source_documents"],
    )

    print("Approved for training: 0")
    print("Phase 4/5 CV-fold audit: Not yet performed")
    print("Original datasets modified: No")
    print("\nSaved:", PROJECT_OUT)
    print("Saved:", DOCUMENT_OUT)
    print("Saved:", SUMMARY_OUT)


if __name__ == "__main__":
    main()
