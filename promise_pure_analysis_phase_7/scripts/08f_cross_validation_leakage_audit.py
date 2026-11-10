
"""
Phase 7 Experiment 08F
Cross-Validation Source Overlap Audit

Uses saved Phase 5 and Phase 6 assignments.
Does not regenerate the cross-validation folds.

Important:
A detected source overlap represents a potential risk
for future PURE-based training, not proof that previous
PROMISE+ models experienced data leakage.
"""

from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

PHASE5_FILE = (
    THESIS
    / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

PHASE6_FILE = (
    THESIS
    / "promise_advanced_improvement_phase_6"
    / "outputs/tables/01_verified_folds.csv"
)

PHASE2_DIR = (
    THESIS
    / "promise_preprocessing_phase_2"
    / "outputs/splits"
)

PROJECT_MAPPING_FILE = (
    ROOT
    / "outputs/tables/08d_project_mapping_candidates.csv"
)

DOCUMENT_INVENTORY_FILE = (
    ROOT
    / "outputs/tables/05_pure_document_statistics.csv"
)

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

PROJECT_OUTPUT = TABLES / "08f_project_fold_mapping.csv"
DOCUMENT_OUTPUT = TABLES / "08f_document_fold_risk.csv"
SUMMARY_OUTPUT = REPORTS / "08f_cv_leakage_summary.json"

EXPECTED_FOLDS = {"1", "2", "3"}


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, columns):
    with path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def require_columns(rows, columns, name):
    if not rows:
        raise ValueError(f"Empty input: {name}")

    missing = set(columns) - set(rows[0])

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


def load_phase2():
    partitions = {
        "Train": PHASE2_DIR / "final_train.csv",
        "Validation": PHASE2_DIR / "final_validation.csv",
        "Test": PHASE2_DIR / "final_test.csv",
    }

    records = {}
    projects = defaultdict(set)
    counts = {}

    for partition, path in partitions.items():
        rows = read_csv(path)
        require_columns(
            rows, ("RecordID", "ProjectID"), path.name
        )

        counts[partition] = len(rows)

        for row in rows:
            rid = row["RecordID"].strip()
            pid = row["ProjectID"].strip()

            if rid in records:
                raise ValueError(
                    f"Duplicate RecordID across "
                    f"Phase 2 partitions: {rid}"
                )

            records[rid] = {
                "ProjectID": pid,
                "Partition": partition,
            }

            projects[pid].add(partition)

    if counts != {
        "Train": 2574,
        "Validation": 553,
        "Test": 550,
    }:
        raise ValueError(
            f"Unexpected Phase 2 counts: {counts}"
        )

    return records, projects, counts


def load_and_verify_folds(phase2_records):
    phase5 = read_csv(PHASE5_FILE)
    phase6 = read_csv(PHASE6_FILE)

    require_columns(
        phase5,
        ("RecordID", "ProjectID", "Fold",
         "ConnectedGroup"),
        "Phase 5 assignments",
    )
    require_columns(
        phase6,
        ("RecordID", "ProjectID", "Phase6Fold"),
        "Phase 6 assignments",
    )

    if len(phase5) != 3127 or len(phase6) != 3127:
        raise ValueError(
            "Expected 3127 assignments in each phase"
        )

    def indexed(rows):
        result = {}

        for row in rows:
            rid = row["RecordID"].strip()

            if rid in result:
                raise ValueError(
                    f"Repeated fold RecordID: {rid}"
                )

            result[rid] = row

        return result

    p5 = indexed(phase5)
    p6 = indexed(phase6)

    if set(p5) != set(p6):
        raise ValueError(
            "Phase 5 and Phase 6 RecordIDs differ"
        )

    expected_development = {
        rid for rid, item in phase2_records.items()
        if item["Partition"] != "Test"
    }

    if set(p5) != expected_development:
        raise ValueError(
            "Fold records do not equal the original "
            "3127 development RecordIDs"
        )

    project_folds = defaultdict(set)
    group_folds = defaultdict(set)
    fold_counts = Counter()
    differences = []

    for rid in sorted(p5, key=int):
        row5 = p5[rid]
        row6 = p6[rid]

        pid5 = row5["ProjectID"].strip()
        pid6 = row6["ProjectID"].strip()

        fold5 = row5["Fold"].strip()
        fold6 = row6["Phase6Fold"].strip()

        if (
            pid5 != pid6
            or fold5 != fold6
            or pid5 != phase2_records[rid]["ProjectID"]
        ):
            differences.append(rid)

        if fold5 not in EXPECTED_FOLDS:
            raise ValueError(
                f"Invalid fold {fold5} for {rid}"
            )

        project_folds[pid5].add(fold5)
        group_folds[
            row5["ConnectedGroup"].strip()
        ].add(fold5)

        fold_counts[fold5] += 1

    if differences:
        raise ValueError(
            "Phase 5/6 or Phase 2 inconsistencies. "
            f"First RecordIDs: {differences[:10]}"
        )

    overlapping_projects = {
        pid: folds
        for pid, folds in project_folds.items()
        if len(folds) > 1
    }

    overlapping_groups = {
        group: folds
        for group, folds in group_folds.items()
        if len(folds) > 1
    }

    if overlapping_projects or overlapping_groups:
        raise ValueError(
            "Development folds violate project or "
            "connected-group separation. "
            f"Projects: {overlapping_projects}; "
            f"Groups: {list(overlapping_groups)[:10]}"
        )

    project_to_fold = {
        pid: next(iter(folds))
        for pid, folds in project_folds.items()
    }

    return project_to_fold, fold_counts, {
        "phase5_phase6_agreement": True,
        "project_group_separation": True,
        "phase5_assignments": len(phase5),
        "phase6_assignments": len(phase6),
    }


def main():
    phase2_records, phase2_projects, split_counts = (
        load_phase2()
    )

    project_folds, fold_counts, checks = (
        load_and_verify_folds(phase2_records)
    )

    mapping = read_csv(PROJECT_MAPPING_FILE)
    inventory = read_csv(DOCUMENT_INVENTORY_FILE)

    require_columns(
        mapping,
        ("DocumentName", "PROMISEProjectID",
         "UniqueDirectCandidates", "HighFuzzyCandidates"),
        "08D mapping",
    )

    require_columns(
        inventory, ("DocumentName",), "PURE inventory"
    )

    project_rows = []
    documents = defaultdict(list)

    for row in mapping:
        name = row["DocumentName"]
        pid = row["PROMISEProjectID"].strip()

        direct = int(row["UniqueDirectCandidates"])
        fuzzy = int(row["HighFuzzyCandidates"])

        partition_names = sorted(
            phase2_projects.get(pid, set())
        )

        if pid in project_folds:
            fold = project_folds[pid]
        elif "Test" in partition_names:
            fold = "Original test"
        else:
            fold = "Unknown"

        result = {
            "DocumentName": name,
            "PROMISEProjectID": pid,
            "Phase2Partitions": ";".join(partition_names),
            "DevelopmentHeldOutFold": fold,
            "DirectCandidateCount": direct,
            "HighFuzzyCandidateCount": fuzzy,
            "ProjectIdentityVerified": "No",
        }

        project_rows.append(result)
        documents[name].append(result)

    document_rows = []

    for item in inventory:
        name = item["DocumentName"]
        relationships = documents.get(name, [])

        direct = [
            relationship
            for relationship in relationships
            if relationship["DirectCandidateCount"] > 0
        ]

        linked_projects = sorted({
            relationship["PROMISEProjectID"]
            for relationship in direct
        })

        heldout_folds = sorted({
            relationship["DevelopmentHeldOutFold"]
            for relationship in direct
            if relationship["DevelopmentHeldOutFold"]
            in EXPECTED_FOLDS
        })

        test_overlap = any(
            "Test" in relationship["Phase2Partitions"].split(";")
            for relationship in direct
        )

        unknown = any(
            relationship["DevelopmentHeldOutFold"]
            == "Unknown"
            for relationship in relationships
        )

        if test_overlap:
            status = "Original test overlap candidate"
        elif len(heldout_folds) > 1:
            status = "Multiple development folds"
        elif len(heldout_folds) == 1:
            status = "Single development fold candidate"
        elif unknown:
            status = "Unresolved project identity"
        else:
            status = "No direct match - provenance unresolved"

        document_rows.append({
            "DocumentName": name,
            "DirectProjectIDs": ";".join(linked_projects),
            "HeldOutFoldsLinked": ";".join(heldout_folds),
            "HeldOutFoldCount": len(heldout_folds),
            "OriginalTestOverlapCandidate": (
                "Yes" if test_overlap else "No"
            ),
            "MultipleDevelopmentFolds": (
                "Yes" if len(heldout_folds) > 1 else "No"
            ),
            "RiskStatus": status,
            "SafeForPhase8Training": "Not established",
            "ManualVerification": "Pending",
        })

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    write_csv(
        PROJECT_OUTPUT,
        project_rows,
        [
            "DocumentName",
            "PROMISEProjectID",
            "Phase2Partitions",
            "DevelopmentHeldOutFold",
            "DirectCandidateCount",
            "HighFuzzyCandidateCount",
            "ProjectIdentityVerified",
        ],
    )

    write_csv(
        DOCUMENT_OUTPUT,
        document_rows,
        [
            "DocumentName",
            "DirectProjectIDs",
            "HeldOutFoldsLinked",
            "HeldOutFoldCount",
            "OriginalTestOverlapCandidate",
            "MultipleDevelopmentFolds",
            "RiskStatus",
            "SafeForPhase8Training",
            "ManualVerification",
        ],
    )

    summary = {
        "experiment": "08F",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "phase2_split_counts": split_counts,
        "development_records": 3127,
        "heldout_fold_sizes": dict(sorted(fold_counts.items())),
        "protocol_checks": checks,
        "pure_documents": len(document_rows),
        "project_relationships": len(project_rows),
        "risk_status_counts": dict(
            Counter(
                item["RiskStatus"]
                for item in document_rows
            )
        ),
        "multiple_fold_source_candidates": sum(
            item["MultipleDevelopmentFolds"] == "Yes"
            for item in document_rows
        ),
        "original_test_overlap_candidates": sum(
            item["OriginalTestOverlapCandidate"] == "Yes"
            for item in document_rows
        ),
        "source_identity_verified": False,
        "future_training_data_approved": 0,
        "existing_model_leakage_proven": False,
        "original_datasets_modified": False,
    }

    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 08F")
    print("=" * 60)
    print("Phase 5 / Phase 6 folds identical: Yes")
    print("Project and connected-group separation: Passed")
    print("Development records:", 3127)

    for fold, count in sorted(fold_counts.items()):
        print(f"Held-out fold {fold}: {count} records")

    print("\nPURE documents:", len(document_rows))

    print("\nSource risk screening:")
    for status, count in sorted(
        summary["risk_status_counts"].items()
    ):
        print(f"  {status}: {count}")

    print(
        "\nDocuments linked to multiple folds:",
        summary["multiple_fold_source_candidates"],
    )

    print(
        "Original test overlap candidates:",
        summary["original_test_overlap_candidates"],
    )

    print("Approved for Phase 8 training: 0")
    print("Existing model leakage established: No")
    print("\nSaved:", PROJECT_OUTPUT)
    print("Saved:", DOCUMENT_OUTPUT)
    print("Saved:", SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
