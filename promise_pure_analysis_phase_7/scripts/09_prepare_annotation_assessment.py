
"""
Phase 7 Experiment 09
AI-Assisted Source-Context Annotation Assessment

Consolidates existing evidence from:
- Experiment 03: Source context extraction
- Experiment 04B: Clean requirement statements

Does NOT:
- Invent independent human reviewer decisions
- Automatically correct PROMISE+ labels
- Reclassify requirements using a model
- Modify the original datasets

The previous AI suggestion is kept for comparison.
Researcher assessment fields are initially blank.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import json

ROOT = Path(__file__).resolve().parents[1]

CONTEXT_FILE = (
    ROOT / "outputs/tables/03_source_context.csv"
)

REQUIREMENTS_FILE = (
    ROOT
    / "outputs/annotation_review"
    / "04b_reviewer_1_blind.csv"
)

OUTPUT_CSV = (
    ROOT / "outputs/tables/09_annotation_assessment.csv"
)

OUTPUT_JSON = (
    ROOT
    / "outputs/reports"
    / "09_annotation_assessment_summary.json"
)

VALID_LABELS = {
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
}

VALID_STATUSES = {
    "Original label supported",
    "Potential inconsistency",
    "Unresolved",
}

FIELDS = [
    "AuditID",
    "RecordID",
    "ProjectID",
    "RequirementText",
    "OriginalLabel",
    "PreviousAISuggestion",
    "SourceDocument",
    "SourceLocation",
    "SourceRequirementID",
    "SourceSection",
    "SourceContext",
    "ResearcherProposedLabel",
    "ResearcherAssessment",
    "ResearcherConfidence",
    "ContextEvidence",
    "ReasonForDecision",
    "NeedsAdditionalVerification",
    "AssessmentMethod",
]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def index_unique(rows, key):
    indexed = {}

    for row in rows:
        identifier = row[key].strip()

        if not identifier:
            raise ValueError(f"Empty {key}")

        if identifier in indexed:
            raise ValueError(
                f"Duplicate {key}: {identifier}"
            )

        indexed[identifier] = row

    return indexed


def main():
    context_rows = read_csv(CONTEXT_FILE)
    requirement_rows = read_csv(REQUIREMENTS_FILE)

    context = index_unique(context_rows, "AuditID")
    requirements = index_unique(
        requirement_rows, "AuditID"
    )

    if (
        len(context) != 11
        or len(requirements) != 11
        or set(context) != set(requirements)
    ):
        raise ValueError(
            "Expected the same 11 unique AuditIDs "
            "in both source files"
        )

    results = []

    for audit_id in sorted(context):
        source = context[audit_id]
        clean_requirement = requirements[audit_id]

        original_label = source["OriginalLabel"].strip()
        ai_label = source["AISuggestedLabel"].strip()

        if (
            original_label not in VALID_LABELS
            or ai_label not in VALID_LABELS
        ):
            raise ValueError(
                f"Invalid category for {audit_id}"
            )

        requirement_text = clean_requirement[
            "RequirementText"
        ].strip()

        if not requirement_text:
            raise ValueError(
                f"Missing requirement text: {audit_id}"
            )

        results.append({
            "AuditID": audit_id,
            "RecordID": source["RecordID"],
            "ProjectID": source["ProjectID"],
            "RequirementText": requirement_text,
            "OriginalLabel": original_label,
            "PreviousAISuggestion": ai_label,
            "SourceDocument": source["SourceDocument"],
            "SourceLocation": source["SourceLocation"],
            "SourceRequirementID": source[
                "SourceRequirementID"
            ],
            "SourceSection": source["SourceSection"],
            "SourceContext": source[
                "SurroundingContext"
            ],
            "ResearcherProposedLabel": "",
            "ResearcherAssessment": "",
            "ResearcherConfidence": "",
            "ContextEvidence": "",
            "ReasonForDecision": "",
            "NeedsAdditionalVerification": "",
            "AssessmentMethod": (
                "AI-assisted evidence preparation; "
                "researcher assessment pending"
            ),
        })

    OUTPUT_CSV.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_JSON.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Protect completed manual work on reruns.
    if OUTPUT_CSV.exists():
        raise FileExistsError(
            f"{OUTPUT_CSV} already exists. "
            "Review or back up the file before rerunning."
        )

    with OUTPUT_CSV.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(results)

    original_counts = Counter(
        row["OriginalLabel"]
        for row in results
    )

    ai_counts = Counter(
        row["PreviousAISuggestion"]
        for row in results
    )

    agreement = sum(
        row["OriginalLabel"]
        == row["PreviousAISuggestion"]
        for row in results
    )

    summary = {
        "experiment": "09",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "title": (
            "AI-Assisted Source-Context "
            "Annotation Assessment"
        ),
        "requirements": len(results),
        "source_documents": len({
            row["SourceDocument"]
            for row in results
        }),
        "original_label_distribution": dict(
            sorted(original_counts.items())
        ),
        "previous_ai_distribution": dict(
            sorted(ai_counts.items())
        ),
        "original_ai_agreements": agreement,
        "original_ai_disagreements": (
            len(results) - agreement
        ),
        "researcher_assessments_completed": 0,
        "independent_expert_reviews_completed": 0,
        "valid_assessment_statuses": sorted(
            VALID_STATUSES
        ),
        "source_traceability_established": True,
        "contextual_label_verification_completed": False,
        "label_corrections_approved": 0,
        "original_datasets_modified": False,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8"
    )

    print("\nPHASE 7 EXPERIMENT 09")
    print("=" * 55)
    print("Requirements:", len(results))
    print(
        "Distinct source files:",
        summary["source_documents"],
    )
    print("Original/previous AI agreements:", agreement)
    print(
        "Original/previous AI disagreements:",
        len(results) - agreement,
    )
    print("Researcher assessments completed: 0")
    print("Independent expert reviews completed: 0")
    print("Original labels modified: No")
    print("\nCSV:", OUTPUT_CSV)
    print("Report:", OUTPUT_JSON)


if __name__ == "__main__":
    main()

