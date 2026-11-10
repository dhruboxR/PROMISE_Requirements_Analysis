
"""Phase 7 Experiment 09B: Record AI-assisted contextual assessment.

Produces a separate AI assessment file. It does not claim to be
independent human review and does not modify original labels.
"""

from pathlib import Path
from collections import Counter
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/tables/09_annotation_assessment.csv"
OUTPUT = ROOT / "outputs/tables/09b_ai_contextual_assessment.csv"
REPORT = ROOT / "outputs/reports/09b_ai_assessment_summary.json"

# Provisional AI assessments based on the supplied source excerpts.
ASSESSMENTS = {
    "AUDIT-005": (
        "F", "Medium",
        "Filename suffix .xml specifies required file behavior or formatting. "
        "Operational classification may also warrant consideration."
    ),
    "AUDIT-009": (
        "F", "High",
        "PINES-031 requires generating reports showing item values."
    ),
    "AUDIT-010": (
        "F", "High",
        "PINES-024 requires generating configurable inactive-patron lists."
    ),
    "AUDIT-012": (
        "SE", "High",
        "Source explicitly requires PGP encryption of XML attachments."
    ),
    "AUDIT-015": (
        "F", "High",
        "Section 2.5.3 requires transmitting deletion messages."
    ),
    "AUDIT-021": (
        "PE", "High",
        "Source sets a four-second status-display update limit."
    ),
    "AUDIT-026": (
        "US", "High",
        "OE-5 requires screen-reader and magnification accessibility."
    ),
    "AUDIT-027": (
        "PE", "High",
        "Source sets an approximately 0.1-second detector readout limit."
    ),
    "AUDIT-028": (
        "F", "High",
        "PINES-012 requires producing patron-behavior statistics."
    ),
    "AUDIT-031": (
        "F", "High",
        "Section 2.3.1 describes a correction-request operation."
    ),
    "AUDIT-034": (
        "F", "Medium",
        "PINES-015 requires an interface displaying branch capacity. "
        "A capacity-related interpretation should also be examined."
    ),
}

FIELDS = [
    "AuditID",
    "RecordID",
    "ProjectID",
    "OriginalLabel",
    "PreviousAISuggestion",
    "ContextualAISuggestion",
    "AIConfidence",
    "AssessmentStatus",
    "EvidenceRationale",
    "SourceDocument",
    "SourceLocation",
    "AssessmentMethod",
    "HumanVerificationStatus",
]

def main():
    if OUTPUT.exists():
        raise FileExistsError(
            "Assessment already exists. Preserve previous results."
        )

    with SOURCE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    ids = [row["AuditID"] for row in rows]

    if len(ids) != 11 or set(ids) != set(ASSESSMENTS):
        raise ValueError("Expected the 11 original priority records")

    results = []

    for row in rows:
        audit_id = row["AuditID"]
        proposed, confidence, rationale = ASSESSMENTS[audit_id]

        results.append({
            "AuditID": audit_id,
            "RecordID": row["RecordID"],
            "ProjectID": row["ProjectID"],
            "OriginalLabel": row["OriginalLabel"],
            "PreviousAISuggestion": row["PreviousAISuggestion"],
            "ContextualAISuggestion": proposed,
            "AIConfidence": confidence,
            "AssessmentStatus": "Potential inconsistency",
            "EvidenceRationale": rationale,
            "SourceDocument": row["SourceDocument"],
            "SourceLocation": row["SourceLocation"],
            "AssessmentMethod": (
                "AI-assisted contextual reassessment; "
                "not independent expert annotation"
            ),
            "HumanVerificationStatus": "Not reviewed",
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    summary = {
        "experiment": "09B",
        "records_assessed": len(results),
        "potential_inconsistencies_flagged": sum(
            r["AssessmentStatus"] == "Potential inconsistency"
            for r in results
        ),
        "proposed_label_counts": dict(
            Counter(r["ContextualAISuggestion"] for r in results)
        ),
        "human_assessments_completed": 0,
        "independent_expert_validation_completed": False,
        "original_dataset_modified": False,
        "confirmed_annotation_errors": 0,
        "note": (
            "All 11 cases were preselected for disagreement. "
            "These findings cannot estimate population error rates."
        ),
    }

    REPORT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nPHASE 7 EXPERIMENT 09B")
    print("=" * 50)
    print("Records assessed:", len(results))
    print("Potential inconsistencies:", len(results))
    print("Confirmed annotation errors: 0")
    print("Human reviews completed: 0")
    print("Original dataset modified: No")
    print("Saved:", OUTPUT)
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()
