
"""
Phase 7 Experiment 08G
Prepare conservative source-aware training eligibility policy.

This script does NOT approve PURE documents for training.
It prepares a screening table for future experiments.
"""

from pathlib import Path
from collections import Counter
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs/tables/08f_document_fold_risk.csv"
OUT = ROOT / "outputs/tables/08g_source_aware_fold_policy.csv"
REPORT = ROOT / "outputs/reports/08g_policy_summary.json"

df = pd.read_csv(INPUT, dtype=str).fillna("")

if len(df) != 79 or df["DocumentName"].nunique() != 79:
    raise ValueError("Expected 79 unique PURE documents")

rows = []

for _, source in df.iterrows():
    document = source["DocumentName"]

    linked_folds = set(
        value.strip()
        for value in source["HeldOutFoldsLinked"].split(";")
        if value.strip()
    )

    test_overlap = (
        source["OriginalTestOverlapCandidate"] == "Yes"
    )

    unresolved = (
        source["RiskStatus"]
        == "No direct match - provenance unresolved"
    )

    if not linked_folds.issubset({"1", "2", "3"}):
        raise ValueError(
            f"Unexpected fold mapping: {document}"
        )

    for fold in ("1", "2", "3"):
        if test_overlap:
            status = "BLOCKED_TEST_SOURCE"
            reason = "Candidate overlap with original test project"

        elif unresolved:
            status = "BLOCKED_UNRESOLVED"
            reason = "Source provenance not established"

        elif fold in linked_folds:
            status = "BLOCKED_HELDOUT_SOURCE"
            reason = "Linked to this fold's held-out projects"

        else:
            status = "REVIEW_REQUIRED"
            reason = (
                "Different held-out fold; independent source, "
                "duplicate, and annotation checks required"
            )

        rows.append({
            "DocumentName": document,
            "TargetHeldOutFold": fold,
            "LinkedSourceFolds": ";".join(
                sorted(linked_folds)
            ),
            "ScreeningStatus": status,
            "Reason": reason,
            "ApprovedForTraining": "No",
            "SourceVerified": "No",
            "AnnotationVerified": "No",
        })

result = pd.DataFrame(rows)

OUT.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(OUT, index=False)

counts = {
    str(k): int(v)
    for k, v in Counter(
        result["ScreeningStatus"]
    ).items()
}

summary = {
    "experiment": "08G",
    "pure_documents": 79,
    "heldout_folds": 3,
    "screening_rows": len(result),
    "screening_status_counts": counts,
    "approved_for_training": 0,
    "policy_is_provisional": True,
    "original_datasets_modified": False,
}

REPORT.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nPHASE 7 EXPERIMENT 08G")
print("=" * 55)
print("Documents:", 79)
print("Held-out folds:", 3)
print("Policy rows:", len(result))

print("\nScreening status:")
for status, count in sorted(counts.items()):
    print(f"{status}: {count}")

print("\nApproved for training: 0")
print("Saved:", OUT)
print("Saved:", REPORT)

