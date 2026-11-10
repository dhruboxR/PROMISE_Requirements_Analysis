
"""
Phase 7 Experiment 11B
Resolve four uncertain fuzzy matches using PURE source context.

Preserves previous decisions and records the evidence for
each revision.

These are researcher-supplied source-context interpretations,
not independent expert annotation.

No original PROMISE+ or PURE dataset is modified.
"""

from pathlib import Path
from datetime import datetime, timezone
import shutil
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

DECISIONS = TABLES / "10a_fuzzy_review_decisions.csv"
BACKUP = TABLES / "11b_decisions_before_source_verification.csv"
AUDIT = TABLES / "11b_source_verification_audit.csv"
SUMMARY = REPORTS / "11b_source_verification_summary.json"

REVISIONS = {
    "FUZZY-035": {
        "decision": "Different requirement",
        "confidence": "High",
        "source_id": "4.3.5",
        "evidence": (
            "PURE section 4.3.5 requires notifying the server "
            "when a player picks up a tile. PROMISE+ RecordID "
            "1217 concerns sending join information. "
            "The operations differ."
        ),
    },
    "FUZZY-042": {
        "decision": "Different requirement",
        "confidence": "High",
        "source_id": "4.3.6",
        "evidence": (
            "PURE section 4.3.6 requires notifying the server "
            "when a player discards a tile. PROMISE+ RecordID "
            "1217 concerns sending join information. "
            "The operations differ."
        ),
    },
    "FUZZY-115": {
        "decision": "Different requirement",
        "confidence": "High",
        "source_id": "ISIN Request / Proprietary Message field",
        "evidence": (
            "PURE specifies that the Proprietary Message field "
            "must have no contents. PROMISE+ RecordID 2902 "
            "concerns replicated web-server content."
        ),
    },
    "FUZZY-119": {
        "decision": "Different requirement",
        "confidence": "High",
        "source_id": "P-025",
        "evidence": (
            "PURE P-025 requires Clarus to respond to an "
            "information request within five minutes. "
            "PROMISE+ RecordID 2512 concerns collecting "
            "Clarus weather data."
        ),
    },
}


def main():
    for path in (BACKUP, AUDIT, SUMMARY):
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}"
            )

    df = pd.read_csv(
        DECISIONS, dtype=str, keep_default_na=False
    )

    if len(df) != 120:
        raise ValueError("Expected 120 review records")

    if df["ReviewID"].duplicated().any():
        raise ValueError("Duplicate ReviewID found")

    mask = df["ReviewID"].isin(REVISIONS)

    if set(df.loc[mask, "ReviewID"]) != set(REVISIONS):
        raise ValueError("Required ReviewIDs not found")

    if not (
        df.loc[mask, "ReviewerDecision"]
        == "Uncertain"
    ).all():
        raise ValueError(
            "Expected all four previous decisions "
            "to be Uncertain. No changes made."
        )

    if int(
        (df["ReviewerDecision"] != "").sum()
    ) != 120:
        raise ValueError(
            "Complete the 120-pair review first"
        )

    expected_records = {
        "FUZZY-035": "1217",
        "FUZZY-042": "1217",
        "FUZZY-115": "2902",
        "FUZZY-119": "2512",
    }

    for review_id, record_id in expected_records.items():
        actual = df.loc[
            df["ReviewID"] == review_id,
            "PROMISERecordID",
        ].iloc[0]

        if actual != record_id:
            raise ValueError(
                f"{review_id} has unexpected PROMISERecordID"
            )

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    for review_id, revision in REVISIONS.items():
        row_mask = df["ReviewID"] == review_id
        original = df.loc[row_mask].iloc[0]

        audit_rows.append({
            "ReviewID": review_id,
            "CandidateID": original["CandidateID"],
            "DocumentName": original["DocumentName"],
            "PROMISERecordID": original["PROMISERecordID"],
            "OriginalDecision": original["ReviewerDecision"],
            "OriginalConfidence": original["ReviewerConfidence"],
            "OriginalNotes": original["ReviewerNotes"],
            "RevisedDecision": revision["decision"],
            "RevisedConfidence": revision["confidence"],
            "SourceRequirementID": revision["source_id"],
            "SourceEvidence": revision["evidence"],
            "VerificationMethod": (
                "AI-assisted interpretation of extracted "
                "PURE source context"
            ),
            "IndependentHumanVerification": "No",
            "RevisedUTC": timestamp,
        })

        df.loc[
            row_mask, "ReviewerDecision"
        ] = revision["decision"]

        df.loc[
            row_mask, "ReviewerConfidence"
        ] = revision["confidence"]

        df.loc[
            row_mask, "ReviewerNotes"
        ] = revision["evidence"]

        df.loc[
            row_mask, "ReviewMethod"
        ] = (
            "ChatGPT AI-assisted source-context assessment; "
            "not independent human validation"
        )

    # Keep a copy of every pre-revision decision.
    shutil.copy2(DECISIONS, BACKUP)

    pd.DataFrame(audit_rows).to_csv(
        AUDIT, index=False
    )

    df.to_csv(DECISIONS, index=False)

    counts = df["ReviewerDecision"].value_counts()

    summary = {
        "experiment": "11B",
        "created_utc": timestamp,
        "source_context_cases": 4,
        "revised_cases": sorted(REVISIONS),
        "reclassified_to_different": 4,
        "remaining_uncertain": int(
            counts.get("Uncertain", 0)
        ),
        "final_decision_counts": {
            name: int(counts.get(name, 0))
            for name in [
                "Same requirement",
                "Partial overlap",
                "Different requirement",
                "Uncertain",
            ]
        },
        "independent_expert_validation": False,
        "original_datasets_modified": False,
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 11B")
    print("=" * 55)
    print("Uncertain cases reviewed: 4")
    print("Reclassified as Different requirement: 4")
    print("Remaining uncertain: 0")

    print("\nUpdated assessment counts:")
    for name, count in summary[
        "final_decision_counts"
    ].items():
        print(f"{name}: {count}")

    print("\nOriginal datasets modified: No")
    print("Backup:", BACKUP)
    print("Audit:", AUDIT)
    print("Summary:", SUMMARY)


if __name__ == "__main__":
    main()
