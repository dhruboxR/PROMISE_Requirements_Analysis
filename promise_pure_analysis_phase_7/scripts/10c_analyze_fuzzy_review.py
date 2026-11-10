
"""
Phase 7 Experiment 10C
Consolidated AI-Assisted Fuzzy-Match Assessment Analysis

Inputs:
    outputs/tables/10a_fuzzy_review_decisions.csv
    outputs/tables/08c_manual_review_sample.csv

Outputs:
    outputs/tables/10c_similarity_group_results.csv
    outputs/tables/10c_confidence_distribution.csv
    outputs/tables/10c_flagged_matches.csv
    outputs/reports/10c_fuzzy_review_summary.json
    outputs/reports/10c_fuzzy_review_report.md

Important:
    AI-assisted assessments are not independent ground truth.
    Results describe the stratified review sample only.
    No original dataset or training split is modified.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DECISIONS = (
    ROOT / "outputs/tables/10a_fuzzy_review_decisions.csv"
)
SAMPLE = (
    ROOT / "outputs/tables/08c_manual_review_sample.csv"
)

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

GROUP_CSV = TABLES / "10c_similarity_group_results.csv"
CONFIDENCE_CSV = TABLES / "10c_confidence_distribution.csv"
FLAGGED_CSV = TABLES / "10c_flagged_matches.csv"

SUMMARY_JSON = REPORTS / "10c_fuzzy_review_summary.json"
REPORT_MD = REPORTS / "10c_fuzzy_review_report.md"

GROUP_ORDER = [
    "0.90-1.00",
    "0.80-0.90",
    "0.60-0.80",
    "0.40-0.60",
    "0.00-0.40",
]

EXPECTED_GROUP_COUNTS = {
    "0.90-1.00": 30,
    "0.80-0.90": 30,
    "0.60-0.80": 20,
    "0.40-0.60": 20,
    "0.00-0.40": 20,
}

GROUP_POPULATIONS = {
    "0.90-1.00": 106,
    "0.80-0.90": 151,
    "0.60-0.80": 286,
    "0.40-0.60": 789,
    "0.00-0.40": 5932,
}

DECISION_ORDER = [
    "Same requirement",
    "Partial overlap",
    "Different requirement",
    "Uncertain",
]

CONFIDENCE_ORDER = ["High", "Medium", "Low"]


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return pd.read_csv(
        path, dtype=str, keep_default_na=False
    )


def group_for_similarity(score):
    if score >= 0.90:
        return "0.90-1.00"
    if score >= 0.80:
        return "0.80-0.90"
    if score >= 0.60:
        return "0.60-0.80"
    if score >= 0.40:
        return "0.40-0.60"
    return "0.00-0.40"


def validate(df, sample):
    required = {
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
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing review columns: {sorted(missing)}"
        )

    if len(df) != 120:
        raise ValueError(
            f"Expected 120 review rows, got {len(df)}"
        )

    if df["ReviewID"].duplicated().any():
        raise ValueError("Duplicate ReviewID detected")

    if df["CandidateID"].duplicated().any():
        raise ValueError("Duplicate CandidateID detected")

    expected_ids = {
        f"FUZZY-{i:03d}" for i in range(1, 121)
    }

    if set(df["ReviewID"]) != expected_ids:
        raise ValueError(
            "Missing or unexpected ReviewIDs"
        )

    incomplete = df[
        df["ReviewerDecision"].str.strip() == ""
    ]

    if not incomplete.empty:
        raise ValueError(
            "Review incomplete. Missing decisions: "
            + ", ".join(incomplete["ReviewID"].tolist())
        )

    invalid_decisions = set(
        df["ReviewerDecision"]
    ) - set(DECISION_ORDER)

    if invalid_decisions:
        raise ValueError(
            f"Invalid decisions: {invalid_decisions}"
        )

    invalid_confidence = set(
        df["ReviewerConfidence"]
    ) - set(CONFIDENCE_ORDER)

    if invalid_confidence:
        raise ValueError(
            f"Invalid confidence values: "
            f"{invalid_confidence}"
        )

    if (
        df["ReviewerNotes"].str.strip() == ""
    ).any():
        raise ValueError(
            "All decisions must include a rationale"
        )

    if (
        df["ReviewMethod"].str.strip() == ""
    ).any():
        raise ValueError(
            "Assessment method missing"
        )

    if not df["ReviewMethod"].str.contains(
        "AI-assisted", case=False, regex=False
    ).all():
        raise ValueError(
            "Review methods must identify AI assistance"
        )

    if len(sample) != 120:
        raise ValueError(
            "Original review sample does not have 120 rows"
        )

    if sample["CandidateID"].duplicated().any():
        raise ValueError(
            "Duplicate CandidateID in original sample"
        )

    if set(df["CandidateID"]) != set(
        sample["CandidateID"]
    ):
        raise ValueError(
            "Review candidates differ from original sample"
        )

    comparison_columns = [
        "DocumentName",
        "PROMISERecordID",
        "PROMISEProjectID",
        "Similarity",
        "SimilarityGroup",
    ]

    original = sample.set_index("CandidateID")
    reviewed = df.set_index("CandidateID")

    for col in comparison_columns:
        if col not in original:
            raise ValueError(
                f"Original sample missing {col}"
            )

        a = reviewed[col].sort_index()
        b = original[col].sort_index()

        if col == "Similarity":
            if not (
                (
                    pd.to_numeric(a)
                    - pd.to_numeric(b)
                ).abs() <= 1e-8
            ).all():
                raise ValueError(
                    "Similarity scores differ from sample"
                )
        elif not a.equals(b):
            raise ValueError(
                f"Review metadata differs: {col}"
            )

    scores = pd.to_numeric(
        df["Similarity"], errors="raise"
    )

    if scores.isna().any() or not scores.between(0, 1).all():
        raise ValueError("Invalid similarity scores")

    groups = scores.map(group_for_similarity)

    if not (
        groups.to_numpy()
        == df["SimilarityGroup"].to_numpy()
    ).all():
        raise ValueError(
            "Similarity group does not match score"
        )

    counts = Counter(df["SimilarityGroup"])

    if dict(counts) != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            "Stratified sample counts differ "
            f"from original design: {dict(counts)}"
        )

    batch_numbers = pd.to_numeric(
        df["Batch"], errors="raise"
    )

    if not batch_numbers.between(1, 12).all():
        raise ValueError("Unexpected batch number")

    if not (df.groupby("Batch").size() == 10).all():
        raise ValueError(
            "Expected 10 pairs per review batch"
        )


def make_group_table(df):
    rows = []

    for group in GROUP_ORDER:
        subset = df[
            df["SimilarityGroup"] == group
        ]

        counts = Counter(subset["ReviewerDecision"])
        n = len(subset)

        row = {
            "SimilarityGroup": group,
            "PopulationCandidates": (
                GROUP_POPULATIONS[group]
            ),
            "ReviewedPairs": n,
            "SameRequirement": counts[
                "Same requirement"
            ],
            "PartialOverlap": counts[
                "Partial overlap"
            ],
            "DifferentRequirement": counts[
                "Different requirement"
            ],
            "Uncertain": counts["Uncertain"],
            "SameRatePercent": round(
                100 * counts["Same requirement"] / n, 2
            ),
            "AnyOverlapRatePercent": round(
                100 * (
                    counts["Same requirement"]
                    + counts["Partial overlap"]
                ) / n,
                2,
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def make_confidence_table(df):
    rows = []

    for decision in DECISION_ORDER:
        subset = df[
            df["ReviewerDecision"] == decision
        ]

        counts = Counter(
            subset["ReviewerConfidence"]
        )

        rows.append({
            "Decision": decision,
            "Total": len(subset),
            "High": counts["High"],
            "Medium": counts["Medium"],
            "Low": counts["Low"],
        })

    return pd.DataFrame(rows)


def markdown_table(frame):
    cols = list(frame.columns)

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(
            ["---"] * len(cols)
        ) + " |",
    ]

    for row in frame.itertuples(
        index=False, name=None
    ):
        lines.append(
            "| " + " | ".join(
                str(value).replace("|", "\\|")
                for value in row
            ) + " |"
        )

    return "\n".join(lines)


def main():
    df = read_csv(DECISIONS)
    sample = read_csv(SAMPLE)

    validate(df, sample)

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    group_table = make_group_table(df)
    confidence_table = make_confidence_table(df)

    flagged = df[
        (df["ReviewerDecision"] == "Uncertain")
        | (df["ReviewerConfidence"] == "Low")
    ].copy()

    flagged = flagged.sort_values(
        ["SimilarityGroup", "ReviewID"]
    )

    group_table.to_csv(GROUP_CSV, index=False)
    confidence_table.to_csv(
        CONFIDENCE_CSV, index=False
    )
    flagged.to_csv(FLAGGED_CSV, index=False)

    totals = Counter(df["ReviewerDecision"])

    overall_counts = {
        decision: totals[decision]
        for decision in DECISION_ORDER
    }

    summary = {
        "experiment": "10C",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "review_pairs": len(df),
        "total_fuzzy_candidates_searched": 7264,
        "sample_design": (
            "Stratified random sample of each PURE "
            "candidate's highest-scoring PROMISE+ match"
        ),
        "random_seed": 42,
        "reviewed_by": (
            "ChatGPT AI-assisted textual assessment"
        ),
        "independent_human_reviewers": 0,
        "decision_counts": overall_counts,
        "same_requirement_count": totals[
            "Same requirement"
        ],
        "any_overlap_count": (
            totals["Same requirement"]
            + totals["Partial overlap"]
        ),
        "uncertain_count": totals["Uncertain"],
        "flagged_for_source_check": len(flagged),
        "group_results": group_table.to_dict(
            orient="records"
        ),
        "confirmed_ground_truth": False,
        "verified_precision_available": False,
        "verified_recall_available": False,
        "threshold_validated_independently": False,
        "source_documents_approved_for_training": 0,
        "original_datasets_modified": False,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = [
        "# Phase 7 Experiment 10C",
        "",
        "## AI-Assisted Fuzzy-Match Assessment",
        "",
        "### Method",
        "",
        "Experiment 08B generated TF-IDF character n-gram "
        "similarity candidates for PURE passages not "
        "matched by Experiment 08A.",
        "",
        "Experiment 08C selected 120 unique PURE "
        "candidates using reproducible stratified random "
        "sampling across five similarity-score ranges.",
        "",
        "The highest-scoring PROMISE+ match for each "
        "selected candidate was assessed through "
        "ChatGPT-assisted textual comparison.",
        "",
        "Decisions were recorded as Same requirement, "
        "Partial overlap, Different requirement, "
        "or Uncertain.",
        "",
        "### Overall assessment",
        "",
        markdown_table(
            pd.DataFrame([
                {
                    "Decision": decision,
                    "Count": totals[decision],
                    "Percent of sample": round(
                        100 * totals[decision] / len(df),
                        2,
                    ),
                }
                for decision in DECISION_ORDER
            ])
        ),
        "",
        "### Assessment by similarity range",
        "",
        markdown_table(group_table),
        "",
        "### Reviewer confidence",
        "",
        markdown_table(confidence_table),
        "",
        "### Findings",
        "",
        "- High-similarity pairs frequently express the "
        "same core requirement.",
        "- Partial matches frequently arise from "
        "different requirement boundaries, omitted "
        "sentences, or additional conditions.",
        "- Lower-similarity pairs more often express "
        "different requirements.",
        "- PDF table extraction, inserted section "
        "identifiers, and line-merging artifacts "
        "affect textual matching.",
        "- Similarity scores should prioritize source "
        "verification, not automatically establish "
        "requirement equivalence.",
        "",
        "### Limitations",
        "",
        "- All decisions are AI-assisted assessments, "
        "not independent expert-validated ground truth.",
        "- The sample is stratified by similarity score; "
        "raw overall proportions are not population "
        "prevalence estimates.",
        "- Only the highest-scoring candidate match "
        "per sampled PURE passage was assessed.",
        "- Uncertain cases remain unresolved.",
        "- No validated precision or recall is reported.",
        "- Source-document independence and annotation "
        "quality are not established by this assessment.",
        "- No PURE records are approved for training.",
        "",
        "### Outputs",
        "",
        f"- {GROUP_CSV.name}",
        f"- {CONFIDENCE_CSV.name}",
        f"- {FLAGGED_CSV.name}",
        f"- {SUMMARY_JSON.name}",
        "",
    ]

    REPORT_MD.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 10C")
    print("=" * 60)

    print("Total assessed:", len(df))

    print("\nDecision counts:")
    for decision in DECISION_ORDER:
        print(f"  {decision}: {totals[decision]}")

    print("\nSimilarity-group results:")
    print(
        group_table.to_string(index=False)
    )

    print(
        "\nCases flagged for source inspection:",
        len(flagged),
    )

    print("\nVerified precision/recall: Not available")
    print("Source documents approved for training: 0")
    print("Original datasets modified: No")

    print("\nOutputs:")
    for path in (
        GROUP_CSV,
        CONFIDENCE_CSV,
        FLAGGED_CSV,
        SUMMARY_JSON,
        REPORT_MD,
    ):
        print(path)


if __name__ == "__main__":
    main()
