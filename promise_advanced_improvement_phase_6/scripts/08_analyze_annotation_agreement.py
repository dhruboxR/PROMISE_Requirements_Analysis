
"""Phase 6 Experiment 08B: AI-to-dataset annotation agreement.

Compares 96 provisional AI annotations with the original labels.
Does not modify the dataset or access the original test partition.
"""

from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "outputs/annotation_audit"

AI_FILE = DIR / "08_ai_assisted_annotations.csv"
KEY_FILE = DIR / "08_original_label_key.csv"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

EXPECTED_IDS = [
    f"AUDIT-{i:03d}" for i in range(1, 97)
]


def load_and_validate():
    ai = pd.read_csv(
        AI_FILE,
        dtype=str,
        keep_default_na=False
    )

    key = pd.read_csv(
        KEY_FILE,
        dtype=str,
        keep_default_na=False
    )

    assert len(ai) == len(key) == 96
    assert ai["AuditID"].is_unique
    assert key["AuditID"].is_unique
    assert set(ai["AuditID"]) == set(EXPECTED_IDS)
    assert set(key["AuditID"]) == set(EXPECTED_IDS)

    assert set(ai["SuggestedLabel"]).issubset(
        set(LABELS)
    )

    assert set(key["OriginalLabel"]).issubset(
        set(LABELS)
    )

    merged = ai.merge(
        key,
        on="AuditID",
        validate="one_to_one"
    )

    assert len(merged) == 96
    assert merged["AuditID"].is_unique

    merged["Agreement"] = (
        merged["SuggestedLabel"]
        == merged["OriginalLabel"]
    )

    return merged


def main():
    df = load_and_validate()

    total = len(df)
    agreed = int(df["Agreement"].sum())
    disagreed = total - agreed

    agreement_rate = agreed / total

    kappa = cohen_kappa_score(
        df["OriginalLabel"],
        df["SuggestedLabel"],
        labels=LABELS
    )

    print("\nPHASE 6 EXPERIMENT 08B")
    print("ANNOTATION AGREEMENT ANALYSIS")
    print("=" * 65)

    print(f"\nTotal reviewed: {total}")
    print(f"Agreements: {agreed}")
    print(f"Disagreements: {disagreed}")
    print(f"Exact agreement: {agreement_rate:.2%}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Compare original and AI-assigned class distributions.
    distribution = pd.DataFrame({
        "Class": LABELS,
        "OriginalCount": [
            int((df["OriginalLabel"] == x).sum())
            for x in LABELS
        ],
        "AISuggestedCount": [
            int((df["SuggestedLabel"] == x).sum())
            for x in LABELS
        ],
    })

    print("\nCLASS DISTRIBUTION COMPARISON")
    print(distribution.to_string(index=False))

    # Agreement for each original class.
    class_summary = (
        df.groupby("OriginalLabel")
        .agg(
            Total=("AuditID", "size"),
            Agreements=("Agreement", "sum")
        )
        .reset_index()
    )

    class_summary["Disagreements"] = (
        class_summary["Total"]
        - class_summary["Agreements"]
    )

    class_summary["AgreementRate"] = (
        class_summary["Agreements"]
        / class_summary["Total"]
    )

    print("\nAGREEMENT BY ORIGINAL CLASS")
    print(class_summary.to_string(index=False))

    # Compare targeted projects against control samples.
    group_summary = (
        df.groupby("SamplingGroup")
        .agg(
            Total=("AuditID", "size"),
            Agreements=("Agreement", "sum")
        )
        .reset_index()
    )

    group_summary["Disagreements"] = (
        group_summary["Total"]
        - group_summary["Agreements"]
    )

    group_summary["AgreementRate"] = (
        group_summary["Agreements"]
        / group_summary["Total"]
    )

    print("\nAGREEMENT BY SAMPLING GROUP")
    print(group_summary.to_string(index=False))

    # Project-level results.
    project_summary = (
        df.groupby("ProjectID")
        .agg(
            Total=("AuditID", "size"),
            Agreements=("Agreement", "sum")
        )
        .reset_index()
    )

    project_summary["Disagreements"] = (
        project_summary["Total"]
        - project_summary["Agreements"]
    )

    project_summary["AgreementRate"] = (
        project_summary["Agreements"]
        / project_summary["Total"]
    )

    project_summary = project_summary.sort_values(
        ["Disagreements", "Total"],
        ascending=False
    )

    # Rows for independent expert verification.
    disagreements = df[
        ~df["Agreement"]
    ].copy()

    disagreements["VerificationStatus"] = "Pending"
    disagreements["ExpertDecision"] = ""
    disagreements["ExpertNotes"] = ""

    # Original-vs-AI confusion table.
    confusion = pd.crosstab(
        pd.Categorical(
            df["OriginalLabel"],
            categories=LABELS
        ),
        pd.Categorical(
            df["SuggestedLabel"],
            categories=LABELS
        ),
        dropna=False
    )

    confusion.index.name = "OriginalLabel"
    confusion.columns.name = "AISuggestedLabel"

    # Save all results.
    distribution.to_csv(
        DIR / "08_class_distribution_comparison.csv",
        index=False
    )

    class_summary.to_csv(
        DIR / "08_class_agreement.csv",
        index=False
    )

    group_summary.to_csv(
        DIR / "08_sampling_group_agreement.csv",
        index=False
    )

    project_summary.to_csv(
        DIR / "08_project_agreement.csv",
        index=False
    )

    disagreements.to_csv(
        DIR / "08_disagreements_for_review.csv",
        index=False
    )

    confusion.to_csv(
        DIR / "08_annotation_confusion_matrix.csv"
    )

    summary = {
        "experiment": "08B",
        "total": total,
        "agreements": agreed,
        "disagreements": disagreed,
        "exact_agreement": agreement_rate,
        "cohens_kappa": float(kappa),
        "comparison": "AI-assisted labels vs original dataset",
        "independent_expert_review_completed": False,
        "original_labels_modified": False,
        "original_test_accessed": False,
        "important_limitation": (
            "This is a targeted, nonrepresentative "
            "sample. Disagreement does not establish "
            "an incorrect dataset annotation."
        )
    }

    (
        DIR / "08_agreement_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\nMOST FREQUENT LABEL DISAGREEMENTS")

    pairs = (
        disagreements.groupby(
            ["OriginalLabel", "SuggestedLabel"]
        )
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    print(pairs.head(15).to_string(index=False))

    print("\nSaved agreement tables and report.")
    print("Original dataset unchanged.")
    print("Original test partition not accessed.")
    print("\nEXPERIMENT 08B COMPLETED")


if __name__ == "__main__":
    main()
