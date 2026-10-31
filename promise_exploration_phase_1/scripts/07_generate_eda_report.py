
"""Generate consolidated PROMISE+ Phase 1 exploratory analysis report."""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json

import pandas as pd

from common import (
    load_dataset,
    DATA_PATH,
    OUTPUT_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    prepare_outputs,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_DIR.parent / "dataset" / "Promise+.arff"

TEXT_COL = "RequirementText"
LABEL_COL = "_class_"
PROJECT_COL = "ProjectID"


def normalize_text(value):
    """Normalize whitespace and case without modifying original data."""
    return " ".join(str(value).casefold().split())


def file_hash(path):
    """Calculate a reproducible SHA-256 checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame):
    """Convert a DataFrame to a Markdown table."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        values = [
            str(value).replace("|", r"\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    prepare_outputs()

    df = load_dataset().copy()

    required = {TEXT_COL, LABEL_COL, PROJECT_COL}
    missing_columns = required - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    total = len(df)

    if total == 0:
        raise ValueError("Dataset is empty.")

    # --------------------------------------------------
    # 1. Dataset overview
    # --------------------------------------------------

    unique_projects = df[PROJECT_COL].nunique()
    unique_classes = df[LABEL_COL].nunique()
    unique_texts = df[TEXT_COL].nunique()

    missing_values = df[
        [PROJECT_COL, TEXT_COL, LABEL_COL]
    ].isna().sum()

    blank_texts = int(
        df[TEXT_COL].fillna("").astype(str).str.strip().eq("").sum()
    )

    # --------------------------------------------------
    # 2. Class distribution
    # --------------------------------------------------

    class_counts = df[LABEL_COL].value_counts()

    class_table = pd.DataFrame({
        "Class": class_counts.index,
        "Count": class_counts.values,
    })

    class_table["Percentage"] = (
        class_table["Count"] / total * 100
    ).round(2)

    class_table.to_csv(
        TABLES_DIR / "07_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    majority_label = str(class_counts.index[0])
    minority_label = str(class_counts.index[-1])

    imbalance_ratio = (
        int(class_counts.iloc[0]) / int(class_counts.iloc[-1])
    )

    # --------------------------------------------------
    # 3. Text statistics
    # --------------------------------------------------

    text = df[TEXT_COL].fillna("").astype(str)

    character_counts = text.str.len()
    word_counts = text.str.split().str.len()

    stats = pd.DataFrame({
        "Character count": character_counts,
        "Word count": word_counts,
    }).describe().round(2)

    stats_table = stats.reset_index().rename(
        columns={"index": "Statistic"}
    )

    # --------------------------------------------------
    # 4. Duplicate analysis
    # --------------------------------------------------

    normalized = text.map(normalize_text)

    exact_columns = [
        PROJECT_COL, TEXT_COL, LABEL_COL
    ]

    exact_excess = int(
        df.duplicated(subset=exact_columns).sum()
    )

    exact_involved = int(
        df.duplicated(
            subset=exact_columns, keep=False
        ).sum()
    )

    normalized_excess = int(normalized.duplicated().sum())

    normalized_involved = int(
        normalized.duplicated(keep=False).sum()
    )

    analysis_df = df.copy()
    analysis_df["NormalizedText"] = normalized

    project_group = analysis_df.groupby(
        "NormalizedText"
    )[PROJECT_COL].nunique()

    cross_project_keys = project_group[
        project_group > 1
    ].index

    cross_project_rows = int(
        normalized.isin(cross_project_keys).sum()
    )

    label_group = analysis_df.groupby(
        "NormalizedText"
    )[LABEL_COL].nunique()

    conflicting_keys = label_group[
        label_group > 1
    ].index

    conflict_rows = analysis_df[
        analysis_df["NormalizedText"].isin(conflicting_keys)
    ]

    conflict_summary = (
        conflict_rows.groupby("NormalizedText")
        .agg(
            Records=(LABEL_COL, "size"),
            Labels=(LABEL_COL, lambda s: ", ".join(
                sorted(set(map(str, s)))
            )),
            Projects=(PROJECT_COL, lambda s: ", ".join(
                sorted(set(map(str, s)))
            )),
            Example=(TEXT_COL, "first"),
        )
        .reset_index(drop=True)
    )

    conflict_summary.to_csv(
        TABLES_DIR / "07_label_conflicts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------
    # 5. Repair documentation
    # --------------------------------------------------

    audit_path = REPORTS_DIR / "arff_repair_audit.csv"

    audit_rows = None

    if audit_path.exists():
        audit_rows = len(pd.read_csv(audit_path))

    # --------------------------------------------------
    # 6. Machine-readable summary
    # --------------------------------------------------

    summary = {
        "total_records": int(total),
        "total_columns": int(len(df.columns)),
        "unique_projects": int(unique_projects),
        "unique_classes": int(unique_classes),
        "unique_requirement_texts": int(unique_texts),
        "missing_values": {
            str(key): int(value)
            for key, value in missing_values.items()
        },
        "blank_requirement_texts": blank_texts,
        "majority_class": majority_label,
        "minority_class": minority_label,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "mean_word_count": round(float(word_counts.mean()), 2),
        "median_word_count": float(word_counts.median()),
        "max_word_count": int(word_counts.max()),
        "exact_duplicate_excess": exact_excess,
        "exact_duplicate_involved": exact_involved,
        "normalized_duplicate_excess": normalized_excess,
        "normalized_duplicate_involved": normalized_involved,
        "cross_project_duplicate_texts": int(len(cross_project_keys)),
        "cross_project_duplicate_rows": cross_project_rows,
        "conflicting_label_texts": int(len(conflicting_keys)),
        "conflicting_label_rows": int(len(conflict_rows)),
        "comma_repair_audit_rows": audit_rows,
        "validated_dataset_sha256": file_hash(DATA_PATH),
        "original_dataset_sha256": (
            file_hash(RAW_PATH) if RAW_PATH.exists() else None
        ),
    }

    json_path = REPORTS_DIR / "07_eda_summary.json"

    json_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 7. Markdown report
    # --------------------------------------------------

    report = []

    report.append("# PROMISE+ Dataset: Phase 1 Exploratory Analysis")
    report.append(
        f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    report.append("\n## 1. Dataset Description")

    report.append(
        f"""
The validated PROMISE+ dataset contains **{total:,} requirements**
associated with **{unique_projects} unique project identifiers**.

The dataset contains three attributes:

- **ProjectID:** Identifier of the source project.
- **RequirementText:** Natural-language requirement.
- **_class_:** Requirement classification label.

There are **{unique_classes} observed classification labels**.
All analyses use the validated working copy; the raw dataset is preserved.
"""
    )

    report.append("\n## 2. Dataset Validation")

    report.append(
        f"""
The source ARFF file required formatting corrections before parsing.

The working copy was successfully loaded with {total:,} records.

Number of entries in the comma-repair audit:
**{audit_rows if audit_rows is not None else 'Not available'}**.

Legacy punctuation escape sequences were also corrected during
validation. Those inferred substitutions require a complete
line-level repair audit before publication.

Original SHA-256:
`{summary['original_dataset_sha256']}`

Validated SHA-256:
`{summary['validated_dataset_sha256']}`
"""
    )

    report.append("\n## 3. Data Quality")

    quality_table = pd.DataFrame([
        ["Total records", total],
        ["Unique requirement texts", unique_texts],
        ["Missing ProjectID", int(missing_values[PROJECT_COL])],
        ["Missing RequirementText", int(missing_values[TEXT_COL])],
        ["Missing class labels", int(missing_values[LABEL_COL])],
        ["Blank requirement texts", blank_texts],
        ["Exact duplicate excess records", exact_excess],
        ["Records involved in exact duplicates", exact_involved],
        ["Normalized duplicate excess records", normalized_excess],
        ["Records involved in normalized duplicates", normalized_involved],
    ], columns=["Metric", "Value"])

    report.append(markdown_table(quality_table))

    report.append("\n## 4. Class Distribution")
    report.append(markdown_table(class_table))

    report.append(
        f"""
The majority class is **{majority_label}** and the minority
class is **{minority_label}**.

The majority-to-minority frequency ratio is approximately
**{imbalance_ratio:.2f}:1**.

This imbalance makes macro-F1, balanced accuracy, and
per-class performance important evaluation metrics.
"""
    )

    report.append("\n## 5. Requirement Length Statistics")
    report.append(markdown_table(stats_table))

    report.append(
        f"""
The mean requirement length is **{word_counts.mean():.2f} words**,
while the median is **{word_counts.median():.0f} words**.

The longest requirement contains **{word_counts.max()} words**.

Word counts use whitespace tokenization, so values may differ
from those produced by linguistic tokenizers.
"""
    )

    report.append("\n## 6. Duplicate and Label Analysis")

    duplicate_table = pd.DataFrame([
        ["Exact duplicate excess records", exact_excess],
        ["Normalized duplicate excess records", normalized_excess],
        ["Repeated texts across projects", len(cross_project_keys)],
        ["Rows involved across projects", cross_project_rows],
        ["Texts with multiple labels", len(conflicting_keys)],
        ["Rows involved in label conflicts", len(conflict_rows)],
    ], columns=["Metric", "Count"])

    report.append(markdown_table(duplicate_table))

    report.append(
        "\n### Potential annotation disagreements\n"
    )

    if len(conflict_summary):
        report.append(
            markdown_table(
                conflict_summary[
                    ["Labels", "Projects", "Example"]
                ]
            )
        )
    else:
        report.append(
            "No label disagreements were found for normalized identical texts."
        )

    report.append(
        """
Identical requirements assigned different class labels may
indicate annotation disagreements or category ambiguity.
They do not directly establish contradictions between
software requirements.
"""
    )

    report.append("\n## 7. Methodological Limitations")

    report.append(
        """
- Some ARFF formatting corrections were necessary to load the dataset.
- Legacy punctuation substitutions were inferred and should be audited.
- Duplicate records may affect model evaluation if they cross data splits.
- Identical requirements with different labels require manual review.
- Class imbalance may reduce minority-class recognition.
- Requirement length varies considerably across records.
- The dataset does not directly provide labels for semantic contradiction detection.
- Project-wise and text-group-wise split strategies should be evaluated before training.
"""
    )

    report.append("\n## 8. Recommendations for Phase 2")

    report.append(
        """
1. Complete the line-level audit for all dataset repairs.
2. Manually review the identified conflicting-label records.
3. Define a consistent duplicate-handling policy.
4. Establish leakage-controlled training and testing splits.
5. Preserve all original requirement texts.
6. Perform controlled preprocessing experiments.
7. Evaluate models using macro-F1 and class-wise metrics.
8. Keep requirement contradiction detection separate from label disagreement analysis.
"""
    )

    report_path = REPORTS_DIR / "07_final_eda_report.md"

    report_path.write_text(
        "\n\n".join(report),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 8. Terminal summary
    # --------------------------------------------------

    print("\nPROMISE+ PHASE 1 EDA REPORT")
    print("=" * 50)

    for key, value in summary.items():
        if "sha256" not in key:
            print(f"{key}: {value}")

    print("\nReport generated successfully.")
    print(f"Markdown: {report_path}")
    print(f"JSON: {json_path}")
    print(f"Tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
