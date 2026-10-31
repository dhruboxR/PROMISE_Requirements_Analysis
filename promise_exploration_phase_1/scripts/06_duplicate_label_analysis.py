
"""Analyze duplicate requirements and label inconsistencies in PROMISE+."""

import pandas as pd
from common import load_dataset, TABLES_DIR, REPORTS_DIR, prepare_outputs


def normalize_text(text):
    """Normalize case and whitespace for duplicate comparison."""
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().split())


def main():
    prepare_outputs()

    df = load_dataset().copy()

    # Preserve row positions for tracing records back to the dataset.
    df.insert(0, "RecordID", range(1, len(df) + 1))

    # Normalize requirements without changing the original text.
    df["NormalizedText"] = df["RequirementText"].apply(normalize_text)

    total = len(df)

    # 1. Exact duplicates across all original dataset attributes
    exact_mask = df.duplicated(
        subset=["ProjectID", "RequirementText", "_class_"],
        keep=False
    )
    exact_duplicates = df.loc[exact_mask].copy()

    exact_excess = df.duplicated(
        subset=["ProjectID", "RequirementText", "_class_"]
    ).sum()

    exact_duplicates.to_csv(
        TABLES_DIR / "06_exact_duplicates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 2. Repeated normalized requirement texts
    text_mask = df.duplicated(
        subset=["NormalizedText"],
        keep=False
    )
    repeated_texts = df.loc[text_mask].copy()

    normalized_excess = df.duplicated(
        subset=["NormalizedText"]
    ).sum()

    repeated_texts.to_csv(
        TABLES_DIR / "06_repeated_requirements.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 3. Repeated text within the same project
    within_mask = df.duplicated(
        subset=["ProjectID", "NormalizedText"],
        keep=False
    )
    within_project = df.loc[within_mask].copy()

    within_project.to_csv(
        TABLES_DIR / "06_within_project_duplicates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 4. Same requirement appearing across different projects
    project_counts = df.groupby(
        "NormalizedText"
    )["ProjectID"].nunique()

    cross_project_keys = project_counts[
        project_counts > 1
    ].index

    cross_project = df[
        df["NormalizedText"].isin(cross_project_keys)
    ].copy()

    cross_project.to_csv(
        TABLES_DIR / "06_cross_project_duplicates.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 5. Same normalized text assigned multiple class labels
    label_counts = df.groupby(
        "NormalizedText"
    )["_class_"].nunique()

    conflict_keys = label_counts[
        label_counts > 1
    ].index

    label_conflicts = df[
        df["NormalizedText"].isin(conflict_keys)
    ].copy()

    label_conflicts.to_csv(
        TABLES_DIR / "06_label_conflicts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 6. Summary of conflicting annotations
    conflict_summary = (
        label_conflicts.groupby("NormalizedText")
        .agg(
            occurrences=("RecordID", "count"),
            unique_projects=("ProjectID", "nunique"),
            unique_labels=("_class_", "nunique"),
            labels=("_class_", lambda x: ", ".join(sorted(set(x)))),
            projects=("ProjectID", lambda x: ", ".join(sorted(set(x), key=str))),
            example_text=("RequirementText", "first")
        )
        .reset_index()
        .sort_values(
            ["unique_labels", "occurrences"],
            ascending=False
        )
    )

    conflict_summary.to_csv(
        TABLES_DIR / "06_label_conflict_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 7. Generate statistical report
    summary = {
        "Total records": total,
        "Exact duplicate excess rows": int(exact_excess),
        "Rows involved in exact duplicates": len(exact_duplicates),
        "Normalized duplicate excess rows": int(normalized_excess),
        "Rows involved in repeated texts": len(repeated_texts),
        "Rows repeated within projects": len(within_project),
        "Rows repeated across projects": len(cross_project),
        "Distinct texts repeated across projects": len(cross_project_keys),
        "Distinct texts with conflicting labels": len(conflict_keys),
        "Rows involved in label conflicts": len(label_conflicts),
    }

    print("\nPROMISE+ Duplicate and Label Analysis")
    print("=" * 48)

    for metric, value in summary.items():
        print(f"{metric}: {value}")

    report_path = REPORTS_DIR / "06_duplicate_label_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("PROMISE+ Duplicate and Label Analysis\n")
        f.write("=" * 48 + "\n\n")

        for metric, value in summary.items():
            f.write(f"{metric}: {value}\n")

        f.write(
            "\nNote: Different labels on identical text may "
            "indicate annotation disagreement. They do not "
            "necessarily indicate contradictory requirements.\n"
        )

    print("\nFiles saved:")
    print(f"Tables: {TABLES_DIR}")
    print(f"Report: {report_path}")

    if not conflict_summary.empty:
        print("\nSample label conflicts:")
        print(
            conflict_summary[
                ["occurrences", "unique_labels", "labels", "example_text"]
            ].head(10).to_string(index=False, max_colwidth=75)
        )
    else:
        print("\nNo conflicting labels detected for normalized identical texts.")


if __name__ == "__main__":
    main()
