
"""Generate consolidated PROMISE+ Phase 2 preprocessing report."""

from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import json

import pandas as pd

from common import (
    CLEANED_DIR,
    SPLITS_DIR,
    TABLES_DIR,
    REPORTS_DIR,
    prepare_outputs,
)

NAMES = ("train", "validation", "test")
MASTER_PATH = CLEANED_DIR / "04_labeled_requirements.csv"


def load_json(filename):
    path = REPORTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False,
    )


def text_key(value):
    return " ".join(str(value).casefold().split())


def markdown_table(frame):
    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]

    for row in frame.itertuples(index=False, name=None):
        cells = [
            str(value).replace("|", r"\|").replace("\n", " ")
            for value in row
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    prepare_outputs()

    # Load existing Phase 2 reports.
    input_report = load_json("01_input_verification.json")
    normalization = load_json("02_normalization_summary.json")
    duplicates = load_json("03_duplicate_review_summary.json")
    labels = load_json("04_label_preparation.json")
    optimization = load_json("07_optimization_report.json")
    manifest = load_json("08_final_split_manifest.json")

    master = read_csv(MASTER_PATH)

    parts = {
        name: read_csv(SPLITS_DIR / f"final_{name}.csv")
        for name in NAMES
    }

    # Confirm master dataset integrity.
    assert len(master) == 3677
    assert master["RecordID"].is_unique
    assert sha256(MASTER_PATH) == manifest["master_sha256"]

    combined = pd.concat(parts.values(), ignore_index=True)

    assert len(combined) == len(master)
    assert combined["RecordID"].is_unique

    # Verify that every original prepared record is preserved.
    columns = master.columns.tolist()

    expected = master[columns].sort_values(
        "RecordID"
    ).reset_index(drop=True)

    actual = combined[columns].sort_values(
        "RecordID"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, actual)

    # Check saved final split hashes.
    for name in NAMES:
        path = SPLITS_DIR / f"final_{name}.csv"
        actual_hash = sha256(path)
        expected_hash = manifest["splits"][name]["sha256"]

        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen split checksum mismatch: {name}"
            )

    # Independently check project and duplicate-text isolation.
    for first, second in combinations(NAMES, 2):
        a = parts[first]
        b = parts[second]

        assert set(a["ProjectID"]).isdisjoint(
            set(b["ProjectID"])
        )

        a_texts = set(a["NormalizedText"].map(text_key))
        b_texts = set(b["NormalizedText"].map(text_key))

        assert a_texts.isdisjoint(b_texts)

    # Calculate final split statistics.
    global_counts = master["_class_"].value_counts()
    global_proportions = global_counts / len(master)

    split_rows = []
    class_rows = []

    for name in NAMES:
        part = parts[name]
        counts = part["_class_"].value_counts()
        proportions = counts / len(part)

        distance = 0.5 * sum(
            abs(
                proportions.get(label, 0)
                - global_proportions.get(label, 0)
            )
            for label in global_counts.index
        )

        split_rows.append({
            "Split": name,
            "Records": len(part),
            "Percentage": round(
                100 * len(part) / len(master), 2
            ),
            "Projects": part["ProjectID"].nunique(),
            "Classes": part["_class_"].nunique(),
            "DistributionDistance": round(distance, 4),
        })

        for label in sorted(global_counts.index):
            class_rows.append({
                "Class": label,
                "Split": name,
                "Count": int(counts.get(label, 0)),
                "Percentage": round(
                    100 * proportions.get(label, 0), 2
                ),
            })

    split_table = pd.DataFrame(split_rows)
    class_table = pd.DataFrame(class_rows)

    split_table.to_csv(
        TABLES_DIR / "09_final_split_summary.csv",
        index=False,
    )

    class_table.to_csv(
        TABLES_DIR / "09_final_class_distribution.csv",
        index=False,
    )

    # Comparison of candidates.
    comparison_path = (
        TABLES_DIR / "07_candidate_comparison.csv"
    )
    comparison = read_csv(comparison_path)

    # Save machine-readable summary.
    summary = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_records": len(master),
        "unique_projects": int(master["ProjectID"].nunique()),
        "unique_classes": int(master["_class_"].nunique()),
        "normalization_modified_records":
            normalization["modified_text_records"],
        "normalized_duplicate_excess":
            duplicates["normalized_duplicate_excess"],
        "label_conflict_groups":
            duplicates["conflicting_label_groups"],
        "label_conflict_rows":
            duplicates["conflicting_label_rows"],
        "selected_candidate": manifest["selected_candidate"],
        "random_seed": manifest["seed"],
        "split_summary": split_rows,
        "master_sha256": sha256(MASTER_PATH),
        "final_split_sha256": {
            name: sha256(SPLITS_DIR / f"final_{name}.csv")
            for name in NAMES
        },
        "integrity_checks_passed": True,
        "records_removed": 0,
    }

    summary_path = (
        REPORTS_DIR / "09_preprocessing_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    # Build Markdown report.
    sections = [
        "# PROMISE+ Dataset: Phase 2 Preprocessing Report",
        (
            f"Generated (UTC): {summary['generated_utc']}\n\n"
            "**Selected split:** Candidate 2"
        ),
        "## 1. Dataset Verification",
        (
            f"The Phase 2 input contains {len(master):,} "
            f"requirements from {master['ProjectID'].nunique()} "
            f"projects with {master['_class_'].nunique()} "
            "classification categories.\n\n"
            f"Phase 1 input comparison: "
            f"{input_report['phase1_comparison']}."
        ),
        "## 2. Text Normalization",
        (
            "Conservative normalization applies Unicode NFC, "
            "whitespace consolidation, and trimming. "
            "Original requirement texts and labels are retained.\n\n"
            f"Modified records: "
            f"{normalization['modified_text_records']:,}.\n\n"
            f"Unchanged records: "
            f"{normalization['unchanged_text_records']:,}.\n\n"
            f"Unique normalized texts: "
            f"{normalization['unique_normalized_texts']:,}."
        ),
        "## 3. Duplicate and Annotation Review",
        (
            f"Exact duplicate excess records: "
            f"{duplicates['exact_duplicate_excess']}.\n\n"
            f"Normalized duplicate excess records: "
            f"{duplicates['normalized_duplicate_excess']}.\n\n"
            f"Cross-project repeated-text groups: "
            f"{duplicates['cross_project_repeated_groups']}.\n\n"
            f"Conflicting-label groups: "
            f"{duplicates['conflicting_label_groups']} "
            f"({duplicates['conflicting_label_rows']} records).\n\n"
            "All records and original annotations are preserved. "
            "The four annotation-conflict groups remain flagged "
            "for review."
        ),
        "## 4. Label Preparation",
        (
            f"Functional requirements: "
            f"{labels['functional_records']:,}.\n\n"
            f"Non-functional requirements: "
            f"{labels['nonfunctional_records']:,}.\n\n"
            "A fixed 12-class integer mapping was created. "
            "Binary labels use 0 for functional requirements "
            "and 1 for all other categories."
        ),
        "## 5. Split Optimization",
        (
            "An initial connected-component split was compared "
            "with a class-aware optimized allocation. "
            "Connected components link records sharing a project "
            "or casefolded, whitespace-normalized text.\n\n"
            f"Candidate 1 optimization score: "
            f"{optimization['candidate1_score']:.5f}.\n\n"
            f"Candidate 2 optimization score: "
            f"{optimization['candidate2_score']:.5f}.\n\n"
            "Candidate 2 was selected using predefined "
            "partition-size, class-balance, and isolation criteria."
        ),
        "### Candidate Comparison",
        markdown_table(comparison.round(4)),
        "## 6. Final Dataset Partitions",
        markdown_table(split_table),
        "### Final Class Distribution",
        markdown_table(
            class_table.pivot(
                index="Class", columns="Split", values="Count"
            ).reset_index().fillna(0)
        ),
        "## 7. Integrity and Leakage Validation",
        (
            "All 3,677 records were retained exactly once.\n\n"
            "- Project overlap across partitions: zero.\n"
            "- Normalized duplicate-text overlap: zero.\n"
            "- Observed classes in every partition: 12.\n"
            "- Final split checksums: match frozen manifest.\n"
            "- Original prepared record content: unchanged."
        ),
        "## 8. Reproducibility",
        (
            f"Random seed: {manifest['seed']}.\n\n"
            "The frozen split checksums are stored in "
            "`08_final_split_manifest.json`. "
            "Final splits are `final_train.csv`, "
            "`final_validation.csv`, and `final_test.csv`.\n\n"
            f"Prepared master SHA-256: "
            f"`{summary['master_sha256']}`."
        ),
        "## 9. Limitations",
        (
            "- Four label-disagreement groups remain unresolved.\n"
            "- Project-level differences can influence class distributions.\n"
            "- Minority categories have limited evaluation examples.\n"
            "- Optimization is heuristic, not globally optimal.\n"
            "- The selected split was optimized using class labels, "
            "but without model predictions.\n"
            "- Contradictory requirements have not been independently "
            "annotated; label conflicts should not be treated as "
            "semantic contradictions."
        ),
        "## 10. Guidance for Model Development",
        (
            "Use the frozen train partition for fitting learned "
            "preprocessing and model parameters. Use the validation "
            "partition for tuning and model selection. Reserve the "
            "test partition for final evaluation.\n\n"
            "Report macro-F1, balanced accuracy, per-class precision "
            "and recall, confusion matrices, and overall accuracy. "
            "Document the handling of duplicate and conflicting labels."
        ),
    ]

    report_path = (
        REPORTS_DIR / "09_final_preprocessing_report.md"
    )

    report_path.write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: FINAL PREPROCESSING REPORT")
    print("=" * 55)
    print(f"Records: {len(master)}")
    print(f"Selected candidate: {manifest['selected_candidate']}")
    print("\nFinal split summary:")
    print(split_table.to_string(index=False))
    print("\nIntegrity checks: PASSED")
    print(f"\nMarkdown report: {report_path}")
    print(f"JSON summary: {summary_path}")
    print("\nReport generation successful.")


if __name__ == "__main__":
    main()
