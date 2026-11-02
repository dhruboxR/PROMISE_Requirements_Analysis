
"""PROMISE+ Phase 3: Consolidated baseline model comparison."""

import json
import pandas as pd
import matplotlib.pyplot as plt

from common import (
    LABELS,
    METRICS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    prepare_outputs,
)


MODELS = {
    "Logistic Regression": {
        "metrics": "01_baseline_metrics.csv",
        "report": "01_logistic_regression_classification_report.json",
        "filter": "TF-IDF + Logistic Regression",
    },
    "Linear SVM": {
        "metrics": "02_svm_metrics.csv",
        "report": "02_svm_classification_report.json",
        "filter": None,
    },
    "Naive Bayes": {
        "metrics": "03_naive_bayes_metrics.csv",
        "report": "03_naive_bayes_classification_report.json",
        "filter": None,
    },
    "Random Forest": {
        "metrics": "04_random_forest_metrics.csv",
        "report": "04_random_forest_classification_report.json",
        "filter": None,
    },
}


def main():
    prepare_outputs()

    print("\nPROMISE+ PHASE 3: BASELINE MODEL COMPARISON")
    print("=" * 60)

    # Combine the original five baseline results.
    metric_files = [
        "01_baseline_metrics.csv",
        "02_svm_metrics.csv",
        "03_naive_bayes_metrics.csv",
        "04_random_forest_metrics.csv",
    ]

    frames = []

    for filename in metric_files:
        path = METRICS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(path)

        frames.append(pd.read_csv(path))

    comparison = pd.concat(frames, ignore_index=True)

    if comparison["Model"].duplicated().any():
        raise ValueError("Duplicate model entries in metrics.")

    if set(comparison["Partition"]) != {"validation"}:
        raise ValueError("Unexpected evaluation partition.")

    comparison = comparison.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    comparison.insert(
        0, "Rank", range(1, len(comparison) + 1)
    )

    comparison.to_csv(
        TABLES_DIR / "05_baseline_comparison.csv",
        index=False,
    )

    print("\nOVERALL VALIDATION PERFORMANCE")
    print(
        comparison[
            [
                "Rank", "Model", "accuracy",
                "balanced_accuracy", "macro_f1",
                "weighted_f1",
            ]
        ].to_string(index=False)
    )

    # Load the detailed per-class classification reports.
    class_rows = []

    for model_name, config in MODELS.items():
        path = REPORTS_DIR / config["report"]

        if not path.exists():
            raise FileNotFoundError(path)

        report = json.loads(path.read_text(encoding="utf-8"))

        for label in LABELS:
            if label not in report:
                raise ValueError(
                    f"Missing class {label} in {model_name}"
                )

            scores = report[label]

            class_rows.append({
                "Model": model_name,
                "Class": label,
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1": scores["f1-score"],
                "Support": scores["support"],
            })

    class_table = pd.DataFrame(class_rows)

    class_table.to_csv(
        TABLES_DIR / "05_per_class_performance.csv",
        index=False,
    )

    # Compare the two strongest current baseline models.
    linear_models = class_table[
        class_table["Model"].isin(
            ["Linear SVM", "Logistic Regression"]
        )
    ]

    per_class_comparison = linear_models.pivot(
        index="Class",
        columns="Model",
        values="F1",
    ).reindex(LABELS)

    per_class_comparison.to_csv(
        TABLES_DIR / "05_linear_models_per_class_f1.csv"
    )

    print("\nLINEAR MODEL PER-CLASS F1")
    print(per_class_comparison.round(4).to_string())

    # Plot overall macro-F1.
    plt.figure(figsize=(10, 5))

    plt.barh(
        comparison["Model"],
        comparison["macro_f1"],
    )

    plt.xlabel("Macro-F1")
    plt.title("PROMISE+ Baseline Model Comparison")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "05_baseline_macro_f1.png",
        dpi=300,
    )
    plt.close()

    # Plot all 12 classes for Linear SVM.
    svm_classes = class_table[
        class_table["Model"] == "Linear SVM"
    ].set_index("Class").reindex(LABELS)

    plt.figure(figsize=(11, 5))

    plt.bar(
        svm_classes.index,
        svm_classes["F1"],
    )

    plt.ylabel("F1-score")
    plt.xlabel("Requirement class")
    plt.title("Linear SVM: Per-Class F1")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "05_svm_per_class_f1.png",
        dpi=300,
    )
    plt.close()

    # Identify classes needing further investigation.
    weak_classes = svm_classes[
        svm_classes["F1"] < 0.30
    ].sort_values("F1")

    print("\nSVM CLASSES WITH F1 BELOW 0.30")

    if weak_classes.empty:
        print("None")
    else:
        print(
            weak_classes[
                ["Precision", "Recall", "F1", "Support"]
            ].round(4).to_string()
        )

    best = comparison.iloc[0]

    summary = {
        "experiment": "Phase 3 baseline comparison",
        "evaluation_partition": "validation",
        "selection_metric": "macro_f1",
        "best_baseline": best["Model"],
        "best_macro_f1": float(best["macro_f1"]),
        "best_accuracy": float(best["accuracy"]),
        "models_compared": len(comparison),
        "test_partition_used": False,
        "weak_svm_classes": weak_classes.index.tolist(),
    }

    (
        REPORTS_DIR / "05_baseline_summary.json"
    ).write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    # Human-readable report.
    report_lines = [
        "# PROMISE+ Phase 3 Baseline Comparison",
        "",
        "## Experimental Setup",
        "",
        "- Task: 12-class requirements classification",
        "- Training records: 2574",
        "- Validation records: 553",
        "- Test records: reserved for final evaluation",
        "- Primary selection metric: macro-F1",
        "",
        "## Model Comparison",
        "",
        comparison.to_string(index=False),
        "",
        "## Best Baseline",
        "",
        f"- Model: {best['Model']}",
        f"- Macro-F1: {best['macro_f1']:.4f}",
        f"- Accuracy: {best['accuracy']:.4f}",
        "",
        "## Linear SVM Per-Class F1",
        "",
        per_class_comparison.round(4).to_string(),
        "",
        "## Classes Requiring Review",
        "",
        (
            weak_classes[
                ["Precision", "Recall", "F1", "Support"]
            ].round(4).to_string()
            if not weak_classes.empty
            else "No classes below the selected threshold."
        ),
        "",
        "## Evaluation Protocol",
        "",
        "All comparisons use the frozen validation partition.",
        "The test partition was not accessed.",
        "Results are preliminary until final model selection.",
    ]

    (
        REPORTS_DIR / "05_baseline_comparison_report.md"
    ).write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\nBest baseline:", best["Model"])
    print("Best macro-F1:", f"{best['macro_f1']:.4f}")

    print("\nGenerated:")
    print("- Consolidated comparison table")
    print("- Per-class performance table")
    print("- Two figures")
    print("- JSON summary")
    print("- Markdown report")

    print("\nBaseline comparison completed.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()

