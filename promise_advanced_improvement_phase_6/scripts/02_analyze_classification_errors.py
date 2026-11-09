
"""Phase 6 Experiment 02: Classification error analysis.

Analyzes Phase 5 out-of-fold predictions for:
- Unweighted, tempered and balanced models
- Per-class precision, recall and F1
- Confusion patterns
- Project-level performance
- Prediction changes between weighting strategies
- Misclassified requirement examples

Uses development data only. No model training.
"""

from pathlib import Path
import importlib.util
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

PHASE4 = THESIS / "promise_advanced_models_phase_4"
PHASE5 = THESIS / "promise_improved_model_phase_5"

PREDICTIONS = (
    PHASE5 / "outputs" / "tables"
    / "06_class_weight_oof_predictions.csv"
)

VERIFIED_FOLDS = (
    ROOT / "outputs" / "tables"
    / "01_verified_folds.csv"
)

TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

CONFIGURATIONS = ["none", "tempered", "balanced"]


def load_reference_data():
    path = (
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py"
    )

    spec = importlib.util.spec_from_file_location(
        "phase4_reference", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Phase 4 loader")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.load_development_data()


def save_csv(data, filename):
    path = TABLES / filename
    data.to_csv(path, index=False)
    print("Saved:", path)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("\nPHASE 6: EXPERIMENT 02")
    print("CLASSIFICATION ERROR ANALYSIS")
    print("=" * 65)

    predictions = pd.read_csv(
        PREDICTIONS,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    development = load_reference_data().copy()

    development["RecordID"] = (
        development["RecordID"].astype(str)
    )
    development["ProjectID"] = (
        development["ProjectID"].astype(str)
    )

    # Verify prediction data before analysis.
    required = {
        "RecordID", "ProjectID", "_class_",
        "Fold", "Configuration", "PredictedLabel"
    }
    if not required.issubset(predictions.columns):
        raise ValueError("Missing prediction columns")

    if len(predictions) != 3127 * 3:
        raise ValueError("Unexpected prediction row count")

    if set(predictions["Configuration"]) != set(
        CONFIGURATIONS
    ):
        raise ValueError("Unexpected configurations")

    if predictions.duplicated(
        ["RecordID", "Configuration"]
    ).any():
        raise ValueError("Duplicate OOF predictions")

    if set(predictions["_class_"]) != set(LABELS):
        raise ValueError("Unexpected true labels")

    if not set(predictions["PredictedLabel"]).issubset(
        set(LABELS)
    ):
        raise ValueError("Unexpected predicted labels")

    assert len(development) == 3127
    assert development["RecordID"].is_unique

    verified = pd.read_csv(
        VERIFIED_FOLDS,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(verified) != 3127
        or verified["RecordID"].duplicated().any()
    ):
        raise ValueError("Invalid Phase 6 fold audit")

    # Join requirement text from verified development data.
    reference = development[
        [
            "RecordID", "ProjectID",
            "NormalizedText", "_class_"
        ]
    ].rename(
        columns={"_class_": "ReferenceClass"}
    )

    data = predictions.merge(
        reference,
        on="RecordID",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_reference"),
    )

    if len(data) != 9381:
        raise ValueError("Development join mismatch")

    if not (
        data["ProjectID"]
        == data["ProjectID_reference"]
    ).all():
        raise ValueError("ProjectID mismatch")

    if not (
        data["_class_"]
        == data["ReferenceClass"]
    ).all():
        raise ValueError("Ground-truth label mismatch")

    fold_reference = verified[
        ["RecordID", "Phase6Fold", "ProjectID"]
    ].rename(
        columns={"ProjectID": "VerifiedProjectID"}
    )

    data = data.merge(
        fold_reference,
        on="RecordID",
        how="inner",
        validate="many_to_one",
    )

    if len(data) != 9381:
        raise ValueError("Fold audit join mismatch")

    if not (
        data["ProjectID"] == data["VerifiedProjectID"]
    ).all():
        raise ValueError("Verified project mismatch")

    if not (
        pd.to_numeric(data["Fold"])
        == pd.to_numeric(data["Phase6Fold"])
    ).all():
        raise ValueError("Fold mismatch")

    for config in CONFIGURATIONS:
        subset = data[data["Configuration"] == config]
        if len(subset) != 3127:
            raise ValueError(
                f"Incomplete OOF predictions: {config}"
            )
        if set(subset["RecordID"]) != set(
            development["RecordID"]
        ):
            raise ValueError(
                f"Record coverage mismatch: {config}"
            )

    print("Development records: 3127")
    print("OOF prediction rows:", len(data))
    print("Prediction and fold integrity: PASSED")

    metrics_rows = []
    class_rows = []
    project_rows = []
    confusion_rows = []
    error_rows = []
    summary = {}

    for config in CONFIGURATIONS:
        part = data[
            data["Configuration"] == config
        ].copy()

        actual = part["_class_"]
        predicted = part["PredictedLabel"]

        accuracy = accuracy_score(actual, predicted)
        macro_f1 = f1_score(
            actual, predicted,
            labels=LABELS, average="macro",
            zero_division=0,
        )
        balanced_accuracy = balanced_accuracy_score(
            actual, predicted
        )

        metrics_rows.append({
            "Configuration": config,
            "OOFAccuracy": accuracy,
            "OOFMacroF1": macro_f1,
            "OOFBalancedAccuracy": balanced_accuracy,
            "Correct": int((actual == predicted).sum()),
            "Incorrect": int((actual != predicted).sum()),
        })

        summary[config] = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "balanced_accuracy": float(
                balanced_accuracy
            ),
        }

        report = classification_report(
            actual,
            predicted,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        for label in LABELS:
            item = report[label]
            class_rows.append({
                "Configuration": config,
                "Class": label,
                "Support": int(item["support"]),
                "Precision": item["precision"],
                "Recall": item["recall"],
                "F1": item["f1-score"],
            })

        matrix = confusion_matrix(
            actual, predicted, labels=LABELS
        )

        for i, true_label in enumerate(LABELS):
            for j, predicted_label in enumerate(LABELS):
                confusion_rows.append({
                    "Configuration": config,
                    "ActualClass": true_label,
                    "PredictedClass": predicted_label,
                    "Count": int(matrix[i, j]),
                })

        part["Correct"] = (
            part["_class_"] == part["PredictedLabel"]
        )

        for project, group in part.groupby("ProjectID"):
            project_rows.append({
                "Configuration": config,
                "ProjectID": project,
                "Requirements": len(group),
                "Correct": int(group["Correct"].sum()),
                "Accuracy": float(
                    group["Correct"].mean()
                ),
            })

        incorrect = part[~part["Correct"]].copy()

        for _, row in incorrect.iterrows():
            error_rows.append({
                "Configuration": config,
                "RecordID": row["RecordID"],
                "ProjectID": row["ProjectID"],
                "Fold": int(row["Fold"]),
                "ActualClass": row["_class_"],
                "PredictedClass": row["PredictedLabel"],
                "NormalizedText": row["NormalizedText"],
            })

        print(
            f"\n{config.upper()}: "
            f"accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}, "
            f"balanced accuracy={balanced_accuracy:.4f}"
        )

    metrics = pd.DataFrame(metrics_rows)
    classes = pd.DataFrame(class_rows)
    projects = pd.DataFrame(project_rows)
    confusions = pd.DataFrame(confusion_rows)
    errors = pd.DataFrame(error_rows)

    save_csv(metrics, "02_oof_metrics.csv")
    save_csv(classes, "02_per_class_metrics.csv")
    save_csv(projects, "02_project_performance.csv")
    save_csv(confusions, "02_confusion_counts.csv")
    save_csv(errors, "02_misclassified_requirements.csv")

    # Compare predictions from all three configurations.
    wide = data.pivot(
        index="RecordID",
        columns="Configuration",
        values="PredictedLabel",
    )

    actual_info = (
        development[
            [
                "RecordID", "ProjectID",
                "_class_", "NormalizedText"
            ]
        ]
        .set_index("RecordID")
    )

    wide = actual_info.join(
        wide, how="inner",
        validate="one_to_one",
    )

    wide["TemperedCorrect"] = (
        wide["tempered"] == wide["_class_"]
    )
    wide["BalancedCorrect"] = (
        wide["balanced"] == wide["_class_"]
    )

    conditions = [
        wide["TemperedCorrect"] & wide["BalancedCorrect"],
        wide["TemperedCorrect"] & ~wide["BalancedCorrect"],
        ~wide["TemperedCorrect"] & wide["BalancedCorrect"],
    ]

    choices = [
        "both_correct",
        "tempered_only_correct",
        "balanced_only_correct",
    ]

    wide["Comparison"] = np.select(
        conditions,
        choices,
        default="both_incorrect",
    )

    comparison_counts = (
        wide["Comparison"]
        .value_counts()
        .to_dict()
    )

    save_csv(
        wide.reset_index(),
        "02_weighting_prediction_comparison.csv",
    )

    # Most frequent mistakes made by tempered model.
    tempered_confusions = confusions[
        (confusions["Configuration"] == "tempered")
        & (
            confusions["ActualClass"]
            != confusions["PredictedClass"]
        )
    ].sort_values("Count", ascending=False)

    tempered_classes = classes[
        classes["Configuration"] == "tempered"
    ].sort_values("Recall")

    difficult_projects = projects[
        projects["Configuration"] == "tempered"
    ].sort_values(
        ["Accuracy", "Requirements"],
        ascending=[True, False],
    )

    print("\nTEMPERED MODEL: LOWEST RECALL CLASSES")
    print(
        tempered_classes[
            ["Class", "Support", "Recall", "F1"]
        ].head(6).to_string(index=False)
    )

    print("\nTEMPERED MODEL: TOP CONFUSIONS")
    print(
        tempered_confusions[
            ["ActualClass", "PredictedClass", "Count"]
        ].head(10).to_string(index=False)
    )

    print("\nTEMPERED MODEL: LOWEST ACCURACY PROJECTS")
    print(
        difficult_projects[
            ["ProjectID", "Requirements", "Accuracy"]
        ].head(10).to_string(index=False)
    )

    print("\nTEMPERED VS BALANCED")
    print(comparison_counts)

    output_report = {
        "phase": 6,
        "experiment": 2,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "development_requirements": 3127,
        "configurations": CONFIGURATIONS,
        "oof_metrics": summary,
        "tempered_vs_balanced": {
            key: int(value)
            for key, value in comparison_counts.items()
        },
        "test_partition_accessed": False,
        "note": (
            "Analysis of existing Phase 5 held-out "
            "predictions. These are pooled OOF metrics, "
            "not mean fold metrics or independent test "
            "results."
        ),
    }

    report_path = (
        REPORTS / "02_error_analysis_summary.json"
    )

    report_path.write_text(
        json.dumps(output_report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", report_path)
    print("\nEXPERIMENT 02 COMPLETED")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
