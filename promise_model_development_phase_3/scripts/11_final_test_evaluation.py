
"""PROMISE+ Phase 3: Final independent test evaluation.

Evaluates the frozen, validation-selected Logistic Regression pipeline.
No training or hyperparameter tuning is performed.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn

from common import (
    LABELS,
    SEED,
    SPLITS_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    prepare_outputs,
    evaluate,
    full_report,
    confusion_table,
)


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    prepare_outputs()

    print("\nPROMISE+ PHASE 3: FINAL TEST EVALUATION")
    print("=" * 65)

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    expected_counts = {
        "train": 2574,
        "validation": 553,
        "test": 550,
    }

    datasets = {}

    # Check all frozen split files before accessing test predictions.
    for name in ("train", "validation", "test"):
        info = manifest["splits"][name]
        path = SPLITS_DIR / info["file"]

        if not path.is_file():
            raise FileNotFoundError(path)

        observed_hash = file_sha256(path)

        if observed_hash != info["sha256"]:
            raise ValueError(f"{name} checksum mismatch")

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"ProjectID": str, "_class_": str},
            keep_default_na=False,
        )

        if len(df) != info["records"]:
            raise ValueError(f"{name} record count mismatch")

        if len(df) != expected_counts[name]:
            raise ValueError(f"Unexpected {name} size")

        required = {
            "RecordID",
            "ProjectID",
            "NormalizedText",
            "_class_",
        }

        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {name}")

        if df["RecordID"].duplicated().any():
            raise ValueError(f"Duplicate RecordIDs in {name}")

        if df["NormalizedText"].astype(str).str.strip().eq("").any():
            raise ValueError(f"Blank text in {name}")

        if not set(df["_class_"]).issubset(set(LABELS)):
            raise ValueError(f"Unknown labels in {name}")

        if df["_class_"].nunique() != info["classes"]:
            raise ValueError(f"Class coverage mismatch in {name}")

        if df["ProjectID"].nunique() != info["projects"]:
            raise ValueError(f"Project count mismatch in {name}")

        datasets[name] = df
        print(f"{name}: {len(df)} records, checksum PASS")

    train = datasets["train"]
    validation = datasets["validation"]
    test = datasets["test"]

    # Independently verify project, text, and RecordID isolation.
    names = ["train", "validation", "test"]

    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left = datasets[left_name]
            right = datasets[right_name]

            if set(left["RecordID"]) & set(right["RecordID"]):
                raise ValueError("RecordID overlap detected")

            if set(left["ProjectID"]) & set(right["ProjectID"]):
                raise ValueError("Project leakage detected")

            left_text = set(
                left["NormalizedText"].astype(str)
                .str.casefold().str.split().str.join(" ")
            )

            right_text = set(
                right["NormalizedText"].astype(str)
                .str.casefold().str.split().str.join(" ")
            )

            if left_text & right_text:
                raise ValueError("Normalized-text leakage detected")

    if sum(len(df) for df in datasets.values()) != manifest["records"]:
        raise ValueError("Total record count mismatch")

    print("Record, project, and text isolation: PASS")

    # Verify the original model-selection record.
    selection_path = (
        REPORTS_DIR / "07_logistic_regression_tuning_summary.json"
    )

    selection = json.loads(
        selection_path.read_text(encoding="utf-8")
    )

    config = selection["best_configuration"]

    if not (
        float(config["C"]) == 10.0
        and config["class_weight"] == "balanced"
        and config["ngram_range"] == "(1, 2)"
        and int(config["min_df"]) == 1
    ):
        raise ValueError("Unexpected selected configuration")

    model_path = (
        MODELS_DIR / "07_tuned_logistic_regression.joblib"
    )

    # Do not overwrite an existing final evaluation.
    final_report_path = (
        REPORTS_DIR / "11_final_test_evaluation.json"
    )

    if final_report_path.exists():
        raise FileExistsError(
            "Final evaluation already exists. "
            "Do not rerun or overwrite the locked test results."
        )

    model_hash = file_sha256(model_path)

    # joblib files can execute code during loading.
    # Load only the model artifact generated by your own scripts.
    model = joblib.load(model_path)

    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    if (
        classifier.C != 10.0
        or classifier.class_weight != "balanced"
        or tfidf.ngram_range != (1, 2)
        or tfidf.min_df != 1
    ):
        raise ValueError("Saved model configuration mismatch")

    print("Selected model configuration: PASS")
    print("Model SHA256:", model_hash)
    print("\nRunning final test predictions...")

    X_test = test["NormalizedText"].astype(str)
    y_test = test["_class_"].astype(str)

    predictions = model.predict(X_test)

    metrics = evaluate(y_test, predictions)
    report = full_report(y_test, predictions)

    print("\nFINAL TEST METRICS")
    print("=" * 65)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nPER-CLASS RESULTS")
    class_rows = []

    for label in LABELS:
        values = report[label]

        class_rows.append({
            "Class": label,
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1-score"],
            "Support": int(values["support"]),
        })

    class_table = pd.DataFrame(class_rows)

    print(class_table.round(4).to_string(index=False))

    # Build prediction-level test record.
    pred_table = test[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    pred_table["PredictedLabel"] = predictions
    pred_table["Correct"] = (
        pred_table["_class_"] == pred_table["PredictedLabel"]
    )

    project_table = (
        pred_table.groupby("ProjectID")
        .agg(
            Records=("RecordID", "size"),
            Correct=("Correct", "sum"),
        )
        .reset_index()
    )

    project_table["Incorrect"] = (
        project_table["Records"] - project_table["Correct"]
    )

    project_table["Accuracy"] = (
        project_table["Correct"] / project_table["Records"]
    )

    project_table = project_table.sort_values("ProjectID")

    print("\nPROJECT-LEVEL TEST ACCURACY")
    print(project_table.round(4).to_string(index=False))

    # Save all final evaluation artifacts.
    pd.DataFrame([{
        "Model": "Tuned TF-IDF + Logistic Regression",
        "Partition": "test",
        **metrics,
    }]).to_csv(
        METRICS_DIR / "11_final_test_metrics.csv",
        index=False,
    )

    class_table.to_csv(
        TABLES_DIR / "11_test_per_class_metrics.csv",
        index=False,
    )

    confusion_table(y_test, predictions).to_csv(
        TABLES_DIR / "11_test_confusion_matrix.csv"
    )

    pred_table.to_csv(
        TABLES_DIR / "11_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    project_table.to_csv(
        TABLES_DIR / "11_test_project_performance.csv",
        index=False,
    )

    result = {
        "experiment": "Final independent test evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": "Tuned TF-IDF + Logistic Regression",
        "model_sha256": model_hash,
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "test_sha256": manifest["splits"]["test"]["sha256"],
        "selected_hyperparameters": {
            "C": 10.0,
            "class_weight": "balanced",
            "ngram_range": [1, 2],
            "min_df": 1,
        },
        "model_selection": "validation macro-F1",
        "training_records": len(train),
        "validation_records": len(validation),
        "test_records": len(test),
        "test_projects": int(test["ProjectID"].nunique()),
        "test_classes": int(test["_class_"].nunique()),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "integrity_checks": "passed",
        "metrics": metrics,
        "per_class_results": class_rows,
        "annotation_conflicts_pending":
            manifest.get("manual_annotation_conflicts_pending"),
    }

    final_report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nFinal artifacts saved successfully.")
    print("Final evaluation completed. No model retraining performed.")


if __name__ == "__main__":
    main()
