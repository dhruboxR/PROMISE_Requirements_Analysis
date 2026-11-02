
"""PROMISE+ Phase 3: Linear SVM hyperparameter tuning."""

import json
import hashlib
import time

import joblib
import pandas as pd
import sklearn

from itertools import product
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from common import (
    SEED,
    MODELS_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    prepare_outputs,
    load_split,
    evaluate,
    full_report,
    confusion_table,
)


def main():
    prepare_outputs()

    train = load_split("train")
    validation = load_split("validation")

    X_train = train["NormalizedText"].astype(str)
    y_train = train["_class_"].astype(str)

    X_val = validation["NormalizedText"].astype(str)
    y_val = validation["_class_"].astype(str)

    print("\nPROMISE+ PHASE 3: LINEAR SVM TUNING")
    print("=" * 60)

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Random seed:", SEED)

    # Parameter combinations.
    C_values = [0.1, 1.0, 10.0]
    class_weights = [None, "balanced"]
    ngram_ranges = [(1, 1), (1, 2)]
    min_df_values = [1, 2]

    configurations = list(product(
        C_values,
        class_weights,
        ngram_ranges,
        min_df_values,
    ))

    print("Total configurations:", len(configurations))

    results = []
    best_score = -1.0
    best_config = None
    best_model = None

    start_time = time.time()

    for idx, (C, weight, ngram, min_df) in enumerate(
        configurations, start=1
    ):
        print(f"\n[{idx}/{len(configurations)}]")
        print(
            f"C={C}, weight={weight}, "
            f"ngram={ngram}, min_df={min_df}"
        )

        model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=ngram,
                    min_df=min_df,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=C,
                    class_weight=weight,
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ])

        # Fit all text features exclusively on training data.
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        scores = evaluate(y_val, predictions)

        row = {
            "Configuration": idx,
            "C": C,
            "class_weight": str(weight),
            "ngram_range": str(ngram),
            "min_df": min_df,
            **scores,
        }

        results.append(row)

        print(
            f"Accuracy={scores['accuracy']:.4f}, "
            f"Macro-F1={scores['macro_f1']:.4f}, "
            f"Balanced accuracy="
            f"{scores['balanced_accuracy']:.4f}"
        )

        # Primary selection metric: validation macro-F1.
        # Keep the first configuration in case of a tie.
        if scores["macro_f1"] > best_score:
            best_score = scores["macro_f1"]
            best_config = row.copy()
            best_model = model

    # Save results from all configurations.
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
        kind="stable",
    )

    results_df.to_csv(
        METRICS_DIR / "06_svm_tuning_results.csv",
        index=False,
    )

    # Save best trained pipeline.
    model_path = (
        MODELS_DIR / "06_tuned_linear_svm.joblib"
    )

    joblib.dump(best_model, model_path)

    best_predictions = best_model.predict(X_val)
    best_metrics = evaluate(y_val, best_predictions)

    # Save detailed class-wise report.
    report = full_report(y_val, best_predictions)

    (
        REPORTS_DIR / "06_tuned_svm_classification_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Save confusion matrix.
    confusion_table(y_val, best_predictions).to_csv(
        TABLES_DIR / "06_tuned_svm_confusion_matrix.csv"
    )

    # Save predictions for error analysis.
    prediction_table = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    prediction_table["PredictedLabel"] = best_predictions
    prediction_table["Correct"] = (
        prediction_table["_class_"]
        == prediction_table["PredictedLabel"]
    )

    prediction_table.to_csv(
        TABLES_DIR / "06_tuned_svm_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    elapsed = time.time() - start_time

    # Save configuration and experiment metadata.
    metadata = {
        "experiment": "Linear SVM hyperparameter tuning",
        "selection_metric": "validation_macro_f1",
        "total_configurations": len(configurations),
        "best_configuration": best_config,
        "best_validation_metrics": best_metrics,
        "training_records": len(train),
        "validation_records": len(validation),
        "random_seed": SEED,
        "scikit_learn_version": sklearn.__version__,
        "elapsed_seconds": round(elapsed, 2),
        "test_partition_used": False,
    }

    (
        REPORTS_DIR / "06_svm_tuning_summary.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("LINEAR SVM TUNING COMPLETED")
    print("=" * 60)

    print("\nTOP 10 CONFIGURATIONS")
    print(
        results_df[
            [
                "Configuration",
                "C",
                "class_weight",
                "ngram_range",
                "min_df",
                "accuracy",
                "macro_f1",
                "balanced_accuracy",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nBEST CONFIGURATION")
    print("C:", best_config["C"])
    print("Class weight:", best_config["class_weight"])
    print("N-gram:", best_config["ngram_range"])
    print("Minimum document frequency:", best_config["min_df"])

    print("\nBEST VALIDATION METRICS")
    for key, value in best_metrics.items():
        print(f"{key}: {value:.4f}")

    baseline_path = METRICS_DIR / "02_svm_metrics.csv"

    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path).iloc[0]
        improvement = (
            best_metrics["macro_f1"] - baseline["macro_f1"]
        )

        print("\nORIGINAL SVM MACRO-F1:",
              f"{baseline['macro_f1']:.4f}")
        print("TUNED SVM MACRO-F1:",
              f"{best_metrics['macro_f1']:.4f}")
        print("CHANGE:", f"{improvement:+.4f}")

    print("\nSaved model:", model_path)
    print("Elapsed seconds:", round(elapsed, 2))
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
