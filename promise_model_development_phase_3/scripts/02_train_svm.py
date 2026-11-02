
"""PROMISE+ Phase 3: TF-IDF + Linear SVM baseline."""

import json
import sys

import joblib
import pandas as pd
import sklearn

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

    # Load frozen Phase 2 training and validation sets.
    train = load_split("train")
    validation = load_split("validation")

    X_train = train["NormalizedText"].astype(str)
    y_train = train["_class_"].astype(str)

    X_val = validation["NormalizedText"].astype(str)
    y_val = validation["_class_"].astype(str)

    print("\nPROMISE+ PHASE 3: LINEAR SVM")
    print("=" * 55)

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Training classes:", y_train.nunique())
    print("Random seed:", SEED)

    # TF-IDF and classifier are trained together.
    # Validation data never enters fit().
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LinearSVC(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=SEED,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Linear SVM...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = evaluate(y_val, predictions)

    print("\nLinear SVM validation metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save the trained pipeline.
    model_path = MODELS_DIR / "02_tfidf_linear_svm.joblib"

    joblib.dump(model, model_path)

    # Save classification report.
    report = full_report(y_val, predictions)

    (
        REPORTS_DIR / "02_svm_classification_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Save confusion matrix.
    confusion_table(y_val, predictions).to_csv(
        TABLES_DIR / "02_svm_confusion_matrix.csv"
    )

    # Save validation predictions for error analysis.
    prediction_table = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    prediction_table["PredictedLabel"] = predictions

    prediction_table["Correct"] = (
        prediction_table["_class_"]
        == prediction_table["PredictedLabel"]
    )

    prediction_table.to_csv(
        TABLES_DIR / "02_svm_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save metrics in the same format as Experiment 1.
    metrics_table = pd.DataFrame([{
        "Model": "TF-IDF + Linear SVM",
        "Partition": "validation",
        **metrics,
    }])

    metrics_table.to_csv(
        METRICS_DIR / "02_svm_metrics.csv",
        index=False,
    )

    metadata = {
        "model": "TF-IDF + Linear SVM",
        "task": "12-class requirements classification",
        "training_records": len(train),
        "validation_records": len(validation),
        "random_seed": SEED,
        "tfidf_fit_partition": "train",
        "evaluation_partition": "validation",
        "class_weight": "balanced",
        "C": 1.0,
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version,
        "test_partition_used": False,
        "validation_metrics": metrics,
    }

    (
        REPORTS_DIR / "02_svm_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nSaved model:", model_path)

    # Compare with Logistic Regression.
    baseline_path = (
        METRICS_DIR / "01_baseline_metrics.csv"
    )

    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)

        comparison = pd.concat(
            [baseline, metrics_table],
            ignore_index=True,
        )

        print("\nMODEL COMPARISON")
        print(
            comparison[
                ["Model", "accuracy", "balanced_accuracy",
                 "macro_f1", "weighted_f1"]
            ].to_string(index=False)
        )

        comparison.to_csv(
            METRICS_DIR / "02_baseline_svm_comparison.csv",
            index=False,
        )

    print("\nLinear SVM experiment completed.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
