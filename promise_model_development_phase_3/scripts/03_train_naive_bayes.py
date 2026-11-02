
"""PROMISE+ Phase 3: TF-IDF + Multinomial Naive Bayes."""

import json
import sys

import joblib
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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

    # Load the frozen training and validation partitions.
    train = load_split("train")
    validation = load_split("validation")

    X_train = train["NormalizedText"].astype(str)
    y_train = train["_class_"].astype(str)

    X_val = validation["NormalizedText"].astype(str)
    y_val = validation["_class_"].astype(str)

    print("\nPROMISE+ PHASE 3: MULTINOMIAL NAIVE BAYES")
    print("=" * 55)

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Training classes:", y_train.nunique())
    print("Random seed:", SEED)

    # Keep TF-IDF settings consistent with earlier experiments.
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
            MultinomialNB(
                alpha=1.0,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Multinomial Naive Bayes...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    metrics = evaluate(y_val, predictions)

    print("\nNaive Bayes validation metrics:")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save trained model and TF-IDF vectorizer.
    model_path = MODELS_DIR / "03_tfidf_naive_bayes.joblib"

    joblib.dump(model, model_path)

    # Save per-class classification report.
    report = full_report(y_val, predictions)

    (
        REPORTS_DIR / "03_naive_bayes_classification_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Save confusion matrix.
    confusion_table(y_val, predictions).to_csv(
        TABLES_DIR / "03_naive_bayes_confusion_matrix.csv"
    )

    # Save individual predictions.
    prediction_table = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    prediction_table["PredictedLabel"] = predictions

    prediction_table["Correct"] = (
        prediction_table["_class_"]
        == prediction_table["PredictedLabel"]
    )

    prediction_table.to_csv(
        TABLES_DIR / "03_naive_bayes_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save comparable metrics.
    metrics_table = pd.DataFrame([{
        "Model": "TF-IDF + Multinomial Naive Bayes",
        "Partition": "validation",
        **metrics,
    }])

    metrics_table.to_csv(
        METRICS_DIR / "03_naive_bayes_metrics.csv",
        index=False,
    )

    # Save experiment configuration.
    metadata = {
        "model": "TF-IDF + Multinomial Naive Bayes",
        "task": "12-class requirements classification",
        "training_records": len(train),
        "validation_records": len(validation),
        "random_seed": SEED,
        "tfidf_fit_partition": "train",
        "evaluation_partition": "validation",
        "alpha": 1.0,
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version,
        "test_partition_used": False,
        "validation_metrics": metrics,
    }

    (
        REPORTS_DIR / "03_naive_bayes_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nSaved model:", model_path)

    # Compare all completed baseline models.
    metric_files = [
        METRICS_DIR / "01_baseline_metrics.csv",
        METRICS_DIR / "02_svm_metrics.csv",
        METRICS_DIR / "03_naive_bayes_metrics.csv",
    ]

    tables = []

    for path in metric_files:
        if path.exists():
            tables.append(pd.read_csv(path))

    if tables:
        comparison = pd.concat(tables, ignore_index=True)

        comparison = comparison.sort_values(
            by="macro_f1",
            ascending=False,
        )

        print("\nMODEL COMPARISON")
        print(
            comparison[
                [
                    "Model",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                ]
            ].to_string(index=False)
        )

        comparison.to_csv(
            METRICS_DIR / "03_model_comparison.csv",
            index=False,
        )

    print("\nNaive Bayes experiment completed.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
