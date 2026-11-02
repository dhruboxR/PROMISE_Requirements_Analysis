
"""PROMISE+ multiclass baseline: Majority class and TF-IDF + Logistic Regression."""

import json
import sys

import joblib
import pandas as pd
import sklearn

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import (
    SEED,
    LABELS,
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

    # Load only training and validation datasets.
    train = load_split("train")
    validation = load_split("validation")

    X_train = train["NormalizedText"].astype(str)
    y_train = train["_class_"].astype(str)

    X_val = validation["NormalizedText"].astype(str)
    y_val = validation["_class_"].astype(str)

    print("\nPROMISE+ PHASE 3: BASELINE CLASSIFICATION")
    print("=" * 55)

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Classes:", len(LABELS))
    print("Random seed:", SEED)

    # Majority-class reference baseline.
    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        [[0]] * len(y_train),
        y_train,
    )

    dummy_predictions = dummy.predict(
        [[0]] * len(y_val)
    )

    dummy_metrics = evaluate(
        y_val, dummy_predictions
    )

    print("\nMajority-class baseline:")
    for metric, value in dummy_metrics.items():
        print(f"{metric}: {value:.4f}")

    # TF-IDF is fitted on training data only.
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
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=SEED,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Logistic Regression...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    metrics = evaluate(y_val, predictions)

    print("\nLogistic Regression validation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save model and fitted TF-IDF vectorizer together.
    model_path = MODELS_DIR / "01_tfidf_logistic_regression.joblib"
    joblib.dump(model, model_path)

    # Save detailed class-wise metrics.
    report = full_report(y_val, predictions)

    report_path = (
        REPORTS_DIR / "01_logistic_regression_classification_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    # Save confusion matrix.
    confusion = confusion_table(y_val, predictions)
    confusion.to_csv(
        TABLES_DIR / "01_logistic_regression_confusion_matrix.csv"
    )

    # Save validation predictions for later error analysis.
    prediction_table = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    prediction_table["PredictedLabel"] = predictions
    prediction_table["Correct"] = (
        prediction_table["_class_"]
        == prediction_table["PredictedLabel"]
    )

    prediction_table.to_csv(
        TABLES_DIR / "01_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save comparable metrics for future models.
    rows = []

    for name, results in [
        ("Majority Class", dummy_metrics),
        ("TF-IDF + Logistic Regression", metrics),
    ]:
        rows.append({
            "Model": name,
            "Partition": "validation",
            **results,
        })

    metrics_table = pd.DataFrame(rows)

    metrics_table.to_csv(
        METRICS_DIR / "01_baseline_metrics.csv",
        index=False,
    )

    metadata = {
        "task": "12-class requirements classification",
        "training_records": len(train),
        "validation_records": len(validation),
        "random_seed": SEED,
        "text_column": "NormalizedText",
        "label_column": "_class_",
        "tfidf_fit_partition": "train",
        "evaluation_partition": "validation",
        "class_weight": "balanced",
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "test_partition_used": False,
        "validation_metrics": metrics,
    }

    (
        REPORTS_DIR / "01_baseline_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    print("\nComparison:")
    print(metrics_table.to_string(index=False))

    print("\nSaved model:", model_path)
    print("\nBaseline experiment completed.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
