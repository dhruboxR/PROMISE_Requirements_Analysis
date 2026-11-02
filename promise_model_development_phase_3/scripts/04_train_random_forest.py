
"""PROMISE+ Phase 3: TF-IDF + Random Forest baseline."""

import json
import sys

import joblib
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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

    train = load_split("train")
    validation = load_split("validation")

    X_train = train["NormalizedText"].astype(str)
    y_train = train["_class_"].astype(str)

    X_val = validation["NormalizedText"].astype(str)
    y_val = validation["_class_"].astype(str)

    print("\nPROMISE+ PHASE 3: RANDOM FOREST")
    print("=" * 55)

    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Training classes:", y_train.nunique())
    print("Random seed:", SEED)

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
            RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nTraining TF-IDF + Random Forest...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    metrics = evaluate(y_val, predictions)

    print("\nRandom Forest validation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    model_path = MODELS_DIR / "04_tfidf_random_forest.joblib"
    joblib.dump(model, model_path)

    report = full_report(y_val, predictions)

    (
        REPORTS_DIR / "04_random_forest_classification_report.json"
    ).write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    confusion_table(y_val, predictions).to_csv(
        TABLES_DIR / "04_random_forest_confusion_matrix.csv"
    )

    prediction_table = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    prediction_table["PredictedLabel"] = predictions
    prediction_table["Correct"] = (
        prediction_table["_class_"]
        == prediction_table["PredictedLabel"]
    )

    prediction_table.to_csv(
        TABLES_DIR / "04_random_forest_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_table = pd.DataFrame([{
        "Model": "TF-IDF + Random Forest",
        "Partition": "validation",
        **metrics,
    }])

    metrics_table.to_csv(
        METRICS_DIR / "04_random_forest_metrics.csv",
        index=False,
    )

    metadata = {
        "model": "TF-IDF + Random Forest",
        "task": "12-class requirements classification",
        "training_records": len(train),
        "validation_records": len(validation),
        "random_seed": SEED,
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "tfidf_fit_partition": "train",
        "evaluation_partition": "validation",
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version,
        "test_partition_used": False,
        "validation_metrics": metrics,
    }

    (
        REPORTS_DIR / "04_random_forest_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8",
    )

    metric_files = [
        METRICS_DIR / "01_baseline_metrics.csv",
        METRICS_DIR / "02_svm_metrics.csv",
        METRICS_DIR / "03_naive_bayes_metrics.csv",
        METRICS_DIR / "04_random_forest_metrics.csv",
    ]

    tables = [
        pd.read_csv(path)
        for path in metric_files
        if path.exists()
    ]

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
        METRICS_DIR / "04_all_baseline_comparison.csv",
        index=False,
    )

    print("\nSaved model:", model_path)
    print("Random Forest experiment completed.")
    print("The test partition was not used.")


if __name__ == "__main__":
    main()
