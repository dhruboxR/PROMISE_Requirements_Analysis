
"""Phase 4 Experiment 02: Word-character feature weight tuning.

Reuses the frozen development data and group-aware folds
from Experiment 01. The original test partition is excluded.
"""

import importlib.util
import json
from pathlib import Path
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PHASE4_DIR = SCRIPT_DIR.parent
METRICS_DIR = PHASE4_DIR / "outputs" / "metrics"
REPORTS_DIR = PHASE4_DIR / "outputs" / "reports"

# Import shared functions from Experiment 01.
module_path = SCRIPT_DIR / "01_group_aware_feature_comparison.py"

spec = importlib.util.spec_from_file_location(
    "phase4_experiment01", module_path
)
experiment01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment01)

SEED = experiment01.SEED
LABELS = experiment01.LABELS


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data = experiment01.load_development_data()

    X = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()
    groups = experiment01.create_connected_groups(data)

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(cv.split(X, y, groups))

    for train_idx, val_idx in folds:
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError("Group leakage detected")

        if set(y[train_idx]) != set(LABELS):
            raise ValueError("Missing training classes")

        if set(y[val_idx]) != set(LABELS):
            raise ValueError("Missing validation classes")

    configurations = list(product(
        [1.0, 10.0],
        [0.5, 1.0, 2.0],
    ))

    print("\nPHASE 4: FEATURE WEIGHT TUNING")
    print("=" * 60)
    print("Development records:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Total configurations:", len(configurations))
    print("CV folds:", len(folds))

    results = []

    for config_id, (C, char_weight) in enumerate(
        configurations, start=1
    ):
        print(
            f"\nConfiguration {config_id}/6 "
            f"| C={C} | Character weight={char_weight}"
        )

        for fold_id, (train_idx, val_idx) in enumerate(
            folds, start=1
        ):
            features = experiment01.make_features(
                "word+character"
            )

            features.set_params(
                transformer_weights={
                    "word": 1.0,
                    "character": char_weight,
                }
            )

            pipeline = Pipeline([
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ])

            pipeline.fit(X[train_idx], y[train_idx])
            predictions = pipeline.predict(X[val_idx])

            row = {
                "configuration": config_id,
                "C": C,
                "word_weight": 1.0,
                "character_weight": char_weight,
                "fold": fold_id,
                "accuracy": accuracy_score(
                    y[val_idx], predictions
                ),
                "macro_f1": f1_score(
                    y[val_idx],
                    predictions,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y[val_idx], predictions
                ),
                "weighted_f1": f1_score(
                    y[val_idx],
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
            }

            results.append(row)

            print(
                f"Fold {fold_id} | "
                f"Accuracy={row['accuracy']:.4f} | "
                f"Macro-F1={row['macro_f1']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        METRICS_DIR / "02_feature_weight_fold_results.csv",
        index=False,
    )

    summary = (
        results_df.groupby(
            ["configuration", "C",
             "word_weight", "character_weight"],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=(
                "balanced_accuracy", "mean"
            ),
            mean_weighted_f1=("weighted_f1", "mean"),
        )
        .sort_values(
            "mean_macro_f1",
            ascending=False,
        )
    )

    summary.to_csv(
        METRICS_DIR / "02_feature_weight_summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("FEATURE WEIGHT TUNING COMPARISON")
    print("=" * 60)
    print(summary.round(4).to_string(index=False))

    best = summary.iloc[0]

    metadata = {
        "experiment": "Phase 4 feature weight tuning",
        "development_records": len(data),
        "connected_groups": len(set(groups)),
        "cv_folds": 3,
        "configurations": len(configurations),
        "selection_metric": "mean_macro_f1",
        "best_configuration": {
            "C": float(best["C"]),
            "word_weight": float(best["word_weight"]),
            "character_weight": float(
                best["character_weight"]
            ),
            "mean_macro_f1": float(best["mean_macro_f1"]),
        },
        "original_test_used": False,
    }

    (
        REPORTS_DIR / "02_feature_weight_summary.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nBest configuration:")
    print("C:", best["C"])
    print("Character weight:", best["character_weight"])
    print("Mean macro-F1:", round(best["mean_macro_f1"], 4))

    print("\nExperiment 02 completed.")
    print("Original test partition was not accessed.")


if __name__ == "__main__":
    main()
