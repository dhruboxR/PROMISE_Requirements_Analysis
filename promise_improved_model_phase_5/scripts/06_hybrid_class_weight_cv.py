
"""Phase 5 Experiment 06: Hybrid class-weight comparison.

Compare:
1. No class weighting
2. Tempered class weighting
3. Balanced class weighting

Uses the same TF-IDF, frozen MiniLM, and project-aware
folds as Phase 5 Experiment 02.

The original test partition is never accessed.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT.parent / "promise_advanced_models_phase_4"

TABLES = ROOT / "outputs/tables"
METRICS = ROOT / "outputs/metrics"
REPORTS = ROOT / "outputs/reports"

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_WEIGHT = 1.0

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_weights(y_train, mode):
    if mode == "none":
        return None

    counts = pd.Series(y_train).value_counts()
    n = len(y_train)
    k = len(LABELS)

    if set(counts.index) != set(LABELS):
        raise ValueError("Training fold lacks a class")

    balanced = {
        label: n / (k * int(counts[label]))
        for label in LABELS
    }

    if mode == "balanced":
        return balanced

    if mode == "tempered":
        raw = {
            label: np.sqrt(balanced[label])
            for label in LABELS
        }

        # Normalize weighted average to 1 so
        # regularization strength stays comparable.
        mean_weight = sum(
            int(counts[label]) * raw[label]
            for label in LABELS
        ) / n

        return {
            label: float(raw[label] / mean_weight)
            for label in LABELS
        }

    raise ValueError(f"Unknown mode: {mode}")


def calculate_metrics(actual, predicted):
    result = {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "weighted_f1": f1_score(
            actual, predicted,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }

    recalls = recall_score(
        actual, predicted,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    for label in ["O", "SC", "PO"]:
        result[f"recall_{label}"] = float(
            recalls[LABELS.index(label)]
        )

    return {
        key: float(value)
        for key, value in result.items()
    }


def main():
    for folder in [TABLES, METRICS, REPORTS]:
        folder.mkdir(parents=True, exist_ok=True)

    phase4_cv = load_module(
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py",
        "phase4_cv",
    )

    phase4_semantic = load_module(
        PHASE4 / "scripts"
        / "17_semantic_embedding_logistic_cv.py",
        "phase4_semantic",
    )

    if phase4_cv.SEED != SEED:
        raise ValueError("Seed mismatch")
    if phase4_cv.LABELS != LABELS:
        raise ValueError("Label mismatch")
    if phase4_semantic.MODEL_NAME != MODEL_NAME:
        raise ValueError("MiniLM mismatch")
    if phase4_semantic.MAX_LENGTH != 128:
        raise ValueError("Maximum token length mismatch")

    data = phase4_cv.load_development_data()

    texts = data["NormalizedText"].astype(str).to_numpy()
    labels = data["_class_"].astype(str).to_numpy()
    groups = phase4_cv.create_connected_groups(data)

    manifest = pd.read_csv(
        TABLES / "01_fixed_fold_assignments.csv",
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(data) != 3127 or len(set(groups)) != 66:
        raise ValueError("Unexpected development data")

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
    ):
        raise ValueError("Invalid fold manifest")

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )
    folds = list(cv.split(texts, labels, groups))

    for fold_number, (train_idx, val_idx) in enumerate(
        folds, start=1
    ):
        expected_ids = set(
            manifest.loc[
                manifest["Fold"].astype(int) == fold_number,
                "RecordID",
            ]
        )

        actual_ids = set(
            data.iloc[val_idx]["RecordID"].astype(str)
        )

        if expected_ids != actual_ids:
            raise ValueError(
                f"Fold {fold_number} assignment mismatch"
            )

        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError("Group leakage detected")

    print("\nPHASE 5: EXPERIMENT 06")
    print("HYBRID CLASS-WEIGHT COMPARISON")
    print("=" * 65)
    print("Development records:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Fold assignments: VERIFIED")

    from transformers import AutoModel, AutoTokenizer

    phase4_semantic.set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)
    encoder.eval()

    print("Generating frozen MiniLM embeddings...")

    embeddings = phase4_semantic.generate_embeddings(
        texts.tolist(),
        tokenizer,
        encoder,
        device,
    )

    if embeddings.shape != (3127, 384):
        raise ValueError("Invalid embedding dimensions")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    modes = ["none", "tempered", "balanced"]
    fold_rows = []
    prediction_rows = []

    for fold, (train_idx, val_idx) in enumerate(
        folds, start=1
    ):
        print(f"\nFOLD {fold}")
        print("-" * 65)

        vectorizer = phase4_cv.make_features(
            "word+character"
        )

        train_lexical = vectorizer.fit_transform(
            texts[train_idx]
        )
        val_lexical = vectorizer.transform(
            texts[val_idx]
        )

        x_train = hstack(
            [
                train_lexical,
                csr_matrix(
                    embeddings[train_idx] * SEMANTIC_WEIGHT
                ),
            ],
            format="csr",
        )

        x_val = hstack(
            [
                val_lexical,
                csr_matrix(
                    embeddings[val_idx] * SEMANTIC_WEIGHT
                ),
            ],
            format="csr",
        )

        for mode in modes:
            weights = calculate_weights(
                labels[train_idx], mode
            )

            classifier = LogisticRegression(
                C=10.0,
                class_weight=weights,
                max_iter=2000,
                random_state=SEED,
            )

            classifier.fit(
                x_train, labels[train_idx]
            )

            predicted = classifier.predict(x_val)

            metrics = calculate_metrics(
                labels[val_idx], predicted
            )

            fold_rows.append({
                "Configuration": mode,
                "Fold": fold,
                "TrainRecords": len(train_idx),
                "ValidationRecords": len(val_idx),
                **metrics,
            })

            frame = data.iloc[val_idx][
                ["RecordID", "ProjectID", "_class_"]
            ].copy()

            frame["Fold"] = fold
            frame["Configuration"] = mode
            frame["PredictedLabel"] = predicted

            prediction_rows.append(frame)

            print(
                f"{mode:<10} "
                f"accuracy={metrics['accuracy']:.4f} "
                f"macro-F1={metrics['macro_f1']:.4f} "
                f"balanced={metrics['balanced_accuracy']:.4f}"
            )

    folds_df = pd.DataFrame(fold_rows)

    folds_df.to_csv(
        METRICS / "06_class_weight_fold_metrics.csv",
        index=False,
    )

    predictions = pd.concat(
        prediction_rows, ignore_index=True
    )

    for mode in modes:
        subset = predictions[
            predictions["Configuration"] == mode
        ]

        if (
            len(subset) != 3127
            or subset["RecordID"].duplicated().any()
            or set(subset["RecordID"].astype(str))
               != set(data["RecordID"].astype(str))
        ):
            raise ValueError(
                f"Incomplete OOF predictions: {mode}"
            )

    predictions.to_csv(
        TABLES / "06_class_weight_oof_predictions.csv",
        index=False,
    )

    summary = (
        folds_df.groupby(
            "Configuration", as_index=False
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=(
                "balanced_accuracy", "mean"
            ),
            mean_weighted_f1=("weighted_f1", "mean"),
            mean_recall_O=("recall_O", "mean"),
            mean_recall_SC=("recall_SC", "mean"),
            mean_recall_PO=("recall_PO", "mean"),
        )
        .sort_values(
            "mean_macro_f1", ascending=False
        )
    )

    summary.to_csv(
        METRICS / "06_class_weight_summary.csv",
        index=False,
    )

    # Verify that the balanced setup reproduces
    # Phase 5 Experiment 02, Hybrid 1.0.
    earlier = pd.read_csv(
        METRICS / "02_hybrid_fold_metrics.csv"
    )

    earlier = earlier[
        earlier["Configuration"] == "Hybrid 1.0"
    ].sort_values("Fold")

    current = folds_df[
        folds_df["Configuration"] == "balanced"
    ].sort_values("Fold")

    if len(earlier) != 3 or len(current) != 3:
        raise ValueError("Missing reference folds")

    for metric in [
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "weighted_f1",
    ]:
        if not np.allclose(
            earlier[metric].to_numpy(),
            current[metric].to_numpy(),
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(
                f"Balanced reproduction mismatch: {metric}"
            )

    print("\nBalanced configuration reproduction: VERIFIED")

    print("\nTHREE-FOLD SUMMARY")
    print("=" * 65)
    print(summary.round(4).to_string(index=False))

    report = {
        "experiment": 6,
        "feature_representation": "TF-IDF + frozen MiniLM",
        "semantic_weight": SEMANTIC_WEIGHT,
        "classifier": "Logistic Regression",
        "C": 10.0,
        "weight_modes": modes,
        "seed": SEED,
        "fold_count": 3,
        "balanced_reproduction": "verified",
        "original_test_accessed": False,
        "development_results_are_exploratory": True,
        "summary": summary.to_dict(orient="records"),
    }

    (
        REPORTS / "06_class_weight_manifest.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment 06 completed.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
