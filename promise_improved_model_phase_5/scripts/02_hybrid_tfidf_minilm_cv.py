
"""PROMISE+ Phase 5, Experiment 02.

Hybrid TF-IDF + frozen MiniLM embeddings with Logistic Regression.

- Reuses Phase 4 Experiment 01 data and feature definitions.
- Reuses Phase 4 Experiment 17 MiniLM encoding procedure.
- Reproduces Phase 5 Experiment 01 fold assignments.
- Fits TF-IDF and Logistic Regression on each training fold only.
- Never accesses the original test partition.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold

from transformers import AutoModel, AutoTokenizer

PHASE5 = Path(__file__).resolve().parents[1]
THESIS = PHASE5.parent
PHASE4 = THESIS / "promise_advanced_models_phase_4"

TABLES = PHASE5 / "outputs" / "tables"
METRICS = PHASE5 / "outputs" / "metrics"
REPORTS = PHASE5 / "outputs" / "reports"

FOLD_FILE = TABLES / "01_fixed_fold_assignments.csv"

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Predefined configurations; do not expand them
# based on held-out fold results.
SEMANTIC_WEIGHTS = [0.5, 1.0]

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def import_source(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_scores(actual, predicted):
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "weighted_f1": float(
            f1_score(
                actual,
                predicted,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def make_classifier():
    return LogisticRegression(
        C=10.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )


def main():
    for folder in [TABLES, METRICS, REPORTS]:
        folder.mkdir(parents=True, exist_ok=True)

    if not FOLD_FILE.exists():
        raise FileNotFoundError(
            "Run Phase 5 Experiment 01 first"
        )

    original = import_source(
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py",
        "phase4_feature_comparison",
    )

    semantic = import_source(
        PHASE4 / "scripts"
        / "17_semantic_embedding_logistic_cv.py",
        "phase4_semantic_encoder",
    )

    if original.SEED != SEED:
        raise ValueError("Seed mismatch")

    if original.LABELS != LABELS:
        raise ValueError("Label order mismatch")

    if semantic.MODEL_NAME != MODEL_NAME:
        raise ValueError("MiniLM model mismatch")

    if semantic.MAX_LENGTH != 128:
        raise ValueError("MiniLM token length mismatch")

    data = original.load_development_data()
    texts = data["NormalizedText"].astype(str).to_numpy()
    labels = data["_class_"].astype(str).to_numpy()
    groups = original.create_connected_groups(data)

    print("\nPHASE 5: EXPERIMENT 02")
    print("HYBRID TF-IDF + MINILM")
    print("=" * 65)
    print("Requirements:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Classes:", len(set(labels)))

    if len(data) != 3127 or len(set(groups)) != 66:
        raise ValueError("Unexpected development dataset")

    # Verify the exact previously frozen fold assignments.
    expected = pd.read_csv(
        FOLD_FILE,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(expected) != len(data)
        or expected["RecordID"].duplicated().any()
    ):
        raise ValueError("Invalid Phase 5 fold manifest")

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(cv.split(texts, labels, groups))

    for fold_number, (tr_idx, va_idx) in enumerate(
        folds, 1
    ):
        observed = set(
            data.iloc[va_idx]["RecordID"].astype(str)
        )
        recorded = set(
            expected.loc[
                expected["Fold"].astype(int) == fold_number,
                "RecordID",
            ].astype(str)
        )

        if observed != recorded:
            raise ValueError(
                f"Fold {fold_number} differs from "
                "Experiment 01"
            )

        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise ValueError("Connected-group leakage")

    print("Phase 5 fold alignment: VERIFIED")

    # Generate embeddings with exactly the Phase 4
    # frozen MiniLM encoding procedure.
    semantic.set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nEmbedding device:", device)
    print("Loading frozen MiniLM...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
    encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder.to(device)
    encoder.eval()

    print("\nGenerating embeddings...")

    embeddings = semantic.generate_embeddings(
        texts.tolist(),
        tokenizer,
        encoder,
        device,
    )

    if embeddings.shape != (3127, 384):
        raise ValueError(
            f"Unexpected embedding shape: "
            f"{embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    configurations = [
        ("TF-IDF baseline", 0.0),
        ("Hybrid 0.5", 0.5),
        ("Hybrid 1.0", 1.0),
    ]

    fold_results = []
    all_predictions = []

    for fold_number, (tr_idx, va_idx) in enumerate(
        folds, start=1
    ):
        print(f"\n{'=' * 65}")
        print(f"FOLD {fold_number} / 3")
        print(f"{'=' * 65}")
        print("Train:", len(tr_idx))
        print("Held out:", len(va_idx))

        # Fit lexical representation on TRAINING data only.
        feature_extractor = original.make_features(
            "word+character"
        )

        x_train_lexical = (
            feature_extractor.fit_transform(
                texts[tr_idx]
            )
        )

        x_val_lexical = (
            feature_extractor.transform(
                texts[va_idx]
            )
        )

        # MiniLM is frozen and label-independent.
        x_train_semantic = csr_matrix(
            embeddings[tr_idx]
        )
        x_val_semantic = csr_matrix(
            embeddings[va_idx]
        )

        for name, semantic_weight in configurations:
            if semantic_weight == 0.0:
                train_features = x_train_lexical
                val_features = x_val_lexical
            else:
                train_features = hstack(
                    [
                        x_train_lexical,
                        x_train_semantic * semantic_weight,
                    ],
                    format="csr",
                )

                val_features = hstack(
                    [
                        x_val_lexical,
                        x_val_semantic * semantic_weight,
                    ],
                    format="csr",
                )

            classifier = make_classifier()
            classifier.fit(
                train_features,
                labels[tr_idx],
            )

            predicted = classifier.predict(
                val_features
            )

            scores = calculate_scores(
                labels[va_idx],
                predicted,
            )

            fold_results.append({
                "Configuration": name,
                "SemanticWeight": semantic_weight,
                "Fold": fold_number,
                "TrainRecords": len(tr_idx),
                "ValidationRecords": len(va_idx),
                **scores,
            })

            frame = data.iloc[va_idx][
                ["RecordID", "ProjectID", "_class_"]
            ].copy()

            frame["Fold"] = fold_number
            frame["Configuration"] = name
            frame["PredictedLabel"] = predicted

            all_predictions.append(frame)

            print(
                f"{name:<18} "
                f"Accuracy={scores['accuracy']:.4f} "
                f"Macro-F1={scores['macro_f1']:.4f} "
                f"Balanced={scores['balanced_accuracy']:.4f}"
            )

    results = pd.DataFrame(fold_results)

    results.to_csv(
        METRICS / "02_hybrid_fold_metrics.csv",
        index=False,
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    for name, _ in configurations:
        subset = predictions[
            predictions["Configuration"] == name
        ]

        if len(subset) != 3127:
            raise ValueError(
                f"Incomplete predictions for {name}"
            )

        if subset["RecordID"].duplicated().any():
            raise ValueError(
                f"Duplicate records for {name}"
            )

        if set(subset["RecordID"]) != set(
            data["RecordID"].astype(str)
        ):
            raise ValueError("Missing prediction records")

    predictions.to_csv(
        TABLES / "02_hybrid_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        results.groupby(
            ["Configuration", "SemanticWeight"],
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
            mean_weighted_f1=(
                "weighted_f1", "mean"
            ),
        )
        .sort_values(
            "mean_macro_f1",
            ascending=False,
        )
    )

    summary.to_csv(
        METRICS / "02_hybrid_cv_summary.csv",
        index=False,
    )

    print("\nTHREE-FOLD COMPARISON")
    print("=" * 65)
    print(
        summary.round(4).to_string(index=False)
    )

    # Store per-class out-of-fold reports.
    for name, _ in configurations:
        subset = predictions[
            predictions["Configuration"] == name
        ]

        report = classification_report(
            subset["_class_"],
            subset["PredictedLabel"],
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )

        filename = (
            name.lower().replace(" ", "_")
            .replace("-", "")
            .replace(".", "_")
        )

        (
            REPORTS / f"02_{filename}_class_report.json"
        ).write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

    metadata = {
        "phase": 5,
        "experiment": 2,
        "model": MODEL_NAME,
        "embedding_dimensions": 384,
        "embedding_max_length": 128,
        "lexical_features": "word and character TF-IDF",
        "classifier": "balanced Logistic Regression C=10",
        "semantic_weights": SEMANTIC_WEIGHTS,
        "seed": SEED,
        "connected_groups": len(set(groups)),
        "fold_count": 3,
        "original_test_accessed": False,
        "evaluation_note": (
            "Development folds have been examined in "
            "previous phases; these results are exploratory."
        ),
        "summary": summary.to_dict(orient="records"),
    }

    (
        REPORTS / "02_hybrid_experiment_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment 02 completed.")
    print("Original test partition was not accessed.")


if __name__ == "__main__":
    main()
