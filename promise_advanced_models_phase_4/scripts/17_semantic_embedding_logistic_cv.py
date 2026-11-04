
"""Phase 4 Experiment 17: Semantic embeddings + Logistic Regression.

Uses:
- Phase 2 train + validation only
- Exact group-aware folds from Experiment 01
- Frozen pretrained MiniLM sentence embeddings
- Fold-specific Logistic Regression fitting

Original test partition is never accessed.
"""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
REPORTS = ROOT / "outputs" / "reports"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 16

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def load_original():
    path = SCRIPTS / "01_group_aware_feature_comparison.py"

    spec = importlib.util.spec_from_file_location(
        "original_experiment_01", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def mean_pool(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(
        hidden_states.dtype
    )

    summed = torch.sum(hidden_states * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed / counts


@torch.inference_mode()
def generate_embeddings(texts, tokenizer, model, device):
    all_vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        output = model(**encoded)

        vectors = mean_pool(
            output.last_hidden_state,
            encoded["attention_mask"],
        )

        vectors = torch.nn.functional.normalize(
            vectors, p=2, dim=1
        )

        all_vectors.append(
            vectors.cpu().numpy().astype(np.float32)
        )

        completed = min(start + BATCH_SIZE, len(texts))

        if completed % 320 == 0 or completed == len(texts):
            print(
                f"Encoded {completed}/{len(texts)} requirements"
            )

    return np.vstack(all_vectors)


def compute_scores(actual, predicted):
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual, predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual, predicted,
                labels=LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def main():
    for directory in [TABLES, METRICS, REPORTS]:
        directory.mkdir(parents=True, exist_ok=True)

    set_seed()

    original = load_original()

    if original.SEED != SEED or original.LABELS != LABELS:
        raise ValueError("Experiment 01 settings mismatch")

    data = original.load_development_data()

    texts = data["NormalizedText"].astype(str).tolist()
    labels = data["_class_"].astype(str).to_numpy()
    groups = original.create_connected_groups(data)

    print("\nEXPERIMENT 17: SEMANTIC EMBEDDING CLASSIFICATION")
    print("=" * 65)
    print("Development requirements:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Model:", MODEL_NAME)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    encoder = AutoModel.from_pretrained(MODEL_NAME)
    encoder.to(device)
    encoder.eval()

    print("\nGenerating semantic embeddings...")

    embeddings = generate_embeddings(
        texts, tokenizer, encoder, device
    )

    if embeddings.shape[0] != len(data):
        raise ValueError("Embedding record mismatch")

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings found")

    print("Embedding shape:", embeddings.shape)

    # The encoder is frozen and uses no task labels.
    # Only the Logistic Regression classifier is fitted
    # separately on each training fold.

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(cv.split(
        np.array(texts, dtype=object),
        labels,
        groups,
    ))

    fold_results = []
    predictions = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds, start=1
    ):
        print(f"\nFOLD {fold_number}")
        print("Training:", len(train_idx))
        print("Validation:", len(val_idx))

        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError("Connected-group leakage")

        if set(labels[train_idx]) != set(LABELS):
            raise ValueError("Missing training classes")

        if set(labels[val_idx]) != set(LABELS):
            raise ValueError("Missing validation classes")

        classifier = LogisticRegression(
            C=10.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=SEED,
        )

        classifier.fit(
            embeddings[train_idx],
            labels[train_idx],
        )

        predicted = classifier.predict(
            embeddings[val_idx]
        )

        scores = compute_scores(
            labels[val_idx],
            predicted,
        )

        fold_results.append({
            "Fold": fold_number,
            "TrainRecords": len(train_idx),
            "ValidationRecords": len(val_idx),
            **scores,
        })

        frame = data.iloc[val_idx][
            ["RecordID", "ProjectID", "_class_"]
        ].copy()

        frame["Fold"] = fold_number
        frame["PredictedLabel"] = predicted

        predictions.append(frame)

        print(
            f"Accuracy: {scores['accuracy']:.4f} | "
            f"Macro-F1: {scores['macro_f1']:.4f} | "
            f"Balanced: {scores['balanced_accuracy']:.4f}"
        )

    results = pd.DataFrame(fold_results)

    results.to_csv(
        METRICS / "17_semantic_embedding_fold_metrics.csv",
        index=False,
    )

    combined = pd.concat(
        predictions, ignore_index=True
    )

    if len(combined) != len(data):
        raise ValueError("Incomplete OOF predictions")

    if combined["RecordID"].duplicated().any():
        raise ValueError("Duplicate OOF RecordIDs")

    if set(combined["RecordID"]) != set(data["RecordID"]):
        raise ValueError("Missing OOF requirements")

    combined.to_csv(
        TABLES / "17_semantic_embedding_oof_predictions.csv",
        index=False,
    )

    summary = {}

    print("\nTHREE-FOLD SUMMARY")
    print("=" * 65)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]:
        mean = results[metric].mean()
        std = results[metric].std(ddof=1)

        summary[metric] = {
            "mean": float(mean),
            "std": float(std),
        }

        print(
            f"{metric:<20} "
            f"mean={mean:.4f} std={std:.4f}"
        )

    oof = compute_scores(
        combined["_class_"],
        combined["PredictedLabel"],
    )

    print("\nOUT-OF-FOLD AGGREGATE")
    for metric, value in oof.items():
        print(f"{metric}: {value:.4f}")

    report = classification_report(
        combined["_class_"],
        combined["PredictedLabel"],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    (
        REPORTS / "17_semantic_embedding_class_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    (
        REPORTS / "17_semantic_embedding_summary.json"
    ).write_text(
        json.dumps({
            "experiment": 17,
            "model": MODEL_NAME,
            "embedding_dimensions": int(embeddings.shape[1]),
            "max_length": MAX_LENGTH,
            "classifier": "LogisticRegression",
            "classifier_C": 10.0,
            "class_weight": "balanced",
            "seed": SEED,
            "fold_summary": summary,
            "oof": oof,
            "original_test_accessed": False,
        }, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment 17 completed.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
