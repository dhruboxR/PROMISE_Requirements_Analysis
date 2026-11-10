
"""
Phase 8 Experiment 03
Binary specialist feasibility: F vs O, F vs SC, F vs PO.

Uses fixed Phase 5 folds, training-fold-only TF-IDF,
and the original frozen MiniLM encoder.

This experiment evaluates binary discrimination only.
It does not claim improvement in 12-class classification.
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
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
PHASE4 = THESIS / "promise_advanced_models_phase_4"
PHASE5 = THESIS / "promise_improved_model_phase_5"

OUT = ROOT / "outputs"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "reports").mkdir(parents=True, exist_ok=True)

TARGETS = ["O", "SC", "PO"]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binary_weights(y):
    counts = pd.Series(y).value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Both binary classes required")

    n = len(y)
    raw = {
        cls: np.sqrt(n / (2 * counts[cls]))
        for cls in [0, 1]
    }
    mean = sum(
        counts[cls] * raw[cls] for cls in [0, 1]
    ) / n

    return {
        cls: float(raw[cls] / mean)
        for cls in [0, 1]
    }


def main():
    cv = load_module(
        PHASE4 / "scripts/01_group_aware_feature_comparison.py",
        "p4_cv_03",
    )
    semantic = load_module(
        PHASE4 / "scripts/17_semantic_embedding_logistic_cv.py",
        "p4_semantic_03",
    )

    if (
        cv.SEED != SEED
        or semantic.MODEL_NAME != MODEL_NAME
        or semantic.MAX_LENGTH != 128
    ):
        raise ValueError("Previous experiment settings mismatch")

    data = cv.load_development_data()
    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data = data.copy()
    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    groups = np.asarray(cv.create_connected_groups(data))
    labels = data["_class_"].astype(str).to_numpy()
    texts = data["NormalizedText"].astype(str).to_numpy()

    manifest = pd.read_csv(
        PHASE5 / "outputs/tables/01_fixed_fold_assignments.csv",
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or data["RecordID"].duplicated().any()
    ):
        raise ValueError("Invalid fixed-fold manifest")

    aligned = (
        manifest.set_index("RecordID")
        .loc[data["RecordID"]]
    )

    if not (
        aligned["ActualClass"].to_numpy() == labels
    ).all():
        raise ValueError("Class mismatch")

    if not (
        aligned["ProjectID"].astype(str).to_numpy()
        == data["ProjectID"].to_numpy()
    ).all():
        raise ValueError("Project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Unexpected folds")

    if (
        pd.DataFrame({"group": groups, "fold": folds})
        .groupby("group")["fold"].nunique() > 1
    ).any():
        raise ValueError("Connected-group leakage")

    semantic.set_seed()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    from transformers import AutoTokenizer, AutoModel

    print("Device:", device)
    print("Loading frozen MiniLM...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)
    encoder.eval()

    embeddings = semantic.generate_embeddings(
        texts.tolist(), tokenizer, encoder, device
    )

    if embeddings.shape != (3127, 384):
        raise ValueError("Unexpected embedding dimensions")

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    predictions = []

    for fold in [1, 2, 3]:
        print(f"\nFOLD {fold}")

        train_idx = np.flatnonzero(folds != fold)
        val_idx = np.flatnonzero(folds == fold)

        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError("Training-validation group overlap")

        vectorizer = cv.make_features("word+character")

        x_train_lex = vectorizer.fit_transform(
            texts[train_idx]
        )
        x_val_lex = vectorizer.transform(
            texts[val_idx]
        )

        x_train = hstack([
            x_train_lex,
            csr_matrix(embeddings[train_idx]),
        ], format="csr")

        x_val = hstack([
            x_val_lex,
            csr_matrix(embeddings[val_idx]),
        ], format="csr")

        for target in TARGETS:
            train_mask = np.isin(
                labels[train_idx], ["F", target]
            )
            val_mask = np.isin(
                labels[val_idx], ["F", target]
            )

            y_train = (
                labels[train_idx][train_mask] == target
            ).astype(int)
            y_val = (
                labels[val_idx][val_mask] == target
            ).astype(int)

            if len(np.unique(y_train)) != 2:
                raise ValueError(
                    f"Missing training class for {target}"
                )
            if len(np.unique(y_val)) != 2:
                raise ValueError(
                    f"Missing validation class for {target}"
                )

            model = LogisticRegression(
                C=10.0,
                class_weight=binary_weights(y_train),
                max_iter=2000,
                random_state=SEED,
            )

            model.fit(x_train[train_mask], y_train)

            probs = model.predict_proba(
                x_val[val_mask]
            )[:, list(model.classes_).index(1)]

            predicted = (probs >= 0.5).astype(int)

            metrics = {
                "Target": target,
                "Fold": fold,
                "TrainRecords": int(train_mask.sum()),
                "ValidationRecords": int(val_mask.sum()),
                "TargetSupport": int(y_val.sum()),
                "Accuracy": accuracy_score(
                    y_val, predicted
                ),
                "BalancedAccuracy": balanced_accuracy_score(
                    y_val, predicted
                ),
                "TargetPrecision": precision_score(
                    y_val, predicted, zero_division=0
                ),
                "TargetRecall": recall_score(
                    y_val, predicted, zero_division=0
                ),
                "TargetF1": f1_score(
                    y_val, predicted, zero_division=0
                ),
                "ROC_AUC": roc_auc_score(y_val, probs),
            }
            rows.append(metrics)

            selected_val = val_idx[val_mask]

            for idx, true_y, pred_y, prob in zip(
                selected_val, y_val, predicted, probs
            ):
                predictions.append({
                    "RecordID": data.iloc[idx]["RecordID"],
                    "Fold": fold,
                    "Target": target,
                    "ActualClass": labels[idx],
                    "ActualBinary": int(true_y),
                    "PredictedBinary": int(pred_y),
                    "TargetProbability": float(prob),
                })

            print(
                f"F vs {target}: "
                f"Recall={metrics['TargetRecall']:.4f} "
                f"F1={metrics['TargetF1']:.4f} "
                f"AUC={metrics['ROC_AUC']:.4f}"
            )

    metrics_df = pd.DataFrame(rows)
    predictions_df = pd.DataFrame(predictions)

    summary = (
        metrics_df.groupby("Target", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanBalancedAccuracy=("BalancedAccuracy", "mean"),
            MeanTargetPrecision=("TargetPrecision", "mean"),
            MeanTargetRecall=("TargetRecall", "mean"),
            MeanTargetF1=("TargetF1", "mean"),
            MeanROC_AUC=("ROC_AUC", "mean"),
        )
    )

    metrics_df.to_csv(
        OUT / "tables/03_specialist_fold_metrics.csv",
        index=False,
    )
    predictions_df.to_csv(
        OUT / "tables/03_specialist_oof_predictions.csv",
        index=False,
    )
    summary.to_csv(
        OUT / "tables/03_specialist_summary.csv",
        index=False,
    )

    metadata = {
        "experiment": "Phase 8 Experiment 03",
        "purpose": "Binary specialist feasibility",
        "targets": TARGETS,
        "development_records": 3127,
        "fixed_folds": 3,
        "classifier": "LogisticRegression",
        "class_weights": "binary tempered",
        "threshold": 0.5,
        "feature_representation": (
            "Word+character TF-IDF and frozen MiniLM"
        ),
        "test_accessed": False,
        "full_12_class_improvement_established": False,
        "outputs_are_exploratory": True,
    }
    (OUT / "reports/03_specialist_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSPECIALIST FEASIBILITY RESULTS")
    print("=" * 75)
    print(summary.to_string(index=False))
    print("\n12-class improvement established: No")
    print("Original test accessed: No")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
