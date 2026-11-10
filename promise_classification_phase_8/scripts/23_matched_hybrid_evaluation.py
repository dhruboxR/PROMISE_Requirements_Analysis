
"""
Phase 8 Experiment 23B
Matched hybrid vs mixed-sampling MiniLM evaluation.

Hybrid:
  Word + character TF-IDF
  Frozen MiniLM embeddings
  Logistic regression C=10
  Normalized sqrt-balanced class weights

Evaluate the same nine inner partitions as Experiment 22.
No outer-fold validation or original test evaluation.
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
    accuracy_score, f1_score, recall_score,
    balanced_accuracy_score
)
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoTokenizer, AutoModel


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

P4 = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)
SEMANTIC = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/17_semantic_embedding_logistic_cv.py"
)
PHASE5 = (
    THESIS / "promise_improved_model_phase_5"
    / "scripts/06_hybrid_class_weight_cv.py"
)
MANIFEST = (
    THESIS / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)
MIXED_RESULTS = (
    TABLES / "22_three_strategy_fold_metrics.csv"
)

SEED = 42
SEMANTIC_WEIGHT = 1.0
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tempered_weights(y_train):
    """
    Match Phase 5 sqrt-balanced weighting for
    classes present in this inner training partition.

    Missing classes are omitted because logistic
    regression cannot learn an absent class.
    """
    counts = pd.Series(y_train).value_counts()
    n = len(y_train)
    k = len(LABELS)

    raw = {
        label: np.sqrt(n / (k * int(count)))
        for label, count in counts.items()
    }

    weighted_mean = sum(
        int(counts[label]) * raw[label]
        for label in counts.index
    ) / n

    return {
        label: float(raw[label] / weighted_mean)
        for label in counts.index
    }


def score_predictions(actual, predicted):
    recalls = recall_score(
        actual, predicted,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    output = {
        "Accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "MacroF1": float(
            f1_score(
                actual, predicted,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "BalancedAccuracyObserved": float(
            balanced_accuracy_score(actual, predicted)
        ),
    }

    for label, recall in zip(LABELS, recalls):
        output[f"Recall_{label}"] = float(recall)

    return output


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    phase4 = import_script(P4, "p4_exp23")
    semantic = import_script(SEMANTIC, "semantic_exp23")
    phase5 = import_script(PHASE5, "p5_exp23")

    if (
        phase4.SEED != SEED
        or phase4.LABELS != LABELS
        or phase5.LABELS != LABELS
        or phase5.MODEL_NAME != MODEL_NAME
        or phase5.SEMANTIC_WEIGHT != SEMANTIC_WEIGHT
    ):
        raise ValueError("Original model settings mismatch")

    if not MIXED_RESULTS.exists():
        raise FileNotFoundError(MIXED_RESULTS)

    old = pd.read_csv(MIXED_RESULTS)

    mixed = old[
        old["Model"] == "Mixed75_25CE"
    ].copy()

    expected_keys = {
        (outer, inner)
        for outer in (1, 2, 3)
        for inner in (1, 2, 3)
    }

    mixed_keys = set(zip(
        mixed["OuterFold"].astype(int),
        mixed["InnerFold"].astype(int),
    ))

    if (
        len(mixed) != 9
        or mixed_keys != expected_keys
        or not (mixed["Epoch"] == 4).all()
    ):
        raise ValueError("Incomplete Experiment 22 results")

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID")

    labels = data["_class_"].astype(str).to_numpy()
    texts = data["NormalizedText"].astype(str).to_numpy()
    projects = data["ProjectID"].to_numpy()

    if set(labels) != set(LABELS):
        raise ValueError("Unexpected labels")

    groups = np.asarray(
        phase4.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"]) != set(data["RecordID"])
    ):
        raise ValueError("Invalid fold manifest")

    aligned = manifest.set_index("RecordID").loc[
        data["RecordID"]
    ]

    if not np.array_equal(
        aligned["ActualClass"].astype(str).to_numpy(),
        labels
    ):
        raise ValueError("Manifest label mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        projects
    ):
        raise ValueError("Manifest project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Invalid outer fold assignments")

    if (
        pd.DataFrame({
            "Group": groups,
            "Fold": folds,
        }).groupby("Group")["Fold"].nunique() > 1
    ).any():
        raise ValueError("Outer group leakage")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    semantic.set_seed()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    encoder.eval()

    print("\nPHASE 8 EXPERIMENT 23B")
    print("=" * 70)
    print("Device:", device)
    print("Inner evaluations: 9")
    print("Classifier: Phase 5 hybrid, tempered weights")
    print("Original test accessed: No")
    print("Outer validation evaluated: No", flush=True)

    results = []
    prediction_rows = []
    support_rows = []

    for outer in (1, 2, 3):
        outer_train = np.flatnonzero(folds != outer)

        # Frozen embeddings are computed for the
        # current outer-training partition only.
        # Outer-held-out text is not encoded.
        print(
            f"\nOUTER TRAINING PARTITION {outer}",
            flush=True
        )

        outer_embeddings = semantic.generate_embeddings(
            texts[outer_train].tolist(),
            tokenizer,
            encoder,
            device,
        )

        if outer_embeddings.shape != (
            len(outer_train), 384
        ):
            raise ValueError("Incorrect embedding shape")

        if not np.isfinite(outer_embeddings).all():
            raise ValueError("Non-finite embeddings")

        inner_cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED,
        )

        for inner, (local_tr, local_val) in enumerate(
            inner_cv.split(
                np.zeros(len(outer_train)),
                labels[outer_train],
                groups[outer_train],
            ),
            start=1
        ):
            train_idx = outer_train[local_tr]
            valid_idx = outer_train[local_val]

            if set(groups[train_idx]) & set(groups[valid_idx]):
                raise ValueError("Inner connected-group leakage")

            if set(projects[train_idx]) & set(projects[valid_idx]):
                raise ValueError("Inner project leakage")

            train_labels = labels[train_idx]
            valid_labels = labels[valid_idx]

            if len(np.unique(train_labels)) < 2:
                raise ValueError("Insufficient training classes")

            vectorizer = phase4.make_features(
                "word+character"
            )

            train_lexical = vectorizer.fit_transform(
                texts[train_idx]
            )

            valid_lexical = vectorizer.transform(
                texts[valid_idx]
            )

            x_train = hstack(
                [
                    train_lexical,
                    csr_matrix(
                        outer_embeddings[local_tr]
                        * SEMANTIC_WEIGHT
                    )
                ],
                format="csr",
            )

            x_val = hstack(
                [
                    valid_lexical,
                    csr_matrix(
                        outer_embeddings[local_val]
                        * SEMANTIC_WEIGHT
                    )
                ],
                format="csr",
            )

            weights = tempered_weights(train_labels)

            # Check exact agreement with Phase 5
            # weighting whenever all classes exist.
            if set(train_labels) == set(LABELS):
                original_weights = (
                    phase5.calculate_weights(
                        train_labels, "tempered"
                    )
                )
                for label in LABELS:
                    if not np.isclose(
                        weights[label],
                        original_weights[label],
                    ):
                        raise ValueError(
                            "Phase 5 weight mismatch"
                        )

            classifier = LogisticRegression(
                C=10.0,
                class_weight=weights,
                max_iter=2000,
                random_state=SEED,
            )

            classifier.fit(
                x_train, train_labels
            )

            predictions = classifier.predict(x_val)

            scores = score_predictions(
                valid_labels,
                predictions,
            )

            row = {
                "Model": "HybridTempered",
                "OuterFold": outer,
                "InnerFold": inner,
                "TrainRecords": len(train_idx),
                "ValidationRecords": len(valid_idx),
                "TrainClasses": len(set(train_labels)),
                "ValidationClasses": len(set(valid_labels)),
                **scores,
            }

            results.append(row)

            for idx, actual, predicted in zip(
                valid_idx, valid_labels, predictions
            ):
                prediction_rows.append({
                    "OuterFold": outer,
                    "InnerFold": inner,
                    "RecordID": data.iloc[idx]["RecordID"],
                    "ProjectID": projects[idx],
                    "ActualClass": actual,
                    "PredictedClass": predicted,
                })

            for label in LABELS:
                support_rows.append({
                    "OuterFold": outer,
                    "InnerFold": inner,
                    "Class": label,
                    "TrainCount": int(
                        np.sum(train_labels == label)
                    ),
                    "ValidationCount": int(
                        np.sum(valid_labels == label)
                    ),
                })

            print(
                f"  Inner fold {inner}: "
                f"Train={len(train_idx)}, "
                f"Val={len(valid_idx)}, "
                f"Accuracy={scores['Accuracy']:.4f}, "
                f"MacroF1={scores['MacroF1']:.4f}",
                flush=True,
            )

            # Save after every completed fold.
            pd.DataFrame(results).to_csv(
                TABLES / "23_hybrid_inner_fold_metrics.csv",
                index=False,
            )

            pd.DataFrame(prediction_rows).to_csv(
                TABLES / "23_hybrid_inner_predictions.csv",
                index=False,
            )

        del outer_embeddings

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hybrid = pd.DataFrame(results)

    if (
        len(hybrid) != 9
        or set(zip(
            hybrid["OuterFold"],
            hybrid["InnerFold"]
        )) != expected_keys
    ):
        raise ValueError("Incomplete hybrid evaluation")

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(
        TABLES / "23_hybrid_inner_class_support.csv",
        index=False,
    )

    merged = hybrid.merge(
        mixed,
        on=["OuterFold", "InnerFold"],
        suffixes=("_Hybrid", "_Mixed"),
        validate="one_to_one",
    )

    for column in ("TrainRecords", "ValidationRecords"):
        if not (
            merged[f"{column}_Hybrid"]
            == merged[f"{column}_Mixed"]
        ).all():
            raise ValueError(
                f"Split mismatch: {column}"
            )

    paired = merged[[
        "OuterFold", "InnerFold",
        "Accuracy_Hybrid", "Accuracy_Mixed",
        "MacroF1_Hybrid", "MacroF1_Mixed",
    ]].copy()

    paired["MixedMinusHybridAccuracy"] = (
        paired["Accuracy_Mixed"]
        - paired["Accuracy_Hybrid"]
    )

    paired["MixedMinusHybridMacroF1"] = (
        paired["MacroF1_Mixed"]
        - paired["MacroF1_Hybrid"]
    )

    paired.to_csv(
        TABLES / "23_hybrid_vs_mixed_paired.csv",
        index=False,
    )

    comparison = pd.concat(
        [
            hybrid.assign(Model="HybridTempered"),
            mixed.assign(Model="Mixed75_25CE"),
        ],
        ignore_index=True,
    )

    summary = comparison.groupby(
        "Model", as_index=False
    ).agg(
        MeanAccuracy=("Accuracy", "mean"),
        StdAccuracy=("Accuracy", "std"),
        MeanMacroF1=("MacroF1", "mean"),
        StdMacroF1=("MacroF1", "std"),
        MeanRecallF=("Recall_F", "mean"),
        MeanRecallL=("Recall_L", "mean"),
        MeanRecallO=("Recall_O", "mean"),
        MeanRecallSC=("Recall_SC", "mean"),
        MeanRecallPO=("Recall_PO", "mean"),
    )

    summary.to_csv(
        TABLES / "23_hybrid_vs_mixed_summary.csv",
        index=False,
    )

    metadata = {
        "experiment": "Phase 8 Experiment 23B",
        "comparison": [
            "HybridTempered",
            "Mixed75_25CE"
        ],
        "inner_comparisons": 9,
        "hybrid_semantic_weight": SEMANTIC_WEIGHT,
        "hybrid_logistic_C": 10.0,
        "hybrid_max_iter": 2000,
        "hybrid_weighting": "sqrt-balanced normalized",
        "missing_training_class_rule":
            "Weight only classes present in training",
        "tfidf_fit_on_inner_training_only": True,
        "outer_validation_evaluated": False,
        "original_test_accessed": False,
        "original_labels_modified": False,
        "exploratory": True,
        "limitations": [
            "The inner splits have been repeatedly examined.",
            "The nine comparisons are not independent.",
            "The hybrid and mixed models use different architectures.",
            "Missing classes cannot be learned by either model.",
            "Saved mixed-model predictions are unavailable "
            "for example-level paired analysis.",
        ],
    }

    (
        REPORTS / "23_hybrid_vs_mixed_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 23B SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nPAIRED DIFFERENCES")
    print(paired.to_string(index=False))

    print("\nMixed macro-F1 wins:",
          int((paired[
              "MixedMinusHybridMacroF1"
          ] > 0).sum()),
          "of 9")

    print("Mixed accuracy wins:",
          int((paired[
              "MixedMinusHybridAccuracy"
          ] > 0).sum()),
          "of 9")

    print("\nOriginal test accessed: No")
    print("Outer validation evaluated: No")
    print("Original datasets modified: No")
    print("Experiment 23B completed.")


if __name__ == "__main__":
    main()
