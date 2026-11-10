
"""
Phase 8, Experiment 01
Hierarchical Hybrid Requirements Classification

Compare:
1. Original tempered 12-class baseline
2. Hard hierarchical classification
3. Soft hierarchical classification

Uses Phase 4 feature construction and MiniLM encoding.
Uses the original Phase 5 fixed project-aware folds.

No test data is accessed.
"""

from pathlib import Path
import importlib.util
import json
import random

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

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

PHASE4 = THESIS / "promise_advanced_models_phase_4"
PHASE5 = THESIS / "promise_improved_model_phase_5"

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

FOLD_FILE = (
    PHASE5 / "outputs/tables/01_fixed_fold_assignments.csv"
)

SEED = 42
C = 10.0
SEMANTIC_WEIGHT = 1.0

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO",
]

NF_LABELS = [label for label in LABELS if label != "F"]


def load_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tempered_weights(labels, class_names):
    labels = np.asarray(labels)
    counts = pd.Series(labels).value_counts()

    if set(counts.index) != set(class_names):
        raise ValueError(
            "Training fold does not contain all required classes"
        )

    n = len(labels)
    k = len(class_names)

    raw = {
        label: np.sqrt(n / (k * counts[label]))
        for label in class_names
    }

    average = sum(
        counts[label] * raw[label]
        for label in class_names
    ) / n

    return {
        label: float(raw[label] / average)
        for label in class_names
    }


def make_classifier(labels, class_names):
    weights = tempered_weights(
        labels, class_names
    )

    return LogisticRegression(
        C=C,
        class_weight=weights,
        max_iter=2000,
        random_state=SEED,
    )


def calculate_metrics(actual, predicted):
    result = {
        "Accuracy": accuracy_score(
            actual, predicted
        ),
        "MacroF1": f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "BalancedAccuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "WeightedF1": f1_score(
            actual,
            predicted,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        ),
    }

    recalls = recall_score(
        actual,
        predicted,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    for label in ["F", "O", "SC", "PO"]:
        result[f"Recall_{label}"] = recalls[
            LABELS.index(label)
        ]

    return {
        key: float(value)
        for key, value in result.items()
    }


def verify_data(data, manifest, groups):
    required = {
        "RecordID",
        "ProjectID",
        "NormalizedText",
        "_class_",
    }

    if not required.issubset(data.columns):
        raise ValueError("Development columns missing")

    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate development RecordID")

    if len(set(groups)) != 66:
        raise ValueError("Unexpected connected-group count")

    if len(manifest) != len(data):
        raise ValueError("Fold manifest size mismatch")

    if manifest["RecordID"].duplicated().any():
        raise ValueError("Duplicate manifest RecordID")

    if set(data["RecordID"]) != set(
        manifest["RecordID"]
    ):
        raise ValueError("Development and manifest IDs differ")

    aligned = (
        manifest.set_index("RecordID")
        .loc[data["RecordID"]]
        .reset_index()
    )

    if not np.array_equal(
        data["ProjectID"].astype(str).to_numpy(),
        aligned["ProjectID"].astype(str).to_numpy(),
    ):
        raise ValueError("ProjectID mismatch")

    if not np.array_equal(
        data["_class_"].astype(str).to_numpy(),
        aligned["ActualClass"].astype(str).to_numpy(),
    ):
        raise ValueError("Class mismatch")

    fold_ids = aligned["Fold"].astype(int).to_numpy()

    if set(fold_ids) != {1, 2, 3}:
        raise ValueError("Unexpected folds")

    group_to_folds = pd.DataFrame({
        "Group": groups,
        "Fold": fold_ids,
    }).groupby("Group")["Fold"].nunique()

    if (group_to_folds > 1).any():
        raise ValueError(
            "Connected-group leakage detected"
        )

    return fold_ids


def predict_hierarchy(
    binary_model,
    nf_model,
    x_val,
):
    binary_prob = binary_model.predict_proba(x_val)
    nf_prob = nf_model.predict_proba(x_val)

    binary_classes = list(binary_model.classes_)
    nf_classes = list(nf_model.classes_)

    p_f = binary_prob[
        :, binary_classes.index("F")
    ]
    p_nf = binary_prob[
        :, binary_classes.index("NF")
    ]

    # Hard routing:
    # Choose functional or non-functional first.
    binary_prediction = binary_model.predict(x_val)
    nf_prediction = nf_model.predict(x_val)

    hard_prediction = np.where(
        binary_prediction == "F",
        "F",
        nf_prediction,
    )

    # Soft routing:
    # P(F) comes from the binary model.
    # P(class_j) = P(NF) * P(class_j | NF).
    scores = np.zeros(
        (x_val.shape[0], len(LABELS)),
        dtype=np.float64,
    )

    scores[:, LABELS.index("F")] = p_f

    for j, label in enumerate(nf_classes):
        scores[:, LABELS.index(label)] = (
            p_nf * nf_prob[:, j]
        )

    if not np.allclose(
        scores.sum(axis=1), 1.0, atol=1e-5
    ):
        raise ValueError(
            "Hierarchical probabilities do not sum to 1"
        )

    soft_prediction = np.asarray(LABELS)[
        scores.argmax(axis=1)
    ]

    return hard_prediction, soft_prediction


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

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

    if phase4_cv.LABELS != LABELS:
        raise ValueError("Phase 4 label order mismatch")

    if phase4_cv.SEED != SEED:
        raise ValueError("Phase 4 seed mismatch")

    if phase4_semantic.MODEL_NAME != MODEL_NAME:
        raise ValueError("MiniLM model mismatch")

    if phase4_semantic.MAX_LENGTH != 128:
        raise ValueError("Token length mismatch")

    data = phase4_cv.load_development_data()

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    texts = data["NormalizedText"].astype(str).to_numpy()
    labels = data["_class_"].astype(str).to_numpy()

    groups = np.asarray(
        phase4_cv.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        FOLD_FILE,
        dtype={
            "RecordID": str,
            "ProjectID": str,
        },
        keep_default_na=False,
    )

    fold_ids = verify_data(
        data, manifest, groups
    )

    print("\nPHASE 8 EXPERIMENT 01")
    print("=" * 65)
    print("Development records:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Fixed fold assignments: VERIFIED")
    print("Original test data: NOT ACCESSED")

    from transformers import (
        AutoModel,
        AutoTokenizer,
    )

    phase4_semantic.set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)
    print("Loading MiniLM:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    encoder.eval()

    print("Generating frozen embeddings...")

    embeddings = phase4_semantic.generate_embeddings(
        texts.tolist(),
        tokenizer,
        encoder,
        device,
    )

    if embeddings.shape != (3127, 384):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite embeddings")

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics_rows = []
    prediction_rows = []

    for fold in [1, 2, 3]:
        print(f"\nFOLD {fold}")
        print("-" * 65)

        train_idx = np.where(
            fold_ids != fold
        )[0]

        val_idx = np.where(
            fold_ids == fold
        )[0]

        if set(groups[train_idx]) & set(
            groups[val_idx]
        ):
            raise ValueError(
                "Group leakage"
            )

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
                    embeddings[train_idx]
                    * SEMANTIC_WEIGHT
                ),
            ],
            format="csr",
        )

        x_val = hstack(
            [
                val_lexical,
                csr_matrix(
                    embeddings[val_idx]
                    * SEMANTIC_WEIGHT
                ),
            ],
            format="csr",
        )

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Model 1: Original tempered baseline.
        baseline = make_classifier(
            y_train, LABELS
        )
        baseline.fit(x_train, y_train)

        baseline_prediction = baseline.predict(
            x_val
        )

        # Model 2: Binary F / NF classifier.
        y_binary = np.where(
            y_train == "F", "F", "NF"
        )

        binary_model = make_classifier(
            y_binary, ["F", "NF"]
        )
        binary_model.fit(
            x_train, y_binary
        )

        # Model 3: Specialized non-functional model.
        nf_mask = y_train != "F"

        nf_model = make_classifier(
            y_train[nf_mask],
            NF_LABELS,
        )

        nf_model.fit(
            x_train[nf_mask],
            y_train[nf_mask],
        )

        hard_prediction, soft_prediction = (
            predict_hierarchy(
                binary_model,
                nf_model,
                x_val,
            )
        )

        predictions = {
            "BaselineTempered": baseline_prediction,
            "HardHierarchy": hard_prediction,
            "SoftHierarchy": soft_prediction,
        }

        for model_name, predicted in predictions.items():
            metrics = calculate_metrics(
                y_val, predicted
            )

            metrics_rows.append({
                "Model": model_name,
                "Fold": fold,
                "TrainRecords": len(train_idx),
                "ValidationRecords": len(val_idx),
                **metrics,
            })

            for index, predicted_label in zip(
                val_idx, predicted
            ):
                prediction_rows.append({
                    "RecordID": data.iloc[index]["RecordID"],
                    "Fold": fold,
                    "Model": model_name,
                    "ActualClass": labels[index],
                    "PredictedClass": predicted_label,
                })

            print(
                f"{model_name:20s} "
                f"Accuracy={metrics['Accuracy']:.4f} "
                f"MacroF1={metrics['MacroF1']:.4f} "
                f"SC={metrics['Recall_SC']:.4f} "
                f"PO={metrics['Recall_PO']:.4f}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    summary = (
        metrics_df.groupby("Model", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanMacroF1=("MacroF1", "mean"),
            MeanBalancedAccuracy=(
                "BalancedAccuracy", "mean"
            ),
            MeanWeightedF1=("WeightedF1", "mean"),
            MeanRecallF=("Recall_F", "mean"),
            MeanRecallO=("Recall_O", "mean"),
            MeanRecallSC=("Recall_SC", "mean"),
            MeanRecallPO=("Recall_PO", "mean"),
        )
    )

    expected_baseline = {
        "MeanAccuracy": 0.6550551723865207,
        "MeanMacroF1": 0.32003390467151266,
    }

    baseline_row = summary.loc[
        summary["Model"] == "BaselineTempered"
    ].iloc[0]

    baseline_reproduced = all(
        abs(float(baseline_row[name]) - target) < 1e-5
        for name, target in expected_baseline.items()
    )

    metrics_df.to_csv(
        TABLES / "01_hierarchical_fold_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        TABLES / "01_hierarchical_oof_predictions.csv",
        index=False,
    )

    summary.to_csv(
        TABLES / "01_hierarchical_summary.csv",
        index=False,
    )

    metadata = {
        "experiment": "Phase 8 Experiment 01",
        "dataset": "PROMISE+",
        "development_records": 3127,
        "folds": 3,
        "feature_representation": (
            "word+character TF-IDF + frozen MiniLM"
        ),
        "embedding_model": MODEL_NAME,
        "classifier": "Logistic Regression",
        "C": C,
        "semantic_weight": SEMANTIC_WEIGHT,
        "weighting": (
            "Tempered; computed separately for "
            "12-class, binary and 11-class models"
        ),
        "baseline_reproduced": bool(
            baseline_reproduced
        ),
        "original_test_accessed": False,
        "source_aware_training_additions": 0,
        "development_results_exploratory": True,
    }

    (
        REPORTS / "01_hierarchical_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nFINAL RESULTS")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\nBaseline reproduction:", (
        "PASS" if baseline_reproduced else "FAILED"
    ))

    print("Original test accessed: No")
    print("Original datasets modified: No")
    print("\nExperiment 01 completed.")


if __name__ == "__main__":
    main()
