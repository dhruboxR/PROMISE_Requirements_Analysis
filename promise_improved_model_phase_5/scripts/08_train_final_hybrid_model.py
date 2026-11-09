
"""PROMISE+ Phase 5, Experiment 08.

Train and save the selected hybrid classifier on all
3,127 development records.

Model:
- Word + character TF-IDF
- Frozen all-MiniLM-L6-v2 embeddings
- Semantic weight = 1.0
- Logistic Regression, C = 10
- Tempered class weighting

The original test partition is not accessed.
"""

from pathlib import Path
import hashlib
import importlib.util
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from transformers import AutoTokenizer, AutoModel


# -------------------------------------------------
# Paths and configuration
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

PHASE4 = THESIS / "promise_advanced_models_phase_4"

MODELS = ROOT / "outputs" / "models"
REPORTS = ROOT / "outputs" / "reports"
FOLD_MANIFEST = (
    ROOT / "outputs" / "tables"
    / "01_fixed_fold_assignments.csv"
)

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_WEIGHT = 1.0
C = 10.0
MAX_ITER = 2000
MAX_LENGTH = 128

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def import_module(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        name, path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def tempered_class_weights(y):
    """Reproduce Experiment 06 tempered weights."""

    counts = pd.Series(y).value_counts()

    if set(counts.index) != set(LABELS):
        raise ValueError("Missing or unexpected class")

    n = len(y)
    k = len(LABELS)

    balanced = {
        label: n / (k * int(counts[label]))
        for label in LABELS
    }

    raw = {
        label: np.sqrt(balanced[label])
        for label in LABELS
    }

    mean_weight = sum(
        int(counts[label]) * raw[label]
        for label in LABELS
    ) / n

    weights = {
        label: float(raw[label] / mean_weight)
        for label in LABELS
    }

    return weights, {
        label: int(counts[label])
        for label in LABELS
    }


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("\nPHASE 5: EXPERIMENT 08")
    print("TRAIN AND SAVE THE SELECTED HYBRID MODEL")
    print("=" * 65)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Reuse the exact data and feature definitions
    # from the original Phase 4 experiments.

    original = import_module(
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py",
        "phase4_cv",
    )

    semantic = import_module(
        PHASE4 / "scripts"
        / "17_semantic_embedding_logistic_cv.py",
        "phase4_semantic",
    )

    if original.SEED != SEED:
        raise ValueError("Seed mismatch")

    if original.LABELS != LABELS:
        raise ValueError("Label order mismatch")

    if semantic.MODEL_NAME != MODEL_NAME:
        raise ValueError("MiniLM model mismatch")

    if semantic.MAX_LENGTH != MAX_LENGTH:
        raise ValueError("Maximum token length mismatch")

    # -------------------------------------------------
    # Load development data
    # -------------------------------------------------

    data = original.load_development_data()

    if len(data) != 3127:
        raise ValueError(
            f"Unexpected record count: {len(data)}"
        )

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs")

    texts = (
        data["NormalizedText"]
        .astype(str)
        .to_numpy()
    )

    labels = (
        data["_class_"]
        .astype(str)
        .to_numpy()
    )

    groups = original.create_connected_groups(data)

    if len(set(groups)) != 66:
        raise ValueError("Connected group count mismatch")

    if set(labels) != set(LABELS):
        raise ValueError("Class labels mismatch")

    # Confirm that the records are exactly those
    # used for the previous Phase 5 CV experiments.

    if not FOLD_MANIFEST.exists():
        raise FileNotFoundError(FOLD_MANIFEST)

    manifest = pd.read_csv(
        FOLD_MANIFEST,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
    ):
        raise ValueError("Invalid fold manifest")

    expected = set(
        manifest["RecordID"].astype(str)
    )

    observed = set(
        data["RecordID"].astype(str)
    )

    if expected != observed:
        raise ValueError(
            "Development records differ from CV data"
        )

    reference = data[
        ["RecordID", "ProjectID", "_class_"]
    ].copy()

    reference["RecordID"] = (
        reference["RecordID"].astype(str)
    )
    reference["ProjectID"] = (
        reference["ProjectID"].astype(str)
    )

    checked = reference.merge(
        manifest[
            ["RecordID", "ProjectID", "ActualClass"]
        ],
        on="RecordID",
        suffixes=("_data", "_manifest"),
        validate="one_to_one",
    )

    if len(checked) != len(data):
        raise ValueError("Manifest matching failed")

    if not (
        checked["ProjectID_data"]
        == checked["ProjectID_manifest"]
    ).all():
        raise ValueError("ProjectID mismatch")

    if not (
        checked["_class_"]
        == checked["ActualClass"]
    ).all():
        raise ValueError("Class label mismatch")

    print("Development records:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Classes:", len(LABELS))
    print("Development manifest: VERIFIED")

    # -------------------------------------------------
    # Fit lexical features on the full development set
    # -------------------------------------------------

    print("\nFitting word + character TF-IDF...")

    vectorizer = original.make_features(
        "word+character"
    )

    x_lexical = vectorizer.fit_transform(texts)

    print("TF-IDF shape:", x_lexical.shape)

    # -------------------------------------------------
    # Encode semantic features
    # -------------------------------------------------

    semantic.set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)
    print("Loading frozen MiniLM...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    )

    encoder.to(device)
    encoder.eval()

    checkpoint_revision = getattr(
        encoder.config,
        "_commit_hash",
        None,
    )

    print("\nGenerating semantic embeddings...")

    embeddings = semantic.generate_embeddings(
        texts.tolist(),
        tokenizer,
        encoder,
        device,
    )

    if embeddings.shape != (3127, 384):
        raise ValueError(
            f"Unexpected embeddings: {embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid embedding values")

    print("MiniLM shape:", embeddings.shape)

    # -------------------------------------------------
    # Create hybrid features
    # -------------------------------------------------

    x_hybrid = hstack(
        [
            x_lexical,
            csr_matrix(
                embeddings * SEMANTIC_WEIGHT
            ),
        ],
        format="csr",
    )

    print("\nHybrid shape:", x_hybrid.shape)

    # -------------------------------------------------
    # Compute training class weights
    # -------------------------------------------------

    weights, class_counts = (
        tempered_class_weights(labels)
    )

    print("\nTempered class weights:")

    for label in LABELS:
        print(
            f"{label:<3} "
            f"count={class_counts[label]:<5} "
            f"weight={weights[label]:.6f}"
        )

    # -------------------------------------------------
    # Train classifier
    # -------------------------------------------------

    print("\nTraining Logistic Regression...")

    classifier = LogisticRegression(
        C=C,
        class_weight=weights,
        max_iter=MAX_ITER,
        random_state=SEED,
    )

    classifier.fit(x_hybrid, labels)

    check_is_fitted(classifier)

    if set(classifier.classes_) != set(LABELS):
        raise ValueError("Trained classes mismatch")

    print("Training completed.")
    print("Iterations:", classifier.n_iter_.tolist())

    # -------------------------------------------------
    # Save model artifacts
    # -------------------------------------------------

    vectorizer_path = (
        MODELS / "08_hybrid_tfidf_vectorizer.joblib"
    )

    classifier_path = (
        MODELS / "08_hybrid_classifier.joblib"
    )

    metadata_path = (
        MODELS / "08_hybrid_model_metadata.json"
    )

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(classifier, classifier_path)

    # Verify serialized components by loading them.
    # Only load joblib artifacts generated by trusted code.

    reloaded_vectorizer = joblib.load(
        vectorizer_path
    )

    reloaded_classifier = joblib.load(
        classifier_path
    )

    check_is_fitted(reloaded_classifier)

    sample_indices = np.array(
        [0, 1, 2, len(texts) - 1]
    )

    sample_lexical = (
        reloaded_vectorizer.transform(
            texts[sample_indices]
        )
    )

    sample_features = hstack(
        [
            sample_lexical,
            csr_matrix(
                embeddings[sample_indices]
                * SEMANTIC_WEIGHT
            ),
        ],
        format="csr",
    )

    original_predictions = classifier.predict(
        x_hybrid[sample_indices]
    )

    reloaded_predictions = (
        reloaded_classifier.predict(sample_features)
    )

    if not np.array_equal(
        original_predictions,
        reloaded_predictions,
    ):
        raise ValueError(
            "Saved artifacts changed predictions"
        )

    print("Serialization check: VERIFIED")

    # -------------------------------------------------
    # Save training metadata
    # -------------------------------------------------

    metadata = {
        "phase": 5,
        "experiment": 8,
        "model_status": "consolidated_development_model",
        "training_records": int(len(data)),
        "connected_groups": int(len(set(groups))),
        "random_seed": SEED,
        "class_labels": LABELS,
        "classifier_classes": (
            classifier.classes_.tolist()
        ),
        "class_counts": class_counts,
        "class_weights": weights,
        "lexical_representation": "word+character",
        "semantic_model": MODEL_NAME,
        "semantic_checkpoint_revision": checkpoint_revision,
        "semantic_embedding_dimensions": 384,
        "semantic_weight": SEMANTIC_WEIGHT,
        "max_token_length": MAX_LENGTH,
        "classifier": "LogisticRegression",
        "C": C,
        "max_iter": MAX_ITER,
        "class_weight_mode": "tempered",
        "text_column": "NormalizedText",
        "normalization_note": (
            "Inference must reproduce the exact Phase 2 "
            "normalization used to create NormalizedText."
        ),
        "vectorizer_file": vectorizer_path.name,
        "classifier_file": classifier_path.name,
        "vectorizer_sha256": file_sha256(
            vectorizer_path
        ),
        "classifier_sha256": file_sha256(
            classifier_path
        ),
        "source_cv_script_sha256": file_sha256(
            PHASE4 / "scripts"
            / "01_group_aware_feature_comparison.py"
        ),
        "source_semantic_script_sha256": file_sha256(
            PHASE4 / "scripts"
            / "17_semantic_embedding_logistic_cv.py"
        ),
        "development_manifest_sha256": file_sha256(
            FOLD_MANIFEST
        ),
        "model_selection_metric": (
            "mean_3_fold_macro_f1"
        ),
        "previous_mean_cv_macro_f1": 0.3200,
        "test_partition_accessed": False,
        "independent_evaluation_completed": False,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved model files:")
    print(vectorizer_path)
    print(classifier_path)
    print(metadata_path)

    print("\nEXPERIMENT 08 COMPLETED")
    print("Original test partition was not accessed.")
    print(
        "No independent evaluation was performed."
    )


if __name__ == "__main__":
    main()
