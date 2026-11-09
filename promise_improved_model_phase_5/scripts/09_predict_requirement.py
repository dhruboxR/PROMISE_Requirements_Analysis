
"""Phase 5 Experiment 09: Hybrid requirement prediction.

Accepts raw requirement text and applies the same
conservative normalization used in Phase 2.

Loads trusted Experiment 08 model artifacts.
Does not access training, validation, or test datasets.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import torch

from scipy.sparse import csr_matrix, hstack
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
PHASE4 = ROOT.parent / "promise_advanced_models_phase_4"
MODELS = ROOT / "outputs" / "models"

METADATA_FILE = MODELS / "08_hybrid_model_metadata.json"


def normalize_text(value):
    """Exact Phase 2 normalization operations."""
    text = str(value)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def import_semantic_module():
    path = (
        PHASE4 / "scripts"
        / "17_semantic_embedding_logistic_cv.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "phase4_semantic_inference", path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import MiniLM encoder")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def predict_requirement(raw_text, top_k=3):
    normalized_text = normalize_text(raw_text)

    if not normalized_text:
        raise ValueError(
            "Requirement text is empty after normalization"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Experiment 08 metadata not found"
        )

    metadata = json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )

    vectorizer_path = MODELS / metadata["vectorizer_file"]
    classifier_path = MODELS / metadata["classifier_file"]

    # Confirm that saved artifacts have not changed.
    for path, expected_hash in [
        (
            vectorizer_path,
            metadata["vectorizer_sha256"],
        ),
        (
            classifier_path,
            metadata["classifier_sha256"],
        ),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

        if sha256(path) != expected_hash:
            raise ValueError(
                f"Artifact checksum mismatch: {path.name}"
            )

    # Only deserialize trusted, locally generated joblib files.
    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    if classifier.classes_.tolist() != sorted(
        metadata["class_labels"]
    ):
        raise ValueError("Classifier class order mismatch")

    semantic = import_semantic_module()

    model_name = metadata["semantic_model"]

    if semantic.MODEL_NAME != model_name:
        raise ValueError("Semantic model mismatch")

    if semantic.MAX_LENGTH != metadata["max_token_length"]:
        raise ValueError("Tokenizer length mismatch")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)

    encoder.to(device)
    encoder.eval()

    embeddings = semantic.generate_embeddings(
        [normalized_text],
        tokenizer,
        encoder,
        device,
    )

    expected_shape = (
        1,
        metadata["semantic_embedding_dimensions"],
    )

    if embeddings.shape != expected_shape:
        raise ValueError("Unexpected embedding shape")

    if not np.isfinite(embeddings).all():
        raise ValueError("Invalid semantic embedding")

    lexical = vectorizer.transform([normalized_text])

    hybrid = hstack(
        [
            lexical,
            csr_matrix(
                embeddings * metadata["semantic_weight"]
            ),
        ],
        format="csr",
    )

    if hybrid.shape[1] != classifier.n_features_in_:
        raise ValueError("Hybrid feature dimension mismatch")

    predicted_label = classifier.predict(hybrid)[0]
    probabilities = classifier.predict_proba(hybrid)[0]

    top_k = max(
        1,
        min(int(top_k), len(classifier.classes_))
    )

    ranking = np.argsort(probabilities)[::-1][:top_k]

    top_predictions = [
        {
            "label": str(classifier.classes_[index]),
            "probability": float(probabilities[index]),
        }
        for index in ranking
    ]

    return {
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "predicted_label": str(predicted_label),
        "top_predictions": top_predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict a PROMISE+ requirement class"
    )

    parser.add_argument(
        "--text",
        required=True,
        help="Raw software requirement text",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of ranked categories to display",
    )

    args = parser.parse_args()

    result = predict_requirement(
        args.text, args.top_k
    )

    print("\nPHASE 5: EXPERIMENT 09")
    print("HYBRID REQUIREMENT PREDICTION")
    print("=" * 55)

    print("Raw requirement:", result["raw_text"])
    print(
        "Normalized requirement:",
        result["normalized_text"]
    )

    print(
        "\nPredicted category:",
        result["predicted_label"]
    )

    print("\nHighest model probabilities:")

    for item in result["top_predictions"]:
        print(
            f"{item['label']}: "
            f"{item['probability']:.4f}"
        )

    print(
        "\nNote: Probabilities are not independently "
        "calibrated confidence estimates."
    )


if __name__ == "__main__":
    main()
