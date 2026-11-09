
"""Phase 5 Experiment 10: Model verification.

Checks:
1. Saved artifact checksums.
2. Model metadata and feature dimensions.
3. Phase 2/Phase 5 normalization agreement.
4. Repeated prediction consistency.
5. Raw-text and normalized-text prediction agreement.
6. Saves a reproducibility report.

No training or evaluation dataset is loaded.
"""

import ast
import hashlib
import importlib.util
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

MODELS = ROOT / "outputs" / "models"
REPORTS = ROOT / "outputs" / "reports"

METADATA_PATH = (
    MODELS / "08_hybrid_model_metadata.json"
)

PREDICT_SCRIPT = (
    ROOT / "scripts" / "09_predict_requirement.py"
)

NORMALIZATION_SCRIPT = (
    THESIS / "promise_preprocessing_phase_2"
    / "scripts" / "02_normalize_text.py"
)

EXPECTED_LABELS = {
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_script(path, module_name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name, path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_original_normalizer():
    """Extract original Phase 2 normalization function."""

    source = NORMALIZATION_SCRIPT.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(source)

    matching = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_text"
    ]

    if len(matching) != 1:
        raise ValueError(
            "Expected one Phase 2 normalize_text function"
        )

    isolated_tree = ast.Module(
        body=[matching[0]],
        type_ignores=[],
    )

    namespace = {
        "re": re,
        "unicodedata": unicodedata,
    }

    # Execute only the normalization function from
    # this trusted local Phase 2 research script.
    exec(
        compile(
            isolated_tree,
            str(NORMALIZATION_SCRIPT),
            "exec",
        ),
        namespace,
    )

    return namespace["normalize_text"]


def check_prediction(result):
    if result["predicted_label"] not in EXPECTED_LABELS:
        raise ValueError("Unknown predicted category")

    ranked = result["top_predictions"]

    if len(ranked) != 3:
        raise ValueError("Expected three ranked categories")

    probabilities = [
        item["probability"]
        for item in ranked
    ]

    if not all(
        np.isfinite(value)
        and 0 <= value <= 1
        for value in probabilities
    ):
        raise ValueError("Invalid model probabilities")

    if any(
        probabilities[i] < probabilities[i + 1]
        for i in range(len(probabilities) - 1)
    ):
        raise ValueError("Probabilities are not ranked")

    if ranked[0]["label"] != result["predicted_label"]:
        raise ValueError("Prediction/ranking mismatch")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("\nPHASE 5: EXPERIMENT 10")
    print("FINAL HYBRID MODEL VERIFICATION")
    print("=" * 65)

    # ------------------------------------------
    # 1. Artifact integrity
    # ------------------------------------------

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    vectorizer_path = (
        MODELS / metadata["vectorizer_file"]
    )

    classifier_path = (
        MODELS / metadata["classifier_file"]
    )

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
        if sha256(path) != expected_hash:
            raise ValueError(
                f"Checksum mismatch: {path.name}"
            )

    print("Artifact checksum verification: PASSED")

    # Only load trusted artifacts generated locally
    # by Experiment 08.
    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    # ------------------------------------------
    # 2. Class and feature dimensions
    # ------------------------------------------

    if set(classifier.classes_) != EXPECTED_LABELS:
        raise ValueError("Class label mismatch")

    if classifier.classes_.tolist() != metadata[
        "classifier_classes"
    ]:
        raise ValueError("Classifier class order mismatch")

    if metadata["training_records"] != 3127:
        raise ValueError("Training record mismatch")

    if metadata["class_weight_mode"] != "tempered":
        raise ValueError("Weighting mismatch")

    if metadata["semantic_weight"] != 1.0:
        raise ValueError("Semantic weight mismatch")

    lexical_dimensions = vectorizer.transform(
        ["verification input"]
    ).shape[1]

    semantic_dimensions = metadata[
        "semantic_embedding_dimensions"
    ]

    total_dimensions = (
        lexical_dimensions + semantic_dimensions
    )

    if lexical_dimensions != 48609:
        raise ValueError("TF-IDF dimension mismatch")

    if semantic_dimensions != 384:
        raise ValueError("MiniLM dimension mismatch")

    if classifier.n_features_in_ != total_dimensions:
        raise ValueError("Classifier dimension mismatch")

    print("Class labels: PASSED")
    print("Lexical dimensions:", lexical_dimensions)
    print("Semantic dimensions:", semantic_dimensions)
    print("Hybrid dimensions:", total_dimensions)
    print("Feature dimension verification: PASSED")

    # ------------------------------------------
    # 3. Normalization equivalence
    # ------------------------------------------

    inference = load_script(
        PREDICT_SCRIPT,
        "phase5_prediction",
    )

    phase2_normalize = load_original_normalizer()

    inputs = [
        "  The   system shall work.  ",
        "The system\nshall\trespond.",
        "Cafe\u0301",
        "THE system shall respond.",
        "The system must support 99.9% availability.",
    ]

    for raw in inputs:
        original = phase2_normalize(raw)
        current = inference.normalize_text(raw)

        if original != current:
            raise ValueError("Normalization mismatch")

    print("Phase 2 normalization comparison: PASSED")

    # ------------------------------------------
    # 4. Repeated inference consistency
    # ------------------------------------------

    text = (
        "The system shall allow users "
        "to reset their passwords."
    )

    print("\nRunning repeated inference checks...")

    first = inference.predict_requirement(
        text, top_k=3
    )

    second = inference.predict_requirement(
        text, top_k=3
    )

    check_prediction(first)
    check_prediction(second)

    if first["predicted_label"] != second["predicted_label"]:
        raise ValueError(
            "Repeated predictions do not match"
        )

    first_probs = np.array([
        x["probability"]
        for x in first["top_predictions"]
    ])

    second_probs = np.array([
        x["probability"]
        for x in second["top_predictions"]
    ])

    first_labels = [
        x["label"] for x in first["top_predictions"]
    ]

    second_labels = [
        x["label"] for x in second["top_predictions"]
    ]

    if first_labels != second_labels:
        raise ValueError(
            "Repeated category ranking mismatch"
        )

    if not np.allclose(
        first_probs,
        second_probs,
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError(
            "Repeated probabilities differ"
        )

    print("Repeated prediction consistency: PASSED")

    # ------------------------------------------
    # 5. Whitespace input consistency
    # ------------------------------------------

    raw_variant = (
        "  The   system shall allow users "
        "to reset their passwords.  "
    )

    variant = inference.predict_requirement(
        raw_variant, top_k=3
    )

    check_prediction(variant)

    if (
        variant["normalized_text"]
        != first["normalized_text"]
    ):
        raise ValueError(
            "Normalized inputs do not match"
        )

    if (
        variant["predicted_label"]
        != first["predicted_label"]
    ):
        raise ValueError(
            "Equivalent inputs changed prediction"
        )

    variant_probs = np.array([
        x["probability"]
        for x in variant["top_predictions"]
    ])

    variant_labels = [
        x["label"]
        for x in variant["top_predictions"]
    ]

    if variant_labels != first_labels:
        raise ValueError(
            "Equivalent inputs changed ranking"
        )

    if not np.allclose(
        variant_probs,
        first_probs,
        rtol=0,
        atol=1e-4,
    ):
        raise ValueError(
            "Equivalent inputs changed probabilities"
        )

    print("Equivalent input consistency: PASSED")

    # ------------------------------------------
    # 6. Save verification report
    # ------------------------------------------

    report = {
        "phase": 5,
        "experiment": 10,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "checks": {
            "artifact_checksums": "passed",
            "class_labels": "passed",
            "feature_dimensions": "passed",
            "phase2_normalization": "passed",
            "repeated_inference": "passed",
            "equivalent_raw_inputs": "passed",
        },
        "training_records": 3127,
        "lexical_dimensions": lexical_dimensions,
        "semantic_dimensions": semantic_dimensions,
        "hybrid_dimensions": total_dimensions,
        "predicted_category": first["predicted_label"],
        "top_predictions": first["top_predictions"],
        "previous_mean_cv_macro_f1": 0.3200,
        "independent_test_performed": False,
        "test_partition_accessed": False,
        "note": (
            "Technical verification only. "
            "Does not establish model generalization."
        ),
    }

    output_path = (
        REPORTS / "10_model_verification.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)
    print("\nEXPERIMENT 10 COMPLETED")
    print("All verification checks passed.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
