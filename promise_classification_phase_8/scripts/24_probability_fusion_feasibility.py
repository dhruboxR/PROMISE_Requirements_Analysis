
"""
Phase 8 Experiment 24A
Matched probability-fusion feasibility.

Outer partition: 1
Inner fold: 1

Recreates:
- Phase 5 hybrid with tempered class weights
- Phase 8 Experiment 22 mixed 75/25 MiniLM

Fixed fusion:
75% hybrid + 25% mixed MiniLM

Only inner-training and validation data are used.
No original test or outer validation evaluation.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

P4_PATH = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

SEMANTIC_PATH = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/17_semantic_embedding_logistic_cv.py"
)

EXP19_PATH = (
    ROOT / "scripts/19_cross_project_contrastive_feasibility.py"
)

EXP21_PATH = (
    ROOT / "scripts/21_multisplit_sampling_evaluation.py"
)

EXP22_PATH = (
    ROOT / "scripts/22_mixed_class_aware_sampling.py"
)

EXP23_PATH = (
    ROOT / "scripts/23_matched_hybrid_evaluation.py"
)

MANIFEST = (
    THESIS / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

SEED = 42
OUTER = 1
INNER = 1
EPOCHS = 4
ALPHA = 0.75

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def import_script(path, name):
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(actual, predicted):
    result = {
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
    return result


def verify_previous(name, scores, previous):
    matched = previous[
        (previous["OuterFold"].astype(int) == OUTER)
        & (previous["InnerFold"].astype(int) == INNER)
        & (previous["Model"] == name)
    ]

    if len(matched) != 1:
        raise ValueError(
            f"Missing previous metric for {name}"
        )

    reference = matched.iloc[0]

    print(f"\nReproduction audit: {name}")

    for metric in ("Accuracy", "MacroF1"):
        old = float(reference[metric])
        new = float(scores[metric])

        difference = abs(new - old)

        print(
            f"  {metric}: "
            f"previous={old:.6f}, "
            f"reproduced={new:.6f}, "
            f"abs_difference={difference:.6f}"
        )

        if difference > 0.005:
            raise RuntimeError(
                f"{name} {metric} differs by more "
                "than 0.005. Check reproducibility "
                "before interpreting fusion."
            )


def predict_mixed_probabilities(model, loader, device):
    model.eval()
    all_probabilities = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in ("labels", "projects")
            }

            logits, _ = model(**inputs)

            probabilities = torch.softmax(
                logits.float(), dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    return np.vstack(all_probabilities)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    p4 = import_script(P4_PATH, "p4_for24")
    semantic = import_script(
        SEMANTIC_PATH, "semantic_for24"
    )
    exp19 = import_script(EXP19_PATH, "exp19_for24")
    exp21 = import_script(EXP21_PATH, "exp21_for24")
    exp22 = import_script(EXP22_PATH, "exp22_for24")
    exp23 = import_script(EXP23_PATH, "exp23_for24")

    if (
        p4.LABELS != LABELS
        or exp19.LABELS != LABELS
        or exp21.LABELS != LABELS
        or exp22.LABELS != LABELS
        or exp23.LABELS != LABELS
        or exp19.SEED != SEED
        or exp19.EPOCHS != EPOCHS
        or exp21.EPOCHS != EPOCHS
        or exp22.EPOCHS != EPOCHS
        or exp19.MODEL_NAME != MODEL_NAME
        or exp23.MODEL_NAME != MODEL_NAME
        or exp23.SEMANTIC_WEIGHT != 1.0
    ):
        raise ValueError("Experiment configuration mismatch")

    previous_path = (
        TABLES / "22_three_strategy_fold_metrics.csv"
    )
    hybrid_previous_path = (
        TABLES / "23_hybrid_inner_fold_metrics.csv"
    )

    previous_mixed = pd.read_csv(previous_path)
    previous_hybrid = pd.read_csv(hybrid_previous_path)

    data = p4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development data")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID")

    labels = data["_class_"].astype(str).to_numpy()
    texts = data["NormalizedText"].astype(str).to_numpy()
    projects = data["ProjectID"].to_numpy()

    if set(labels) != set(LABELS):
        raise ValueError("Unexpected classes")

    groups = np.asarray(
        p4.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"])
           != set(data["RecordID"])
    ):
        raise ValueError("Invalid fold manifest")

    aligned = manifest.set_index("RecordID").loc[
        data["RecordID"]
    ]

    if not np.array_equal(
        aligned["ActualClass"].astype(str).to_numpy(),
        labels,
    ):
        raise ValueError("Label mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        projects,
    ):
        raise ValueError("Project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Unexpected outer folds")

    outer_training = np.flatnonzero(folds != OUTER)

    inner_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    chosen = None

    for index, (tr, va) in enumerate(
        inner_cv.split(
            np.zeros(len(outer_training)),
            labels[outer_training],
            groups[outer_training],
        ),
        start=1,
    ):
        if index == INNER:
            chosen = (outer_training[tr], outer_training[va])
            break

    if chosen is None:
        raise ValueError("Inner fold not found")

    train_idx, valid_idx = chosen

    if set(groups[train_idx]) & set(groups[valid_idx]):
        raise ValueError("Connected-group leakage")

    if set(projects[train_idx]) & set(projects[valid_idx]):
        raise ValueError("Project leakage")

    if len(train_idx) != 1369 or len(valid_idx) != 790:
        raise ValueError("Split mismatch")

    y_train = labels[train_idx]
    y_valid = labels[valid_idx]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nPHASE 8 EXPERIMENT 24A")
    print("=" * 70)
    print("Device:", device)
    print("Outer training partition:", OUTER)
    print("Inner validation fold:", INNER)
    print("Training records:", len(train_idx))
    print("Validation records:", len(valid_idx))
    print("Fusion alpha:", ALPHA)
    print("Original test accessed: No")
    print("Outer validation evaluated: No")

    exp19.set_seed()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------------------
    # MODEL 1: Frozen hybrid
    # ---------------------------------------
    print("\nTraining hybrid model...", flush=True)

    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)
    encoder.eval()

    # Only current inner train and validation texts.
    selected_idx = np.concatenate(
        [train_idx, valid_idx]
    )

    semantic_vectors = semantic.generate_embeddings(
        texts[selected_idx].tolist(),
        tokenizer,
        encoder,
        device,
    )

    if semantic_vectors.shape != (
        len(selected_idx), 384
    ):
        raise ValueError("Unexpected embedding dimensions")

    if not np.isfinite(semantic_vectors).all():
        raise ValueError("Invalid embeddings")

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vectorizer = p4.make_features(
        "word+character"
    )

    lexical_train = vectorizer.fit_transform(
        texts[train_idx]
    )

    lexical_valid = vectorizer.transform(
        texts[valid_idx]
    )

    n_train = len(train_idx)

    x_train = hstack(
        [
            lexical_train,
            csr_matrix(
                semantic_vectors[:n_train]
            ),
        ],
        format="csr",
    )

    x_valid = hstack(
        [
            lexical_valid,
            csr_matrix(
                semantic_vectors[n_train:]
            ),
        ],
        format="csr",
    )

    classifier = LogisticRegression(
        C=10.0,
        class_weight=exp23.tempered_weights(y_train),
        max_iter=2000,
        random_state=SEED,
    )

    classifier.fit(x_train, y_train)

    # Logistic regression returns classes in
    # classifier.classes_ order. Align to LABELS.
    raw_hybrid = classifier.predict_proba(x_valid)

    hybrid_prob = np.zeros(
        (len(valid_idx), len(LABELS)),
        dtype=np.float64,
    )

    for col, label in enumerate(classifier.classes_):
        hybrid_prob[:, LABELS.index(label)] = (
            raw_hybrid[:, col]
        )

    hybrid_pred = np.asarray(LABELS)[
        hybrid_prob.argmax(axis=1)
    ]

    hybrid_metrics = metrics(y_valid, hybrid_pred)

    verify_previous(
        "HybridTempered",
        hybrid_metrics,
        previous_hybrid,
    )

    del x_train, x_valid
    del lexical_train, lexical_valid
    del semantic_vectors, classifier

    # ---------------------------------------
    # MODEL 2: Mixed 75/25 fine-tuned MiniLM
    # ---------------------------------------
    print("\nTraining mixed MiniLM...", flush=True)

    exp19.set_seed()

    encoded = tokenizer(
        texts[selected_idx].tolist(),
        truncation=True,
        padding="max_length",
        max_length=exp19.MAX_LENGTH,
        return_tensors="pt",
    )

    project_codes, _ = pd.factorize(
        projects[selected_idx],
        sort=True,
    )

    y_train_ids = np.asarray(
        [LABELS.index(x) for x in y_train],
        dtype=int,
    )

    y_valid_ids = np.asarray(
        [LABELS.index(x) for x in y_valid],
        dtype=int,
    )

    training_data = exp19.RequirementDataset(
        {
            key: value[:n_train]
            for key, value in encoded.items()
        },
        y_train_ids,
        project_codes[:n_train],
    )

    validation_data = exp19.RequirementDataset(
        {
            key: value[n_train:]
            for key, value in encoded.items()
        },
        y_valid_ids,
        project_codes[n_train:],
    )

    sampler = exp22.MixedBatchSampler(
        y_train_ids,
        project_codes[:n_train],
        seed=SEED + OUTER * 100 + INNER,
    )

    training_loader = DataLoader(
        training_data,
        batch_sampler=sampler,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=exp19.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    exp19.set_seed()

    model = exp19.MiniLMContrastiveClassifier().to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=exp21.make_weights(
            y_train_ids
        ).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp19.LR,
        weight_decay=0.01,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in training_loader:
            actual_ids = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in ("labels", "projects")
            }

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(**inputs)
            loss = criterion(logits, actual_ids)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += float(loss.item())

        print(
            f"  Mixed epoch {epoch}: "
            f"mean CE loss="
            f"{total_loss / len(training_loader):.4f}",
            flush=True,
        )

    mixed_prob = predict_mixed_probabilities(
        model, validation_loader, device
    )

    mixed_pred = np.asarray(LABELS)[
        mixed_prob.argmax(axis=1)
    ]

    mixed_metrics = metrics(y_valid, mixed_pred)

    verify_previous(
        "Mixed75_25CE",
        mixed_metrics,
        previous_mixed,
    )

    del model, optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------
    # FIXED PROBABILITY FUSION
    # ---------------------------------------
    if hybrid_prob.shape != mixed_prob.shape:
        raise ValueError("Probability shape mismatch")

    for name, probabilities in [
        ("Hybrid", hybrid_prob),
        ("Mixed", mixed_prob),
    ]:
        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Non-finite {name} probabilities"
            )

        if not np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                f"{name} probability normalization error"
            )

    fusion_prob = (
        ALPHA * hybrid_prob
        + (1.0 - ALPHA) * mixed_prob
    )

    fused_pred = np.asarray(LABELS)[
        fusion_prob.argmax(axis=1)
    ]

    fusion_metrics = metrics(
        y_valid, fused_pred
    )

    # ---------------------------------------
    # SAVE MATCHED PROBABILITIES
    # ---------------------------------------
    probabilities_df = pd.DataFrame({
        "OuterFold": OUTER,
        "InnerFold": INNER,
        "RecordID":
            data.iloc[valid_idx]["RecordID"].to_numpy(),
        "ProjectID": projects[valid_idx],
        "ActualClass": y_valid,
        "HybridPrediction": hybrid_pred,
        "MixedPrediction": mixed_pred,
        "FusionPrediction": fused_pred,
    })

    for i, label in enumerate(LABELS):
        probabilities_df[
            f"HybridProb_{label}"
        ] = hybrid_prob[:, i]

        probabilities_df[
            f"MixedProb_{label}"
        ] = mixed_prob[:, i]

        probabilities_df[
            f"FusionProb_{label}"
        ] = fusion_prob[:, i]

    prediction_file = (
        TABLES / "24_matched_probabilities_fold1.csv"
    )

    probabilities_df.to_csv(
        prediction_file, index=False
    )

    summary = pd.DataFrame([
        {"Model": "Hybrid", **hybrid_metrics},
        {"Model": "Mixed75_25", **mixed_metrics},
        {"Model": "Fusion75_25", **fusion_metrics},
    ])

    summary.to_csv(
        TABLES / "24_fusion_fold1_summary.csv",
        index=False,
    )

    hybrid_correct = hybrid_pred == y_valid
    mixed_correct = mixed_pred == y_valid
    fusion_correct = fused_pred == y_valid

    audit = {
        "Experiment": "Phase 8 Experiment 24A",
        "OuterFold": OUTER,
        "InnerFold": INNER,
        "FusionAlphaHybrid": ALPHA,
        "FusionAlphaMixed": 1.0 - ALPHA,
        "TrainingRecords": int(len(train_idx)),
        "ValidationRecords": int(len(valid_idx)),
        "HybridCorrect": int(hybrid_correct.sum()),
        "MixedCorrect": int(mixed_correct.sum()),
        "FusionCorrect": int(fusion_correct.sum()),
        "FusionCorrectedHybridErrors": int(
            ((~hybrid_correct) & fusion_correct).sum()
        ),
        "FusionIntroducedHybridErrors": int(
            (hybrid_correct & (~fusion_correct)).sum()
        ),
        "HybridMixedDisagreements": int(
            (hybrid_pred != mixed_pred).sum()
        ),
        "ProbabilitiesCalibrated": False,
        "OriginalTestAccessed": False,
        "OuterValidationEvaluated": False,
        "Exploratory": True,
    }

    (
        REPORTS / "24_fusion_fold1_audit.json"
    ).write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 24A RESULTS")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\nPREDICTION AUDIT")
    print("Hybrid correct:", audit["HybridCorrect"])
    print("Mixed correct:", audit["MixedCorrect"])
    print("Fusion correct:", audit["FusionCorrect"])
    print(
        "Fusion corrected hybrid errors:",
        audit["FusionCorrectedHybridErrors"]
    )
    print(
        "Fusion introduced hybrid errors:",
        audit["FusionIntroducedHybridErrors"]
    )
    print(
        "Hybrid/Mixed disagreements:",
        audit["HybridMixedDisagreements"]
    )

    print("\nSaved:", prediction_file)
    print("Original test accessed: No")
    print("Outer validation evaluated: No")
    print("Original labels modified: No")
    print("Experiment 24A completed.")


if __name__ == "__main__":
    main()
