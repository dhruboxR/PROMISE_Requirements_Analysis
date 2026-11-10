
"""
Phase 8 Experiment 08
Fine-tuned MiniLM feasibility experiment.

Outer fold 1 is reserved for final evaluation.
An inner project-aware split is used for early stopping.

This is a single-fold exploratory experiment.
It does not establish three-fold performance.
"""

from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import random

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

P4 = THESIS / "promise_advanced_models_phase_4"
P5 = THESIS / "promise_improved_model_phase_5"

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
OUTER_FOLD = 1

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

LABEL_TO_ID = {
    label: i for i, label in enumerate(LABELS)
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RequirementDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(
            labels, dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


class MiniLMClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            hidden_size, len(LABELS)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None) :
        encoder_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)

        hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(
            hidden.dtype
        )

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

        return self.classifier(
            self.dropout(pooled)
        )


def compute_class_weights(labels):
    counts = np.bincount(
        labels, minlength=len(LABELS)
    )

    if (counts == 0).any():
        missing = [
            LABELS[i]
            for i in np.where(counts == 0)[0]
        ]
        raise ValueError(
            f"Training partition missing classes: {missing}"
        )

    n = len(labels)
    k = len(LABELS)

    raw = np.sqrt(n / (k * counts))

    normalizer = np.sum(
        counts * raw
    ) / n

    return torch.tensor(
        raw / normalizer,
        dtype=torch.float32
    )


def evaluate(model, loader, device):
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            logits = model(**inputs)
            predictions = logits.argmax(dim=1)

            actual.extend(labels.numpy().tolist())
            predicted.extend(
                predictions.cpu().numpy().tolist()
            )

    return np.asarray(actual), np.asarray(predicted)


def calculate_metrics(actual, predicted):
    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(
                actual, predicted,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
    }


def main():
    set_seed()

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    phase4 = load_module(
        P4 / "scripts/01_group_aware_feature_comparison.py",
        "phase4_cv08",
    )

    if phase4.SEED != SEED or phase4.LABELS != LABELS:
        raise ValueError("Phase 4 configuration mismatch")

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    texts = data["NormalizedText"].astype(str).tolist()
    labels = data["_class_"].astype(str).to_numpy()

    if not set(labels).issubset(LABEL_TO_ID):
        raise ValueError("Unexpected classification labels")

    y = np.asarray(
        [LABEL_TO_ID[label] for label in labels]
    )

    groups = np.asarray(
        phase4.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        P5 / "outputs/tables/01_fixed_fold_assignments.csv",
        dtype={
            "RecordID": str,
            "ProjectID": str,
        },
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or data["RecordID"].duplicated().any()
        or set(manifest["RecordID"]) != set(data["RecordID"])
    ):
        raise ValueError("Invalid fold manifest")

    aligned = (
        manifest.set_index("RecordID")
        .loc[data["RecordID"]]
    )

    if not np.array_equal(
        aligned["ActualClass"].to_numpy(), labels
    ):
        raise ValueError("Manifest class mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        data["ProjectID"].to_numpy(),
    ):
        raise ValueError("Manifest project mismatch")

    outer_folds = aligned["Fold"].astype(int).to_numpy()

    if set(outer_folds) != {1, 2, 3}:
        raise ValueError("Unexpected fold assignments")

    group_fold_count = pd.DataFrame({
        "Group": groups,
        "Fold": outer_folds,
    }).groupby("Group")["Fold"].nunique()

    if (group_fold_count > 1).any():
        raise ValueError("Connected-group leakage")

    outer_train = np.flatnonzero(
        outer_folds != OUTER_FOLD
    )

    outer_val = np.flatnonzero(
        outer_folds == OUTER_FOLD
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    selected_split = None

    for inner_train, inner_val in inner_cv.split(
        np.asarray(texts)[outer_train],
        y[outer_train],
        groups[outer_train],
    ):
        train_indices = outer_train[inner_train]
        validation_indices = outer_train[inner_val]

        # Require all 12 labels in the training partition.
        if len(np.unique(y[train_indices])) == len(LABELS):
            selected_split = (
                train_indices,
                validation_indices,
            )
            break

    if selected_split is None:
        raise ValueError(
            "No suitable inner training split found"
        )

    train_idx, inner_val_idx = selected_split

    if (
        set(groups[train_idx])
        & set(groups[inner_val_idx])
    ):
        raise ValueError("Inner group leakage")

    if (
        set(groups[outer_train])
        & set(groups[outer_val])
    ):
        raise ValueError("Outer group leakage")

    print("\nPHASE 8 EXPERIMENT 08", flush=True)
    print("=" * 65)

    print("Inner training records:", len(train_idx))
    print("Inner validation records:", len(inner_val_idx))
    print("Outer validation records:", len(outer_val))
    print("Original test accessed: No")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = RequirementDataset(encodings, y)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    inner_loader = DataLoader(
        Subset(dataset, inner_val_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    outer_loader = DataLoader(
        Subset(dataset, outer_val.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = MiniLMClassifier().to(device)

    class_weights = compute_class_weights(
        y[train_idx]
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            labels_batch = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }

            optimizer.zero_grad(set_to_none=True)

            logits = model(**inputs)

            loss = criterion(
                logits, labels_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            batch_size = len(labels_batch)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        actual, predicted = evaluate(
            model, inner_loader, device
        )

        results = calculate_metrics(
            actual, predicted
        )

        avg_loss = total_loss / total_examples

        history.append({
            "Epoch": epoch,
            "TrainLoss": avg_loss,
            "InnerAccuracy": results["accuracy"],
            "InnerMacroF1": results["macro_f1"],
            "InnerBalancedAccuracy": results[
                "balanced_accuracy"
            ],
        })

        print(
            f"Epoch {epoch}: "
            f"Loss={avg_loss:.4f} "
            f"InnerAcc={results['accuracy']:.4f} "
            f"InnerMacroF1={results['macro_f1']:.4f}",
            flush=True,
        )

        if results["macro_f1"] > best_f1:
            best_f1 = results["macro_f1"]
            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not complete")

    # Restore the best inner-validation checkpoint.
    model.load_state_dict(best_state)

    # Evaluate the outer fold only after checkpoint selection.
    actual, predicted = evaluate(
        model, outer_loader, device
    )

    outer_metrics = calculate_metrics(
        actual, predicted
    )

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(history).to_csv(
        TABLES / "08_finetune_epoch_history.csv",
        index=False,
    )

    predictions = pd.DataFrame({
        "RecordID": data.iloc[outer_val][
            "RecordID"
        ].to_numpy(),
        "Fold": OUTER_FOLD,
        "ActualClass": [
            LABELS[index] for index in actual
        ],
        "PredictedClass": [
            LABELS[index] for index in predicted
        ],
    })

    predictions.to_csv(
        TABLES / "08_finetune_fold1_predictions.csv",
        index=False,
    )

    summary = {
        "experiment": "Phase 8 Experiment 08",
        "model": MODEL_NAME,
        "training_method": "End-to-end fine-tuning",
        "outer_fold": OUTER_FOLD,
        "inner_train_records": len(train_idx),
        "inner_validation_records": len(inner_val_idx),
        "outer_validation_records": len(outer_val),
        "best_epoch": best_epoch,
        "best_inner_macro_f1": best_f1,
        "outer_metrics": outer_metrics,
        "class_report": report,
        "original_test_accessed": False,
        "three_fold_evaluation_completed": False,
        "exploratory_development_result": True,
    }

    (
        REPORTS / "08_finetune_feasibility.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nOUTER FOLD 1 RESULTS")
    print("=" * 65)

    print("Selected epoch:", best_epoch)

    for name, value in outer_metrics.items():
        print(f"{name}: {value:.6f}")

    print("\nMinority-class results:")

    for label in ["O", "SC", "PO"]:
        values = report[label]

        print(
            f"{label}: "
            f"Precision={values['precision']:.4f} "
            f"Recall={values['recall']:.4f} "
            f"F1={values['f1-score']:.4f}"
        )

    print("\nOriginal test accessed: No")
    print("Three-fold improvement established: No")
    print("Experiment 08 completed.")


if __name__ == "__main__":
    main()
