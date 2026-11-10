
"""
Phase 8 Experiment 12
Class-weight sensitivity for fine-tuned MiniLM.

Step 1: Compare weight exponents on an inner group-aware split.
Step 2: Select exponent and epoch count using inner macro-F1.
Step 3: Initialize a fresh model and train on full outer training.
Step 4: Evaluate outer fold 1 once.

Exploratory development experiment.
No original test data accessed.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

P4_SCRIPT = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

P5_MANIFEST = (
    THESIS / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

EXP08_SCRIPT = ROOT / "scripts/08_finetune_minilm_feasibility.py"

SEED = 42
OUTER_FOLD = 1
EXPONENTS = [0.0, 0.25, 0.5]
MAX_EPOCHS = 4


def import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_weights(y, exponent, num_classes):
    counts = np.bincount(
        y, minlength=num_classes
    ).astype(float)

    if np.any(counts == 0):
        raise ValueError("Training partition missing classes")

    n = len(y)
    k = num_classes

    weights = (n / (k * counts)) ** exponent

    # Match the normalization convention used previously.
    mean_weight = np.sum(counts * weights) / n
    weights = weights / mean_weight

    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_records = 0

    for batch in loader:
        labels = batch["labels"].to(device)

        inputs = {
            name: tensor.to(device)
            for name, tensor in batch.items()
            if name != "labels"
        }

        optimizer.zero_grad(set_to_none=True)

        logits = model(**inputs)

        loss = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        batch_count = labels.size(0)
        total_loss += float(loss.item()) * batch_count
        total_records += batch_count

    return total_loss / total_records


def create_model_and_optimizer(exp08, y_train, exponent, device):
    model = exp08.MiniLMClassifier().to(device)

    weights = class_weights(
        y_train,
        exponent=exponent,
        num_classes=len(exp08.LABELS),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp08.LR,
        weight_decay=0.01,
    )

    return model, optimizer, criterion


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    exp08 = import_module(EXP08_SCRIPT, "phase8_exp08")
    phase4 = import_module(P4_SCRIPT, "phase4_exp12")

    if (
        phase4.SEED != SEED
        or phase4.LABELS != exp08.LABELS
        or exp08.SEED != SEED
        or exp08.MAX_LENGTH != 128
        or exp08.MODEL_NAME !=
            "sentence-transformers/all-MiniLM-L6-v2"
    ):
        raise ValueError("Experiment configuration mismatch")

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development dataset size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID values")

    texts = data["NormalizedText"].astype(str).tolist()
    labels = data["_class_"].astype(str).to_numpy()

    if set(labels) != set(exp08.LABELS):
        raise ValueError("Invalid class labels")

    y = np.array(
        [exp08.LABEL_TO_ID[label] for label in labels],
        dtype=int,
    )

    groups = np.asarray(phase4.create_connected_groups(data))

    manifest = pd.read_csv(
        P5_MANIFEST,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != 3127
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"]) != set(data["RecordID"])
    ):
        raise ValueError("Fold manifest mismatch")

    aligned = (
        manifest.set_index("RecordID")
        .loc[data["RecordID"]]
    )

    if not np.array_equal(
        aligned["ActualClass"].to_numpy(), labels
    ):
        raise ValueError("Manifest label mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        data["ProjectID"].to_numpy(),
    ):
        raise ValueError("Manifest project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Unexpected outer fold assignments")

    group_folds = pd.DataFrame({
        "Group": groups, "Fold": folds
    }).groupby("Group")["Fold"].nunique()

    if (group_folds > 1).any():
        raise ValueError("Group overlap across outer folds")

    outer_train = np.flatnonzero(folds != OUTER_FOLD)
    outer_valid = np.flatnonzero(folds == OUTER_FOLD)

    if len(outer_train) != 2159 or len(outer_valid) != 968:
        raise ValueError("Unexpected outer fold sizes")

    if set(groups[outer_train]) & set(groups[outer_valid]):
        raise ValueError("Outer train/validation leakage")

    # Reproduce Experiment 08's deterministic inner split.
    inner_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    selected_split = None

    for tr, va in inner_cv.split(
        np.asarray(texts)[outer_train],
        y[outer_train],
        groups[outer_train],
    ):
        train_indices = outer_train[tr]
        valid_indices = outer_train[va]

        if len(np.unique(y[train_indices])) == len(exp08.LABELS):
            selected_split = (train_indices, valid_indices)
            break

    if selected_split is None:
        raise ValueError("No suitable inner split")

    inner_train, inner_valid = selected_split

    if set(groups[inner_train]) & set(groups[inner_valid]):
        raise ValueError("Inner group leakage")

    print("\nPHASE 8 EXPERIMENT 12")
    print("=" * 70)
    print("Inner training records:", len(inner_train))
    print("Inner validation records:", len(inner_valid))
    print("Full outer training records:", len(outer_train))
    print("Outer validation records:", len(outer_valid))
    print("Group separation: PASS")
    print("Original test accessed: No")

    exp08.set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device, flush=True)

    tokenizer = exp08.AutoTokenizer.from_pretrained(
        exp08.MODEL_NAME
    )

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=exp08.MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = exp08.RequirementDataset(encoded, y)

    inner_train_loader = DataLoader(
        Subset(dataset, inner_train.tolist()),
        batch_size=exp08.BATCH_SIZE,
        shuffle=True,
    )

    inner_valid_loader = DataLoader(
        Subset(dataset, inner_valid.tolist()),
        batch_size=exp08.BATCH_SIZE,
        shuffle=False,
    )

    history = []
    candidates = []

    # ------------------------------
    # Inner selection
    # ------------------------------
    for exponent in EXPONENTS:
        print(
            f"\nCLASS WEIGHT EXPONENT: {exponent:.2f}",
            flush=True,
        )

        # Reset random initialization for each candidate.
        exp08.set_seed()

        model, optimizer, criterion = create_model_and_optimizer(
            exp08, y[inner_train], exponent, device
        )

        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_epoch(
                model, inner_train_loader,
                optimizer, criterion, device
            )

            actual, predicted = exp08.evaluate(
                model, inner_valid_loader, device
            )

            scores = exp08.calculate_metrics(
                actual, predicted
            )

            row = {
                "Exponent": exponent,
                "Epoch": epoch,
                "TrainLoss": loss,
                "InnerAccuracy": scores["accuracy"],
                "InnerMacroF1": scores["macro_f1"],
                "InnerBalancedAccuracy":
                    scores["balanced_accuracy"],
            }

            history.append(row)
            candidates.append(row)

            print(
                f"  Epoch {epoch}: "
                f"Loss={loss:.4f} "
                f"Acc={scores['accuracy']:.4f} "
                f"MacroF1={scores['macro_f1']:.4f}",
                flush=True,
            )

        del model, optimizer, criterion

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    selection = pd.DataFrame(candidates)

    # Primary objective: macro-F1.
    # Ties: higher accuracy, milder weighting, fewer epochs.
    ranked = selection.sort_values(
        by=[
            "InnerMacroF1",
            "InnerAccuracy",
            "Exponent",
            "Epoch",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    )

    chosen = ranked.iloc[0]

    best_exponent = float(chosen["Exponent"])
    best_epoch = int(chosen["Epoch"])

    print("\nINNER VALIDATION SELECTION")
    print("=" * 70)
    print("Selected exponent:", best_exponent)
    print("Selected epochs:", best_epoch)
    print("Selected inner macro-F1:",
          round(float(chosen["InnerMacroF1"]), 6))

    # ------------------------------
    # Retrain on all outer training
    # ------------------------------
    print("\nRETRAINING ON FULL OUTER TRAINING PARTITION")

    full_train_loader = DataLoader(
        Subset(dataset, outer_train.tolist()),
        batch_size=exp08.BATCH_SIZE,
        shuffle=True,
    )

    outer_valid_loader = DataLoader(
        Subset(dataset, outer_valid.tolist()),
        batch_size=exp08.BATCH_SIZE,
        shuffle=False,
    )

    exp08.set_seed()

    model, optimizer, criterion = create_model_and_optimizer(
        exp08, y[outer_train], best_exponent, device
    )

    full_history = []

    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(
            model, full_train_loader,
            optimizer, criterion, device
        )

        full_history.append({
            "Epoch": epoch,
            "TrainLoss": loss,
            "Exponent": best_exponent,
        })

        print(
            f"Epoch {epoch}/{best_epoch}: "
            f"TrainLoss={loss:.4f}",
            flush=True,
        )

    # No outer evaluation occurred during tuning.
    print("\nEVALUATING OUTER FOLD 1", flush=True)

    actual, predicted = exp08.evaluate(
        model, outer_valid_loader, device
    )

    if (
        len(actual) != len(outer_valid)
        or not np.array_equal(actual, y[outer_valid])
        or not np.isin(
            predicted, np.arange(len(exp08.LABELS))
        ).all()
    ):
        raise ValueError("Prediction validation failed")

    metrics = exp08.calculate_metrics(actual, predicted)

    class_report = classification_report(
        actual, predicted,
        labels=list(range(len(exp08.LABELS))),
        target_names=exp08.LABELS,
        zero_division=0,
        output_dict=True,
    )

    # ------------------------------
    # Save outputs
    # ------------------------------
    pd.DataFrame(history).to_csv(
        TABLES / "12_inner_weight_comparison.csv",
        index=False,
    )

    pd.DataFrame(full_history).to_csv(
        TABLES / "12_full_training_history.csv",
        index=False,
    )

    pd.DataFrame({
        "RecordID": data.iloc[outer_valid]["RecordID"].to_numpy(),
        "Fold": OUTER_FOLD,
        "ActualClass": [
            exp08.LABELS[i] for i in actual
        ],
        "PredictedClass": [
            exp08.LABELS[i] for i in predicted
        ],
    }).to_csv(
        TABLES / "12_outer_fold1_predictions.csv",
        index=False,
    )

    report = {
        "experiment": "Phase 8 Experiment 12",
        "method": "Class-weight sensitivity fine-tuning",
        "candidate_exponents": EXPONENTS,
        "selected_exponent": best_exponent,
        "selected_epochs": best_epoch,
        "selection_metric": "Inner macro-F1",
        "inner_training_records": len(inner_train),
        "inner_validation_records": len(inner_valid),
        "outer_training_records": len(outer_train),
        "outer_validation_records": len(outer_valid),
        "outer_metrics": metrics,
        "class_report": class_report,
        "original_test_accessed": False,
        "exploratory_development_result": True,
        "prior_outer_results_informed_method": True,
    }

    (
        REPORTS / "12_class_weight_sensitivity.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 8 EXPERIMENT 12 RESULTS")
    print("=" * 70)

    print("Selected exponent:", best_exponent)
    print("Selected epochs:", best_epoch)

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    print("\nPER-CLASS RESULTS")

    for target in ["F", "O", "SC", "PO", "PE"]:
        values = class_report[target]

        print(
            f"{target}: "
            f"Precision={values['precision']:.4f} "
            f"Recall={values['recall']:.4f} "
            f"F1={values['f1-score']:.4f}"
        )

    print("\nOriginal test accessed: No")
    print("Three-fold improvement established: No")
    print("Experiment 12 completed.")


if __name__ == "__main__":
    main()
