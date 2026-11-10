
"""
Phase 8 Experiment 09
Full-training fine-tuned MiniLM, outer fold 1.

Epoch count is imported from Experiment 08's inner
validation result. The pretrained model is reinitialized,
then trained on the complete outer-training partition.

Outer-fold labels are never used for model selection.
Original test records are not loaded.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"
MODELS = ROOT / "outputs/models"

THESIS = ROOT.parent

P4 = (
    THESIS
    / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

P5_MANIFEST = (
    THESIS
    / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

EXP08_SCRIPT = (
    ROOT / "scripts/08_finetune_minilm_feasibility.py"
)

EXP08_REPORT = (
    REPORTS / "08_finetune_feasibility.json"
)

OUTER_FOLD = 1


def import_script(path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    exp08 = import_script(
        EXP08_SCRIPT, "experiment08_reuse"
    )

    phase4 = import_script(
        P4, "phase4_data_reuse09"
    )

    # Keep Experiment 09 consistent with Experiment 08.
    if (
        phase4.SEED != exp08.SEED
        or phase4.LABELS != exp08.LABELS
        or exp08.MAX_LENGTH != 128
        or exp08.MODEL_NAME
        != "sentence-transformers/all-MiniLM-L6-v2"
    ):
        raise ValueError("Configuration mismatch")

    if not EXP08_REPORT.exists():
        raise FileNotFoundError(
            "Experiment 08 report missing. "
            "Run Experiment 08 first."
        )

    previous = json.loads(
        EXP08_REPORT.read_text(encoding="utf-8")
    )

    if (
        previous["experiment"]
        != "Phase 8 Experiment 08"
        or int(previous["outer_fold"]) != OUTER_FOLD
    ):
        raise ValueError("Experiment 08 metadata mismatch")

    selected_epoch = int(previous["best_epoch"])

    if not 1 <= selected_epoch <= exp08.EPOCHS:
        raise ValueError("Invalid selected epoch")

    print("\nPHASE 8 EXPERIMENT 09")
    print("=" * 65)

    print(
        "Epochs selected by Experiment 08:",
        selected_epoch,
    )

    print("Reloading development data...", flush=True)

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate record IDs")

    texts = data["NormalizedText"].astype(str).tolist()
    labels = data["_class_"].astype(str).to_numpy()

    if set(labels) != set(exp08.LABELS):
        raise ValueError("Unexpected classes")

    y = np.asarray(
        [exp08.LABEL_TO_ID[label] for label in labels],
        dtype=int,
    )

    groups = np.asarray(
        phase4.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        P5_MANIFEST,
        dtype={
            "RecordID": str,
            "ProjectID": str,
        },
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"])
        != set(data["RecordID"])
    ):
        raise ValueError("Invalid outer fold manifest")

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

    outer_folds = (
        aligned["Fold"].astype(int).to_numpy()
    )

    if set(outer_folds) != {1, 2, 3}:
        raise ValueError("Invalid fold assignments")

    group_distribution = pd.DataFrame({
        "Group": groups,
        "Fold": outer_folds,
    }).groupby("Group")["Fold"].nunique()

    if (group_distribution > 1).any():
        raise ValueError("Connected-group leakage")

    train_idx = np.flatnonzero(
        outer_folds != OUTER_FOLD
    )

    val_idx = np.flatnonzero(
        outer_folds == OUTER_FOLD
    )

    if len(train_idx) != 2159 or len(val_idx) != 968:
        raise ValueError("Unexpected fold sizes")

    if set(groups[train_idx]) & set(groups[val_idx]):
        raise ValueError("Training/validation group overlap")

    if len(np.unique(y[train_idx])) != len(exp08.LABELS):
        raise ValueError("Training data missing classes")

    print("Training records:", len(train_idx))
    print("Outer validation records:", len(val_idx))
    print("Group isolation: PASS")
    print("Original test accessed: No")

    exp08.set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    tokenizer = exp08.AutoTokenizer.from_pretrained(
        exp08.MODEL_NAME
    )

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=exp08.MAX_LENGTH,
        return_tensors="pt",
    )

    dataset = exp08.RequirementDataset(
        encodings, y
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=exp08.BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=exp08.BATCH_SIZE,
        shuffle=False,
    )

    print(
        "\nInitializing a fresh MiniLM checkpoint...",
        flush=True,
    )

    model = exp08.MiniLMClassifier().to(device)

    class_weights = exp08.compute_class_weights(
        y[train_idx]
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp08.LR,
        weight_decay=0.01,
    )

    history = []

    print(
        "\nTraining on all outer-training records...",
        flush=True,
    )

    for epoch in range(1, selected_epoch + 1):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            labels_batch = (
                batch["labels"].to(device)
            )

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

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        avg_loss = total_loss / total_samples

        history.append({
            "Epoch": epoch,
            "TrainLoss": avg_loss,
            "TrainingRecords": len(train_idx),
        })

        print(
            f"Epoch {epoch}/{selected_epoch}: "
            f"TrainLoss={avg_loss:.4f}",
            flush=True,
        )

    print(
        "\nEvaluating fixed outer fold 1...",
        flush=True,
    )

    actual, predicted = exp08.evaluate(
        model, validation_loader, device
    )

    if (
        len(actual) != len(val_idx)
        or not np.array_equal(actual, y[val_idx])
        or not np.isin(
            predicted, np.arange(len(exp08.LABELS))
        ).all()
    ):
        raise ValueError("Invalid evaluation predictions")

    results = exp08.calculate_metrics(
        actual, predicted
    )

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(exp08.LABELS))),
        target_names=exp08.LABELS,
        zero_division=0,
        output_dict=True,
    )

    predictions = pd.DataFrame({
        "RecordID": data.iloc[val_idx][
            "RecordID"
        ].to_numpy(),
        "Fold": OUTER_FOLD,
        "ActualClass": [
            exp08.LABELS[i] for i in actual
        ],
        "PredictedClass": [
            exp08.LABELS[i] for i in predicted
        ],
    })

    predictions.to_csv(
        TABLES / "09_full_training_fold1_predictions.csv",
        index=False,
    )

    pd.DataFrame(history).to_csv(
        TABLES / "09_full_training_history.csv",
        index=False,
    )

    summary = {
        "experiment": "Phase 8 Experiment 09",
        "method": "Full-training fine-tuned MiniLM",
        "model": exp08.MODEL_NAME,
        "outer_fold": OUTER_FOLD,
        "training_records": len(train_idx),
        "evaluation_records": len(val_idx),
        "selected_epoch_source": "Experiment 08 inner validation",
        "selected_epochs": selected_epoch,
        "learning_rate": exp08.LR,
        "batch_size": exp08.BATCH_SIZE,
        "outer_metrics": results,
        "class_report": report,
        "original_test_accessed": False,
        "three_fold_evaluation_completed": False,
        "exploratory_development_result": True,
        "previous_outer_results_examined": True,
    }

    (
        REPORTS / "09_full_training_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 09 RESULTS")
    print("=" * 65)

    print("Training records:", len(train_idx))
    print("Selected epochs:", selected_epoch)

    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")

    print("\nMinority-class results:")

    for target in ["O", "SC", "PO"]:
        values = report[target]

        print(
            f"{target}: "
            f"Precision={values['precision']:.4f} "
            f"Recall={values['recall']:.4f} "
            f"F1={values['f1-score']:.4f}"
        )

    print("\nOriginal test accessed: No")
    print("Full three-fold evaluation: No")
    print("Experiment 09 completed.")


if __name__ == "__main__":
    main()
