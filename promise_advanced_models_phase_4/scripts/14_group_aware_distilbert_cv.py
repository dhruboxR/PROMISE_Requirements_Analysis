
"""PROMISE+ Phase 4, Experiment 14.

Three-fold group-aware cross-validation of weighted DistilBERT.

Reuses the exact folds from Experiment 01.
Never accesses the original test partition.
No checkpoint selection using held-out fold labels.
"""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

OUTPUT = ROOT / "outputs"
METRICS_DIR = OUTPUT / "metrics"
REPORTS_DIR = OUTPUT / "reports"
TABLES_DIR = OUTPUT / "tables"
MODELS_DIR = OUTPUT / "models" / "14_group_cv_distilbert"

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 6
MAX_LENGTH = 128

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}


def import_experiment_01():
    """Import the original data and grouping implementation."""
    path = SCRIPT_DIR / "01_group_aware_feature_comparison.py"

    spec = importlib.util.spec_from_file_location(
        "original_group_aware_experiment", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


class RequirementDataset(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer):
        texts = frame["NormalizedText"].astype(str).tolist()

        self.encoding = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        self.labels = [
            LABEL_TO_ID[label]
            for label in frame["_class_"].astype(str)
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encoding.items()
        }

        item["labels"] = self.labels[index]
        return item


def calculate_metrics(prediction):
    logits, label_ids = prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            label_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(label_ids, predicted_ids),
        "balanced_accuracy": balanced_accuracy_score(
            label_ids, predicted_ids
        ),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
        "weighted_f1": f1_score(
            label_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            average="weighted",
            zero_division=0,
        ),
    }


def calculate_class_weights(frame):
    counts = np.array(
        [
            (frame["_class_"] == label).sum()
            for label in LABELS
        ],
        dtype=np.float64,
    )

    if np.any(counts == 0):
        raise ValueError("Training fold is missing a class")

    weights = np.sqrt(
        len(frame) / (len(LABELS) * counts)
    )

    weights /= np.average(weights, weights=counts)
    weights = np.clip(weights, 0.5, 3.0)

    return weights


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(
    fold_number,
    train_frame,
    validation_frame,
    tokenizer,
):
    print(f"\n{'=' * 65}")
    print(f"FOLD {fold_number} / 3")
    print(f"{'=' * 65}")

    print("Training records:", len(train_frame))
    print("Held-out records:", len(validation_frame))

    fold_dir = MODELS_DIR / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(SEED)

    weights = calculate_class_weights(train_frame)
    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
    )

    print("\nClass weights:")
    for label, weight in zip(LABELS, weights):
        print(f"{label:>3}: {weight:.4f}")

    train_dataset = RequirementDataset(
        train_frame, tokenizer
    )

    validation_dataset = RequirementDataset(
        validation_frame, tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    def weighted_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"]

        loss_sum = F.cross_entropy(
            logits.float(),
            labels.long(),
            weight=weight_tensor.to(logits.device),
            reduction="sum",
        )

        denominator = (
            num_items_in_batch
            if num_items_in_batch is not None
            else labels.numel()
        )

        return loss_sum / denominator

    args = TrainingArguments(
        output_dir=str(fold_dir / "checkpoints"),
        num_train_epochs=EPOCHS,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=50,
        eval_accumulation_steps=4,
        dataloader_num_workers=0,
        report_to="none",
        push_to_hub=False,
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_loss_func=weighted_loss,
    )

    print("\nTraining...")
    trainer.train()

    print("\nEvaluating held-out fold...")
    predictions = trainer.predict(validation_dataset)

    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)
    actual_ids = np.asarray(predictions.label_ids)

    y_true = [ID_TO_LABEL[int(x)] for x in actual_ids]
    y_pred = [ID_TO_LABEL[int(x)] for x in predicted_ids]

    metrics = calculate_metrics(
        (logits, actual_ids)
    )

    metrics["fold"] = fold_number
    metrics["train_records"] = len(train_frame)
    metrics["validation_records"] = len(validation_frame)

    print("\nFOLD RESULTS")
    for key in [
        "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1",
    ]:
        print(f"{key}: {metrics[key]:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_path = (
        REPORTS_DIR
        / f"14_distilbert_fold_{fold_number}_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    result_frame = validation_frame[
        ["RecordID", "ProjectID", "_class_"]
    ].copy()

    result_frame["PredictedLabel"] = y_pred
    result_frame["Fold"] = fold_number

    result_frame.to_csv(
        TABLES_DIR
        / f"14_distilbert_fold_{fold_number}_predictions.csv",
        index=False,
    )

    trainer.save_model(str(fold_dir / "final_model"))
    tokenizer.save_pretrained(
        str(fold_dir / "final_model")
    )

    del trainer
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, result_frame


def main():
    for folder in [
        METRICS_DIR,
        REPORTS_DIR,
        TABLES_DIR,
        MODELS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    print("\nPHASE 4: GROUP-AWARE DISTILBERT CV")
    print("=" * 65)
    print("GPU:", torch.cuda.get_device_name(0))

    original = import_experiment_01()
    data = original.load_development_data()

    if original.SEED != SEED or original.LABELS != LABELS:
        raise ValueError("Original experiment settings differ")

    X = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()
    groups = original.create_connected_groups(data)

    print("Development records:", len(data))
    print("Connected groups:", len(set(groups)))
    print("Classes:", len(set(y)))

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    folds = list(cv.split(X, y, groups))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_metrics = []
    all_predictions = []

    for fold_number, (train_idx, val_idx) in enumerate(
        folds, start=1
    ):
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise ValueError("Connected-group leakage")

        train_frame = data.iloc[train_idx].copy()
        val_frame = data.iloc[val_idx].copy()

        if set(train_frame["ProjectID"]) & set(
            val_frame["ProjectID"]
        ):
            raise ValueError("Project leakage")

        if set(train_frame["NormalizedText"].str.casefold()) & set(
            val_frame["NormalizedText"].str.casefold()
        ):
            raise ValueError("Text overlap")

        if set(train_frame["_class_"]) != set(LABELS):
            raise ValueError("Training fold missing classes")

        if set(val_frame["_class_"]) != set(LABELS):
            raise ValueError("Validation fold missing classes")

        metrics, predictions = train_one_fold(
            fold_number,
            train_frame,
            val_frame,
            tokenizer,
        )

        all_metrics.append(metrics)
        all_predictions.append(predictions)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_df.to_csv(
        METRICS_DIR / "14_distilbert_fold_metrics.csv",
        index=False,
    )

    combined = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    if len(combined) != len(data):
        raise ValueError("Incomplete out-of-fold predictions")

    if combined["RecordID"].duplicated().any():
        raise ValueError("Duplicate out-of-fold predictions")

    if set(combined["RecordID"]) != set(data["RecordID"]):
        raise ValueError("Missing out-of-fold RecordIDs")

    combined.to_csv(
        TABLES_DIR / "14_distilbert_oof_predictions.csv",
        index=False,
    )

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    summary = {}

    print("\nTHREE-FOLD CROSS-VALIDATION SUMMARY")
    print("=" * 65)

    for metric in metric_names:
        mean = metrics_df[metric].mean()
        std = metrics_df[metric].std(ddof=1)

        summary[metric] = {
            "mean": float(mean),
            "std": float(std),
        }

        print(
            f"{metric:<22} "
            f"mean={mean:.4f}  std={std:.4f}"
        )

    print("\nOUT-OF-FOLD AGGREGATE")

    actual = combined["_class_"]
    predicted = combined["PredictedLabel"]

    oof_accuracy = accuracy_score(actual, predicted)
    oof_macro_f1 = f1_score(
        actual,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    oof_balanced = balanced_accuracy_score(
        actual, predicted
    )

    print(f"OOF accuracy: {oof_accuracy:.4f}")
    print(f"OOF macro-F1: {oof_macro_f1:.4f}")
    print(f"OOF balanced accuracy: {oof_balanced:.4f}")

    summary["out_of_fold"] = {
        "accuracy": float(oof_accuracy),
        "macro_f1": float(oof_macro_f1),
        "balanced_accuracy": float(oof_balanced),
        "records": len(combined),
    }

    summary["configuration"] = {
        "model": MODEL_NAME,
        "folds": 3,
        "epochs": EPOCHS,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "grouping": "Experiment 01 connected groups",
        "held_out_fold_used_for_selection": False,
        "original_test_accessed": False,
    }

    (
        REPORTS_DIR / "14_distilbert_cv_summary.json"
    ).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment 14 completed.")
    print("Original test partition was not accessed.")


if __name__ == "__main__":
    main()
