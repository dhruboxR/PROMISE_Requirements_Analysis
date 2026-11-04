
"""PROMISE+ Phase 4, Experiment 03.

Fine-tune DistilBERT for 12-class requirements classification.
Uses frozen Phase 2 train/validation partitions only.
The original test partition is never opened.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


SEED = 42
MODEL_NAME = "distilbert-base-uncased"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

PHASE4 = Path(__file__).resolve().parents[1]
ROOT = PHASE4.parent
PHASE2 = ROOT / "promise_preprocessing_phase_2"

SPLITS = PHASE2 / "outputs" / "splits"
MANIFEST = (
    PHASE2 / "outputs" / "reports"
    / "08_final_split_manifest.json"
)

RUN_DIR = PHASE4 / "outputs" / "models" / "03_distilbert"
METRICS_DIR = PHASE4 / "outputs" / "metrics"
REPORTS_DIR = PHASE4 / "outputs" / "reports"
TABLES_DIR = PHASE4 / "outputs" / "tables"


def checksum(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_split(name):
    if name not in ("train", "validation"):
        raise ValueError("Test data is not allowed in this experiment")

    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    info = manifest["splits"][name]
    path = SPLITS / info["file"]

    if checksum(path) != info["sha256"]:
        raise ValueError(f"{name} checksum mismatch")

    df = pd.read_csv(
        path,
        dtype={
            "RecordID": str,
            "ProjectID": str,
            "_class_": str,
        },
        keep_default_na=False,
    )

    if len(df) != info["records"]:
        raise ValueError(f"{name} record count mismatch")

    if set(df["_class_"]) != set(LABELS):
        raise ValueError(f"Unexpected classes in {name}")

    if df["NormalizedText"].astype(str).str.strip().eq("").any():
        raise ValueError(f"Blank text in {name}")

    return df


class RequirementsDataset(Dataset):
    def __init__(self, frame, tokenizer):
        texts = frame["NormalizedText"].astype(str).tolist()

        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=False,
        )

        self.labels = [
            label2id[label]
            for label in frame["_class_"]
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            **{
                key: values[index]
                for key, values in self.encodings.items()
            },
            "labels": self.labels[index],
        }


def calculate_metrics(eval_prediction):
    logits = eval_prediction.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    predicted = np.argmax(logits, axis=-1)
    actual = eval_prediction.label_ids

    precision, recall, _, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=list(range(12)),
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1_score(
            actual, predicted,
            labels=list(range(12)),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            actual, predicted,
            average="weighted",
            zero_division=0,
        ),
    }


def main():
    for folder in (
        RUN_DIR, METRICS_DIR, REPORTS_DIR, TABLES_DIR
    ):
        folder.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU training required")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train = load_split("train")
    validation = load_split("validation")

    if set(train["RecordID"]) & set(validation["RecordID"]):
        raise ValueError("Record overlap")

    if set(train["ProjectID"]) & set(validation["ProjectID"]):
        raise ValueError("Project leakage")

    print("\nPHASE 4: DISTILBERT FINE-TUNING")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Training records:", len(train))
    print("Validation records:", len(validation))
    print("Classes:", len(LABELS))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = RequirementsDataset(train, tokenizer)
    validation_dataset = RequirementsDataset(
        validation, tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )

    args = TrainingArguments(
        output_dir=str(RUN_DIR / "checkpoints"),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=20,
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
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=calculate_metrics,
    )

    print("\nBeginning training...")
    trainer.train()

    print("\nEvaluating selected checkpoint...")
    result = trainer.evaluate()

    print("\nFINAL VALIDATION METRICS")
    for key, value in result.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    # Validation-only detailed predictions.
    predictions = trainer.predict(validation_dataset)

    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    pred_ids = np.argmax(logits, axis=-1)
    actual_ids = predictions.label_ids

    true_labels = [id2label[int(i)] for i in actual_ids]
    predicted_labels = [
        id2label[int(i)] for i in pred_ids
    ]

    validation_results = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    validation_results["PredictedLabel"] = predicted_labels
    validation_results["Correct"] = (
        validation_results["_class_"]
        == validation_results["PredictedLabel"]
    )

    validation_results.to_csv(
        TABLES_DIR / "03_distilbert_validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    (
        REPORTS_DIR / "03_distilbert_classification_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    metric_record = {
        "model": MODEL_NAME,
        "partition": "validation",
        **{
            key: float(value)
            for key, value in result.items()
            if isinstance(value, (int, float))
        },
    }

    (
        REPORTS_DIR / "03_distilbert_training_summary.json"
    ).write_text(
        json.dumps({
            "experiment": "03 DistilBERT",
            "model": MODEL_NAME,
            "seed": SEED,
            "training_records": len(train),
            "validation_records": len(validation),
            "test_partition_used": False,
            "metrics": metric_record,
        }, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([metric_record]).to_csv(
        METRICS_DIR / "03_distilbert_validation_metrics.csv",
        index=False,
    )

    final_model_dir = RUN_DIR / "best_model"

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print("\nSaved model:", final_model_dir)
    print("Training completed.")
    print("Original test partition was not accessed.")


if __name__ == "__main__":
    main()
