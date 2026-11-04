
"""Phase 4 Experiment 04: Class-weighted DistilBERT.

Same architecture and data as Experiment 03.
Adds square-root inverse-frequency class weights.
The original test partition is not used.
"""

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import EarlyStoppingCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = Path(__file__).resolve().parent

spec = importlib.util.spec_from_file_location(
    "original_distilbert",
    SCRIPT_DIR / "03_train_distilbert.py",
)

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

SEED = base.SEED
LABELS = base.LABELS
MODEL_NAME = base.MODEL_NAME

PHASE4 = SCRIPT_DIR.parent
RUN_DIR = PHASE4 / "outputs/models/05_extended_weighted_distilbert"
METRICS_DIR = PHASE4 / "outputs/metrics"
REPORTS_DIR = PHASE4 / "outputs/reports"
TABLES_DIR = PHASE4 / "outputs/tables"


def main():
    for folder in (
        RUN_DIR, METRICS_DIR, REPORTS_DIR, TABLES_DIR
    ):
        folder.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train = base.load_split("train")
    validation = base.load_split("validation")

    if set(train["ProjectID"]) & set(validation["ProjectID"]):
        raise ValueError("Project overlap detected")

    if set(train["RecordID"]) & set(validation["RecordID"]):
        raise ValueError("Record overlap detected")

    # Calculate class weights using TRAINING LABELS ONLY.
    counts = np.array(
        [(train["_class_"] == label).sum() for label in LABELS],
        dtype=np.float64,
    )

    if np.any(counts == 0):
        raise ValueError("A training class has zero examples")

    weights = np.sqrt(len(train) / (len(LABELS) * counts))

    # Normalize average training-sample weight to 1.
    weights = weights / np.average(weights, weights=counts)

    # Limit extreme weights to reduce instability.
    weights = np.clip(weights, 0.5, 3.0)

    weight_tensor = torch.tensor(
        weights, dtype=torch.float32
    )

    print("\nPHASE 4: WEIGHTED DISTILBERT")
    print("=" * 60)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Training records:", len(train))
    print("Validation records:", len(validation))

    print("\nTRAINING CLASS WEIGHTS")
    for label, count, weight in zip(LABELS, counts, weights):
        print(
            f"{label:>3}: count={int(count):>4}, "
            f"weight={weight:.4f}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_data = base.RequirementsDataset(train, tokenizer)
    validation_data = base.RequirementsDataset(
        validation, tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=base.label2id,
        id2label=base.id2label,
    )

    # Trainer 5.x custom loss receives the total label count
    # across gradient-accumulation microbatches.
    def weighted_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs["logits"]

        per_example = F.cross_entropy(
            logits.float(),
            labels.long(),
            weight=weight_tensor.to(logits.device),
            reduction="sum",
        )

        if num_items_in_batch is None:
            denominator = labels.numel()
        else:
            denominator = num_items_in_batch

        return per_example / denominator

    args = TrainingArguments(
        output_dir=str(RUN_DIR / "checkpoints"),
        num_train_epochs=6,
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
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=base.calculate_metrics,
        compute_loss_func=weighted_loss,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    print("\nBeginning weighted training...")
    trainer.train()

    results = trainer.evaluate()

    print("\nWEIGHTED DISTILBERT VALIDATION METRICS")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")

    predictions = trainer.predict(validation_data)

    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=-1)

    actual_labels = [
        base.id2label[int(i)]
        for i in predictions.label_ids
    ]

    predicted_labels = [
        base.id2label[int(i)]
        for i in predicted_ids
    ]

    from sklearn.metrics import classification_report

    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    (
        REPORTS_DIR / "05_extended_weighted_distilbert_classification_report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    frame = validation[
        ["RecordID", "ProjectID", "RequirementText", "_class_"]
    ].copy()

    frame["PredictedLabel"] = predicted_labels
    frame["Correct"] = (
        frame["_class_"] == frame["PredictedLabel"]
    )

    frame.to_csv(
        TABLES_DIR / "05_extended_weighted_distilbert_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([{
        "Model": "Class-weighted DistilBERT",
        "Partition": "validation",
        **{
            k: float(v)
            for k, v in results.items()
            if isinstance(v, (int, float))
        },
    }]).to_csv(
        METRICS_DIR / "05_extended_weighted_distilbert_metrics.csv",
        index=False,
    )

    (
        REPORTS_DIR / "05_extended_weighted_distilbert_summary.json"
    ).write_text(
        json.dumps({
            "model": MODEL_NAME,
            "experiment": "class-weighted cross entropy",
            "class_weights": dict(
                zip(LABELS, [float(v) for v in weights])
            ),
            "epochs": 6,
            "learning_rate": 2e-5,
            "selection_metric": "validation_macro_f1",
            "test_partition_used": False,
            "validation_results": {
                k: float(v)
                for k, v in results.items()
                if isinstance(v, (int, float))
            },
        }, indent=2),
        encoding="utf-8",
    )

    final_dir = RUN_DIR / "best_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("\nSaved model:", final_dir)
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
