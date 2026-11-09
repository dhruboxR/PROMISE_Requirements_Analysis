
"""Phase 6 Experiment 03: DeBERTa-v3-base fine-tuning.

Run one verified project-aware fold at a time.
No original test-partition access.

Initial baseline: unweighted cross-entropy, 3 epochs.
"""

import argparse
import importlib.util
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
PHASE4 = THESIS / "promise_advanced_models_phase_4"

FOLD_FILE = (
    ROOT / "outputs" / "tables" / "01_verified_folds.csv"
)

TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
REPORTS = ROOT / "outputs" / "reports"
MODELS = ROOT / "outputs" / "models"

MODEL_NAME = "microsoft/deberta-v3-base"
SEED = 42
MAX_LENGTH = 96

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

LABEL_TO_ID = {
    label: i for i, label in enumerate(LABELS)
}


class RequirementDataset(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer):
        self.frame = frame.reset_index(drop=True)

        self.encoded = tokenizer(
            self.frame["NormalizedText"].astype(str).tolist(),
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        self.labels = [
            LABEL_TO_ID[label]
            for label in self.frame["_class_"].astype(str)
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: values[index]
            for key, values in self.encoded.items()
        }
        item["labels"] = self.labels[index]
        return item


def load_reference_data():
    path = (
        PHASE4 / "scripts"
        / "01_group_aware_feature_comparison.py"
    )

    spec = importlib.util.spec_from_file_location(
        "phase4_reference_for_phase6", path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load Phase 4 data module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.load_development_data()


def compute_metrics(prediction):
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predicted = np.argmax(logits, axis=1)
    actual = prediction.label_ids

    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            actual, predicted
        ),
        "weighted_f1": f1_score(
            actual, predicted,
            labels=list(range(len(LABELS))),
            average="weighted",
            zero_division=0,
        ),
    }



from transformers import TrainerCallback
import math

class StopOnNonFinite(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if logs and key in logs:
                value = float(logs[key])
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"Non-finite {key} at step {state.global_step}"
                    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold", type=int, choices=[1, 2, 3],
        default=1,
    )

    parser.add_argument(
        "--epochs", type=int, default=3,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this initial experiment"
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    for directory in [TABLES, METRICS, REPORTS, MODELS]:
        directory.mkdir(parents=True, exist_ok=True)

    print("\nPHASE 6 EXPERIMENT 03")
    print("DEBERTA-V3-BASE FINE-TUNING")
    print("=" * 60)

    data = load_reference_data().copy()

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    folds = pd.read_csv(
        FOLD_FILE,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(data) != 3127 or len(folds) != 3127:
        raise ValueError("Unexpected development data size")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID in development data")

    if folds["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID in fold file")

    data = data.merge(
        folds[["RecordID", "ProjectID", "Phase6Fold"]],
        on="RecordID",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_verified"),
    )

    if len(data) != 3127:
        raise ValueError("Fold assignment join failed")

    if not (
        data["ProjectID"] == data["ProjectID_verified"]
    ).all():
        raise ValueError("Project assignment mismatch")

    if set(data["_class_"]) != set(LABELS):
        raise ValueError("Class labels mismatch")

    train = data[data["Phase6Fold"] != args.fold].copy()
    valid = data[data["Phase6Fold"] == args.fold].copy()

    expected_sizes = {
        1: (2159, 968),
        2: (2024, 1103),
        3: (2071, 1056),
    }

    if (len(train), len(valid)) != expected_sizes[args.fold]:
        raise ValueError("Fold sizes do not match Phase 5")

    if not set(train["ProjectID"]).isdisjoint(
        set(valid["ProjectID"])
    ):
        raise ValueError("Project leakage detected")

    if not set(train["NormalizedText"]).isdisjoint(
        set(valid["NormalizedText"])
    ):
        raise ValueError("Normalized-text leakage detected")

    if set(train["_class_"]) != set(LABELS):
        raise ValueError("Training fold does not cover all classes")

    print("Fold:", args.fold)
    print("Training records:", len(train))
    print("Held-out records:", len(valid))
    print("Leakage checks: PASSED")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
    )

    train_ds = RequirementDataset(train, tokenizer)
    valid_ds = RequirementDataset(valid, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={i: name for i, name in enumerate(LABELS)},
        label2id=LABEL_TO_ID,
    )

    checkpoint_dir = (
        MODELS / f"03_deberta_smoke_fold{args.fold}"
    )

    # Mixed precision and gradient checkpointing reduce
    # GPU memory usage during fine-tuning.
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.epochs,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        weight_decay=0.01,
        max_grad_norm=0.5,
        optim="adamw_torch",
        max_steps=20,
        warmup_steps=0.1,
        eval_strategy="no",
        save_strategy="no",
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_nan_inf_filter=False,
        report_to="none",
        push_to_hub=False,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        callbacks=[StopOnNonFinite()],
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\nStarting fine-tuning...")
    train_output = trainer.train()

    import math
    if not math.isfinite(train_output.training_loss):
        raise RuntimeError("Non-finite training loss detected")

    print("\n20-STEP STABILITY TEST FINISHED")
    print("Training loss:", train_output.training_loss)
    print("This is not a completed fold evaluation.")
    return

    print("\nGenerating held-out predictions...")
    result = trainer.predict(valid_ds)

    logits = result.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predicted_ids = np.argmax(logits, axis=1)

    if len(predicted_ids) != len(valid):
        raise ValueError("Prediction count mismatch")

    actual_ids = np.array([
        LABEL_TO_ID[label]
        for label in valid["_class_"]
    ])

    final_metrics = compute_metrics(
        type(
            "PredictionOutput",
            (),
            {
                "predictions": logits,
                "label_ids": actual_ids,
            },
        )()
    )

    prediction_table = valid[
        ["RecordID", "ProjectID", "_class_", "NormalizedText"]
    ].copy()

    prediction_table["Fold"] = args.fold
    prediction_table["PredictedLabel"] = [
        LABELS[index] for index in predicted_ids
    ]
    prediction_table["Correct"] = (
        prediction_table["_class_"]
        == prediction_table["PredictedLabel"]
    )

    output_predictions = (
        TABLES / f"03_deberta_fold{args.fold}_predictions.csv"
    )

    prediction_table.to_csv(
        output_predictions, index=False
    )

    class_report = classification_report(
        actual_ids,
        predicted_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_path = (
        REPORTS / f"03_deberta_fold{args.fold}_report.json"
    )

    report = {
        "phase": 6,
        "experiment": 3,
        "model": MODEL_NAME,
        "fold": args.fold,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "training_records": len(train),
        "held_out_records": len(valid),
        "epochs": args.epochs,
        "max_length": MAX_LENGTH,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-5,
        "class_weighting": "none",
        "training_loss": float(
            train_output.training_loss
        ),
        "held_out_metrics": {
            key: float(value)
            for key, value in final_metrics.items()
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "class_report": class_report,
        "original_test_accessed": False,
        "evaluation_note": (
            "Held-out development fold. No independent "
            "test evaluation."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    metrics_path = (
        METRICS / f"03_deberta_fold{args.fold}_metrics.csv"
    )

    pd.DataFrame([{
        "Fold": args.fold,
        "TrainRecords": len(train),
        "HeldOutRecords": len(valid),
        **final_metrics,
    }]).to_csv(metrics_path, index=False)

    print("\nHELD-OUT FOLD RESULTS")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nSaved:")
    print(output_predictions)
    print(metrics_path)
    print(report_path)
    print("\nEXPERIMENT 03 FOLD", args.fold, "COMPLETED")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()

