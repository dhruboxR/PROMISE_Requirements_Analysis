
"""
Phase 8 Experiment 20
Sampling strategy ablation.

Train an ordinary shuffled-batch CE model and
compare with Experiment 19's class-aware models.

Same inner data, loss weights, backbone and seed.
No outer-fold evaluation or original test access.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

EXP19 = ROOT / "scripts/19_cross_project_contrastive_feasibility.py"

P4 = (
    ROOT.parent
    / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

MANIFEST = (
    ROOT.parent
    / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    exp = import_script(EXP19, "experiment19")
    phase4 = import_script(P4, "phase4_experiment20")

    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if (
        exp.SEED != 42
        or exp.OUTER_FOLD != 1
        or exp.BATCH_SIZE != 16
        or exp.EPOCHS != 4
        or exp.WEIGHT_EXPONENT != 0.25
        or phase4.LABELS != exp.LABELS
    ):
        raise ValueError("Experiment 19 configuration mismatch")

    previous_path = TABLES / "19_contrastive_inner_history.csv"
    if not previous_path.exists():
        raise FileNotFoundError(previous_path)

    previous = pd.read_csv(previous_path)

    if not {"CE_Only", "CE_SupCon"}.issubset(
        set(previous["Model"])
    ):
        raise ValueError("Experiment 19 comparison incomplete")

    data = phase4.load_development_data().copy()

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if len(data) != 3127 or data["RecordID"].duplicated().any():
        raise ValueError("Unexpected development data")

    labels = data["_class_"].astype(str).to_numpy()
    if set(labels) != set(exp.LABELS):
        raise ValueError("Unexpected labels")

    y = np.array([
        exp.LABELS.index(label) for label in labels
    ])

    groups = np.asarray(
        phase4.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        MANIFEST,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False
    )

    if (
        len(manifest) != 3127
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"]) != set(data["RecordID"])
    ):
        raise ValueError("Invalid fold manifest")

    aligned = manifest.set_index("RecordID").loc[
        data["RecordID"]
    ]

    if not np.array_equal(
        aligned["ActualClass"].to_numpy(), labels
    ):
        raise ValueError("Label mismatch")

    if not np.array_equal(
        aligned["ProjectID"].to_numpy(),
        data["ProjectID"].to_numpy()
    ):
        raise ValueError("Project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Unexpected outer folds")

    if (
        pd.DataFrame({"Group": groups, "Fold": folds})
        .groupby("Group")["Fold"].nunique() > 1
    ).any():
        raise ValueError("Connected-group overlap")

    outer_train = np.flatnonzero(
        folds != exp.OUTER_FOLD
    )

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=exp.SEED
    )

    selected = None

    for tr, va in cv.split(
        np.zeros(len(outer_train)),
        y[outer_train],
        groups[outer_train]
    ):
        tr_idx = outer_train[tr]
        va_idx = outer_train[va]

        if len(np.unique(y[tr_idx])) == len(exp.LABELS):
            selected = (tr_idx, va_idx)
            break

    if selected is None:
        raise ValueError("No suitable inner split")

    train_idx, valid_idx = selected

    if set(groups[train_idx]) & set(groups[valid_idx]):
        raise ValueError("Inner connected-group overlap")

    train_projects = set(data.iloc[train_idx]["ProjectID"])
    valid_projects = set(data.iloc[valid_idx]["ProjectID"])

    if train_projects & valid_projects:
        raise ValueError("Inner project overlap")

    if len(train_idx) != 1369 or len(valid_idx) != 790:
        raise ValueError(
            "Inner split does not match Experiment 19"
        )

    print("\nPHASE 8 EXPERIMENT 20")
    print("=" * 65)
    print("Inner training records:", len(train_idx))
    print("Inner validation records:", len(valid_idx))
    print("Group isolation: PASS")
    print("Outer-fold evaluation: No")

    selected_idx = np.concatenate([train_idx, valid_idx])

    texts = (
        data.iloc[selected_idx]["NormalizedText"]
        .astype(str).tolist()
    )

    selected_y = y[selected_idx]

    project_codes, _ = pd.factorize(
        data.iloc[selected_idx]["ProjectID"].to_numpy(),
        sort=True
    )

    exp.set_seed()

    tokenizer = exp.AutoTokenizer.from_pretrained(
        exp.MODEL_NAME
    )

    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=exp.MAX_LENGTH,
        return_tensors="pt"
    )

    n = len(train_idx)

    train_data = exp.RequirementDataset(
        {key: value[:n] for key, value in encoded.items()},
        selected_y[:n],
        project_codes[:n]
    )

    valid_data = exp.RequirementDataset(
        {key: value[n:] for key, value in encoded.items()},
        selected_y[n:],
        project_codes[n:]
    )

    # Ordinary shuffled batches, without forced class pairs.
    train_loader = DataLoader(
        train_data,
        batch_size=exp.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=exp.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Training ordinary shuffled CE model...")

    exp.set_seed()

    model = exp.MiniLMContrastiveClassifier().to(device)

    weights = exp.make_class_weights(
        selected_y[:n]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp.LR,
        weight_decay=0.01
    )

    rows = []

    for epoch in range(1, exp.EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for batch in train_loader:
            actual = batch["labels"].to(device)

            inputs = {
                name: value.to(device)
                for name, value in batch.items()
                if name not in ("labels", "projects")
            }

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(**inputs)

            loss = criterion(logits, actual)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += float(loss.item())

        scores = exp.evaluate(
            model, valid_loader, device
        )

        row = {
            "Model": "CE_Shuffled",
            "Epoch": epoch,
            "Accuracy": scores["Accuracy"],
            "MacroF1": scores["MacroF1"],
            "Recall_F": scores["Recall_F"],
            "Recall_L": scores["Recall_L"],
            "Recall_O": scores["Recall_O"],
            "Recall_SC": scores["Recall_SC"],
            "Recall_PO": scores["Recall_PO"],
            "MeanBatchCELoss":
                total_loss / len(train_loader)
        }

        rows.append(row)

        print(
            f"Epoch {epoch}: "
            f"Accuracy={row['Accuracy']:.4f} "
            f"MacroF1={row['MacroF1']:.4f}",
            flush=True
        )

    new_results = pd.DataFrame(rows)

    new_results.to_csv(
        TABLES / "20_shuffled_ce_history.csv",
        index=False
    )

    previous_final = previous[
        previous["Epoch"] == exp.EPOCHS
    ].copy()

    current_final = new_results[
        new_results["Epoch"] == exp.EPOCHS
    ].copy()

    comparison = pd.concat(
        [current_final, previous_final],
        ignore_index=True
    )

    comparison.to_csv(
        TABLES / "20_sampling_ablation_comparison.csv",
        index=False
    )

    report = {
        "experiment": "Phase 8 Experiment 20",
        "methods": [
            "Ordinary shuffled CE",
            "Class-aware CE",
            "Class-aware CE plus SupCon"
        ],
        "inner_training_records": len(train_idx),
        "inner_validation_records": len(valid_idx),
        "outer_fold_evaluated": False,
        "original_test_accessed": False,
        "original_labels_modified": False,
        "comparison_limitations": [
            "The inner split has been previously examined.",
            "Only one split and one random seed were used.",
            "Class-aware sampling changes effective class frequency."
        ]
    }

    (
        REPORTS / "20_sampling_ablation_metadata.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nEXPERIMENT 20 FINAL COMPARISON")
    print("=" * 65)

    print(
        comparison[
            [
                "Model", "Accuracy", "MacroF1",
                "Recall_F", "Recall_L",
                "Recall_O", "Recall_SC", "Recall_PO"
            ]
        ].to_string(index=False)
    )

    print("\nOriginal test accessed: No")
    print("Outer fold evaluated: No")
    print("Experiment 20 completed.")


if __name__ == "__main__":
    main()
