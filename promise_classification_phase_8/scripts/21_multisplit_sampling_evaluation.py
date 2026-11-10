
"""
Phase 8 Experiment 21
Multi-split project-aware sampling evaluation.

For each of 3 outer-training partitions:
    Run 3 inner group-aware validation folds.
    Compare shuffled CE against class-aware CE.

Total: 9 paired comparisons / 18 model trainings.

No outer-fold predictions generated.
No original test records accessed.
No labels changed.
"""

from pathlib import Path
import importlib.util
import json
import math
import random

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, Sampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

EXP19 = ROOT / "scripts/19_cross_project_contrastive_feasibility.py"

P4_SCRIPT = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

MANIFEST = (
    THESIS / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

SEED = 42
EPOCHS = 4
BATCH_SIZE = 16
WEIGHT_EXPONENT = 0.25

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FlexibleClassAwareSampler(Sampler):
    """
    Produces batches of 8 class pairs.

    Prefer different projects for each pair.
    If a class exists in only one project, use that project.

    Can operate when fewer than 8 classes are represented.
    Samples with replacement, so records may repeat.

    Exact same batch generation is deterministic for a
    given seed, outer fold, inner fold, and epoch.
    """

    def __init__(self, labels, projects, seed):
        self.labels = np.asarray(labels)
        self.projects = np.asarray(projects)
        self.seed = seed
        self.epoch = 0

        self.steps = math.ceil(len(labels) / BATCH_SIZE)
        self.pools = {}

        for c in np.unique(self.labels):
            positions = np.flatnonzero(self.labels == c)
            project_pools = {}

            for pos in positions:
                project = int(self.projects[pos])
                project_pools.setdefault(project, []).append(
                    int(pos)
                )

            self.pools[int(c)] = project_pools

        if len(self.pools) < 2:
            raise ValueError("Need at least two training classes")

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self.epoch
        )
        self.epoch += 1

        classes = sorted(self.pools)

        for _ in range(self.steps):
            chosen_classes = rng.choice(
                classes,
                size=8,
                replace=len(classes) < 8,
            )

            batch = []

            for c in chosen_classes:
                project_pools = self.pools[int(c)]
                project_ids = list(project_pools)

                if len(project_ids) >= 2:
                    chosen_projects = rng.choice(
                        project_ids,
                        size=2,
                        replace=False,
                    )

                    for p in chosen_projects:
                        batch.append(int(rng.choice(
                            project_pools[int(p)]
                        )))
                else:
                    single_pool = next(
                        iter(project_pools.values())
                    )

                    pair = rng.choice(
                        single_pool,
                        size=2,
                        replace=len(single_pool) < 2,
                    )
                    batch.extend(int(i) for i in pair)

            rng.shuffle(batch)
            yield batch


def make_weights(y):
    """
    Supports missing classes in an inner training fold.
    Classes without training records receive zero weight.
    """
    counts = np.bincount(
        y, minlength=len(LABELS)
    ).astype(float)

    weights = np.zeros(len(LABELS), dtype=float)
    present = counts > 0

    n = len(y)
    k_present = int(present.sum())

    weights[present] = (
        n / (k_present * counts[present])
    ) ** WEIGHT_EXPONENT

    average = (
        counts[present] * weights[present]
    ).sum() / n

    weights[present] /= average

    return torch.tensor(weights, dtype=torch.float32)


def calculate_scores(actual, predicted):
    label_ids = list(range(len(LABELS)))

    recalls = recall_score(
        actual,
        predicted,
        labels=label_ids,
        average=None,
        zero_division=0,
    )

    result = {
        "Accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "MacroF1": float(
            f1_score(
                actual,
                predicted,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
    }

    for label in LABELS:
        result[f"Recall_{label}"] = float(
            recalls[LABELS.index(label)]
        )

    return result


def evaluate(model, loader, device):
    model.eval()
    actual = []
    predicted = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                k: value.to(device)
                for k, value in batch.items()
                if k not in ("labels", "projects")
            }

            logits, _ = model(**inputs)

            actual.extend(batch["labels"].tolist())
            predicted.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

    return calculate_scores(
        np.asarray(actual),
        np.asarray(predicted),
    )


def train_one(
    exp19, mode, train_dataset, valid_dataset,
    train_y, train_projects, device, seed,
):
    exp19.set_seed()

    if mode == "ClassAwareCE":
        sampler = FlexibleClassAwareSampler(
            train_y, train_projects, seed=seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=0,
        )
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = exp19.MiniLMContrastiveClassifier().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=make_weights(train_y).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp19.LR,
        weight_decay=0.01,
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels = batch["labels"].to(device)

            inputs = {
                k: value.to(device)
                for k, value in batch.items()
                if k not in ("labels", "projects")
            }

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(**inputs)
            loss = criterion(logits, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            optimizer.step()

            total_loss += float(loss.item())

        scores = evaluate(model, valid_loader, device)

        history.append({
            "Model": mode,
            "Epoch": epoch,
            "TrainBatchLoss": total_loss / len(train_loader),
            **scores,
        })

        print(
            f"    {mode} Epoch {epoch}: "
            f"Acc={scores['Accuracy']:.4f} "
            f"MacroF1={scores['MacroF1']:.4f}",
            flush=True,
        )

    del model, optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    exp19 = import_script(EXP19, "exp19_for21")
    phase4 = import_script(P4_SCRIPT, "phase4_for21")

    if (
        exp19.SEED != SEED
        or exp19.MODEL_NAME
            != "sentence-transformers/all-MiniLM-L6-v2"
        or exp19.MAX_LENGTH != 128
        or exp19.LR != 2e-5
        or phase4.SEED != SEED
        or phase4.LABELS != LABELS
    ):
        raise ValueError("Configuration mismatch")

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordID")

    y_labels = data["_class_"].astype(str).to_numpy()

    if set(y_labels) != set(LABELS):
        raise ValueError("Unexpected labels")

    y = np.array(
        [LABELS.index(c) for c in y_labels],
        dtype=int,
    )

    texts = data["NormalizedText"].astype(str).to_numpy()
    projects = data["ProjectID"].to_numpy()
    groups = np.asarray(
        phase4.create_connected_groups(data)
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
        aligned["ActualClass"].to_numpy(), y_labels
    ):
        raise ValueError("Manifest labels mismatch")

    if not np.array_equal(
        aligned["ProjectID"].to_numpy(), projects
    ):
        raise ValueError("Manifest projects mismatch")

    outer_folds = aligned["Fold"].astype(int).to_numpy()

    if set(outer_folds) != {1, 2, 3}:
        raise ValueError("Invalid outer folds")

    if (
        pd.DataFrame({
            "Group": groups,
            "Fold": outer_folds,
        })
        .groupby("Group")["Fold"].nunique() > 1
    ).any():
        raise ValueError("Outer group leakage")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nPHASE 8 EXPERIMENT 21")
    print("=" * 70)
    print("Device:", device)
    print("Outer training partitions: 3")
    print("Inner folds per partition: 3")
    print("Models per inner fold: 2")
    print("Original test accessed: No")
    print("Outer validation evaluated: No")

    # Tokenization is fixed and label-independent.
    # No fitting of text statistics is performed.
    exp19.set_seed()

    tokenizer = exp19.AutoTokenizer.from_pretrained(
        exp19.MODEL_NAME
    )

    all_history = []
    support_rows = []

    for outer in [1, 2, 3]:
        outer_train = np.flatnonzero(
            outer_folds != outer
        )

        inner_cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED,
        )

        print(f"\nOUTER TRAINING PARTITION {outer}")

        for inner_number, (tr, va) in enumerate(
            inner_cv.split(
                np.zeros(len(outer_train)),
                y[outer_train],
                groups[outer_train],
            ),
            start=1,
        ):
            train_idx = outer_train[tr]
            valid_idx = outer_train[va]

            if set(groups[train_idx]) & set(groups[valid_idx]):
                raise ValueError("Inner connected-group leakage")

            if set(projects[train_idx]) & set(projects[valid_idx]):
                raise ValueError("Inner project leakage")

            if len(np.unique(y[train_idx])) < 2:
                raise ValueError("Insufficient training classes")

            print(
                f"\n  INNER FOLD {inner_number}: "
                f"Train={len(train_idx)} "
                f"Validation={len(valid_idx)}",
                flush=True,
            )

            for label in LABELS:
                label_id = LABELS.index(label)

                support_rows.append({
                    "OuterFold": outer,
                    "InnerFold": inner_number,
                    "Class": label,
                    "TrainCount": int(
                        np.sum(y[train_idx] == label_id)
                    ),
                    "ValidationCount": int(
                        np.sum(y[valid_idx] == label_id)
                    ),
                    "TrainProjects": int(
                        len(set(projects[
                            train_idx[y[train_idx] == label_id]
                        ]))
                    ),
                    "ValidationProjects": int(
                        len(set(projects[
                            valid_idx[y[valid_idx] == label_id]
                        ]))
                    ),
                })

            # Tokenize only the current training/validation pair.
            selected = np.concatenate(
                [train_idx, valid_idx]
            )

            encoded = tokenizer(
                texts[selected].tolist(),
                truncation=True,
                padding="max_length",
                max_length=exp19.MAX_LENGTH,
                return_tensors="pt",
            )

            n_train = len(train_idx)

            train_enc = {
                k: v[:n_train]
                for k, v in encoded.items()
            }
            valid_enc = {
                k: v[n_train:]
                for k, v in encoded.items()
            }

            # Project IDs for contrast-aware pair sampling.
            project_codes, _ = pd.factorize(
                projects[selected],
                sort=True,
            )

            train_dataset = exp19.RequirementDataset(
                train_enc,
                y[train_idx],
                project_codes[:n_train],
            )

            valid_dataset = exp19.RequirementDataset(
                valid_enc,
                y[valid_idx],
                project_codes[n_train:],
            )

            run_seed = (
                SEED + outer * 100 + inner_number
            )

            for mode in ["ShuffledCE", "ClassAwareCE"]:
                run_history = train_one(
                    exp19,
                    mode,
                    train_dataset,
                    valid_dataset,
                    y[train_idx],
                    project_codes[:n_train],
                    device,
                    seed=run_seed,
                )

                for row in run_history:
                    row["OuterFold"] = outer
                    row["InnerFold"] = inner_number
                    row["TrainRecords"] = len(train_idx)
                    row["ValidationRecords"] = len(valid_idx)

                all_history.extend(run_history)

    history = pd.DataFrame(all_history)
    support = pd.DataFrame(support_rows)

    # Use the same predefined epoch for both methods.
    final = history[
        history["Epoch"] == EPOCHS
    ].copy()

    if (
        len(final) != 18
        or final.groupby(
            ["OuterFold", "InnerFold", "Model"]
        ).size().max() != 1
    ):
        raise ValueError("Incomplete experiment results")

    summary = (
        final.groupby("Model", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            StdAccuracy=("Accuracy", "std"),
            MeanMacroF1=("MacroF1", "mean"),
            StdMacroF1=("MacroF1", "std"),
            MeanRecallF=("Recall_F", "mean"),
            MeanRecallL=("Recall_L", "mean"),
            MeanRecallO=("Recall_O", "mean"),
            MeanRecallSC=("Recall_SC", "mean"),
            MeanRecallPO=("Recall_PO", "mean"),
        )
    )

    paired = final.pivot(
        index=["OuterFold", "InnerFold"],
        columns="Model",
        values=["Accuracy", "MacroF1"],
    )

    paired_rows = []

    for (outer, inner), values in paired.iterrows():
        paired_rows.append({
            "OuterFold": outer,
            "InnerFold": inner,
            "AccuracyDifference": float(
                values[("Accuracy", "ClassAwareCE")]
                - values[("Accuracy", "ShuffledCE")]
            ),
            "MacroF1Difference": float(
                values[("MacroF1", "ClassAwareCE")]
                - values[("MacroF1", "ShuffledCE")]
            ),
        })

    paired_df = pd.DataFrame(paired_rows)

    history.to_csv(
        TABLES / "21_sampling_epoch_history.csv",
        index=False,
    )

    final.to_csv(
        TABLES / "21_sampling_final_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        TABLES / "21_sampling_summary.csv",
        index=False,
    )

    support.to_csv(
        TABLES / "21_inner_class_support.csv",
        index=False,
    )

    paired_df.to_csv(
        TABLES / "21_sampling_paired_differences.csv",
        index=False,
    )

    metadata = {
        "experiment": "Phase 8 Experiment 21",
        "title": "Multi-Split Sampling Evaluation",
        "outer_training_partitions": 3,
        "inner_folds_per_partition": 3,
        "paired_comparisons": 9,
        "training_runs": 18,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "weight_exponent": WEIGHT_EXPONENT,
        "learning_rate": exp19.LR,
        "selection_rule": "Predefined final epoch 4",
        "class_aware_sampling":
            "Eight class pairs per batch; prefer different projects",
        "outer_validation_evaluated": False,
        "original_test_accessed": False,
        "original_labels_modified": False,
        "independent_test_evidence": False,
        "limitations": [
            "Inner folds reuse development records.",
            "Nine paired results are not nine independent test sets.",
            "The development protocol was informed by earlier experiments.",
            "Sampling with replacement changes effective class frequencies.",
            "Fixed learning rate and epoch count were not independently tuned.",
        ],
    }

    (
        REPORTS / "21_sampling_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 21 SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\nPAIRED DIFFERENCES")
    print(paired_df.to_string(index=False))

    print("\nMean paired accuracy difference:",
          paired_df["AccuracyDifference"].mean())

    print("Mean paired macro-F1 difference:",
          paired_df["MacroF1Difference"].mean())

    print("\nOriginal test accessed: No")
    print("Outer-fold evaluation performed: No")
    print("Original datasets modified: No")
    print("Experiment 21 completed.")


if __name__ == "__main__":
    main()

