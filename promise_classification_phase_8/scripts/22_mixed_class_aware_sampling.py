
"""
Phase 8 Experiment 22
Mixed class-aware sampling evaluation.

75% ordinary sampling + 25% minority-focused sampling.

Uses the nine inner project-aware splits from Experiment 21.
Trains only the new mixed-sampling model.

No outer-fold predictions or original test access.
"""

from pathlib import Path
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader, Sampler
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

EXP19_PATH = (
    ROOT / "scripts/19_cross_project_contrastive_feasibility.py"
)

EXP21_PATH = (
    ROOT / "scripts/21_multisplit_sampling_evaluation.py"
)

P4_PATH = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

MANIFEST = (
    THESIS / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

SEED = 42
BATCH_SIZE = 16
ORDINARY_COUNT = 12
FOCUSED_COUNT = 4
EPOCHS = 4

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


class MixedBatchSampler(Sampler):
    """
    Each batch:
      12 ordinary examples from a shuffled stream
       4 focused examples:
         2 non-F classes x 2 examples

    Different source projects are preferred for
    focused same-class pairs.

    The epoch has ceil(N / 16) batches, matching
    Experiment 21's approximate update count.

    Ordinary samples can remain unseen in an epoch.
    Focused samples may repeat.
    """

    def __init__(self, labels, projects, seed):
        self.labels = np.asarray(labels, dtype=int)
        self.projects = np.asarray(projects, dtype=int)
        self.seed = int(seed)
        self.epoch = 0

        if len(self.labels) != len(self.projects):
            raise ValueError("Label/project length mismatch")

        if len(self.labels) == 0:
            raise ValueError("Empty training partition")

        self.steps = math.ceil(
            len(self.labels) / BATCH_SIZE
        )

        self.pools = {}

        for label in sorted(np.unique(self.labels)):
            if label == LABELS.index("F"):
                continue

            positions = np.flatnonzero(
                self.labels == label
            )

            by_project = {}

            for pos in positions:
                project = int(self.projects[pos])

                by_project.setdefault(
                    project, []
                ).append(int(pos))

            self.pools[int(label)] = by_project

        if len(self.pools) < 2:
            raise ValueError(
                "Need at least two non-F classes"
            )

        self.last_counts = None
        self.last_unique_count = None
        self.last_total_draws = None
        self.last_cross_project_pairs = None

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self.epoch
        )
        self.epoch += 1

        n = len(self.labels)

        permutation = rng.permutation(n)
        cursor = 0

        sampled = []
        cross_project_pairs = 0

        for _ in range(self.steps):
            ordinary = []

            while len(ordinary) < ORDINARY_COUNT:
                available = len(permutation) - cursor
                needed = ORDINARY_COUNT - len(ordinary)
                take = min(available, needed)

                ordinary.extend(
                    permutation[cursor:cursor + take].tolist()
                )

                cursor += take

                if cursor >= len(permutation):
                    permutation = rng.permutation(n)
                    cursor = 0

            selected_classes = rng.choice(
                list(self.pools),
                size=2,
                replace=False,
            )

            focused = []

            for label in selected_classes:
                project_pools = self.pools[int(label)]
                project_ids = list(project_pools)

                if len(project_ids) >= 2:
                    chosen_projects = rng.choice(
                        project_ids,
                        size=2,
                        replace=False,
                    )

                    for project in chosen_projects:
                        focused.append(
                            int(rng.choice(
                                project_pools[int(project)]
                            ))
                        )

                    cross_project_pairs += 1

                else:
                    only_pool = next(
                        iter(project_pools.values())
                    )

                    pair = rng.choice(
                        only_pool,
                        size=2,
                        replace=len(only_pool) < 2,
                    )

                    focused.extend(
                        int(pos) for pos in pair
                    )

            batch = ordinary + focused

            if len(batch) != BATCH_SIZE:
                raise ValueError("Invalid batch size")

            rng.shuffle(batch)
            sampled.extend(batch)

            yield batch

        sampled_array = np.asarray(
            sampled, dtype=int
        )

        self.last_counts = np.bincount(
            self.labels[sampled_array],
            minlength=len(LABELS),
        )

        self.last_unique_count = int(
            len(np.unique(sampled_array))
        )
        self.last_total_draws = int(
            len(sampled_array)
        )
        self.last_cross_project_pairs = int(
            cross_project_pairs
        )


def train_mixed(
    exp19,
    exp21,
    train_dataset,
    valid_dataset,
    train_labels,
    train_projects,
    device,
    seed,
):
    exp19.set_seed()

    sampler = MixedBatchSampler(
        train_labels,
        train_projects,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = exp19.MiniLMContrastiveClassifier().to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=exp21.make_weights(
            train_labels
        ).to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp19.LR,
        weight_decay=0.01,
    )

    history = []
    sampling_rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0

        for batch in train_loader:
            labels = batch["labels"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in ("labels", "projects")
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

        scores = exp21.evaluate(
            model, valid_loader, device
        )

        if sampler.last_counts is None:
            raise RuntimeError("Sampling statistics unavailable")

        history.append({
            "Model": "Mixed75_25CE",
            "Epoch": epoch,
            "TrainBatchLoss":
                total_loss / len(train_loader),
            **scores,
        })

        for index, label in enumerate(LABELS):
            sampling_rows.append({
                "Epoch": epoch,
                "Class": label,
                "SampledCount":
                    int(sampler.last_counts[index]),
                "SampledShare":
                    float(
                        sampler.last_counts[index]
                        / sampler.last_total_draws
                    ),
                "UniqueTrainingRecords":
                    sampler.last_unique_count,
                "TotalSampledDraws":
                    sampler.last_total_draws,
                "CrossProjectFocusedPairs":
                    sampler.last_cross_project_pairs,
            })

        print(
            f"    Epoch {epoch}: "
            f"Accuracy={scores['Accuracy']:.4f} "
            f"MacroF1={scores['MacroF1']:.4f} "
            f"RecallF={scores['Recall_F']:.4f} "
            f"SampledF={sampler.last_counts[0]}",
            flush=True,
        )

    del model, optimizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history, sampling_rows


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if ORDINARY_COUNT + FOCUSED_COUNT != BATCH_SIZE:
        raise ValueError("Invalid batch composition")

    exp19 = import_script(
        EXP19_PATH, "exp19_for22"
    )

    exp21 = import_script(
        EXP21_PATH, "exp21_for22"
    )

    phase4 = import_script(
        P4_PATH, "phase4_for22"
    )

    if (
        exp19.SEED != SEED
        or exp19.MODEL_NAME !=
            "sentence-transformers/all-MiniLM-L6-v2"
        or exp19.MAX_LENGTH != 128
        or exp19.LR != 2e-5
        or exp21.SEED != SEED
        or exp21.EPOCHS != EPOCHS
        or exp21.BATCH_SIZE != BATCH_SIZE
        or exp21.WEIGHT_EXPONENT != 0.25
        or phase4.LABELS != LABELS
    ):
        raise ValueError(
            "Experiment 21 configuration mismatch"
        )

    previous_path = (
        TABLES / "21_sampling_final_fold_metrics.csv"
    )

    if not previous_path.exists():
        raise FileNotFoundError(previous_path)

    previous = pd.read_csv(previous_path)

    expected_pairs = {
        (o, i, m)
        for o in [1, 2, 3]
        for i in [1, 2, 3]
        for m in ["ShuffledCE", "ClassAwareCE"]
    }

    observed_pairs = set(
        zip(
            previous["OuterFold"].astype(int),
            previous["InnerFold"].astype(int),
            previous["Model"],
        )
    )

    if (
        len(previous) != 18
        or observed_pairs != expected_pairs
        or not (previous["Epoch"] == EPOCHS).all()
    ):
        raise ValueError(
            "Previous Experiment 21 results incomplete"
        )

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected dataset size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs")

    labels = data["_class_"].astype(str).to_numpy()

    if set(labels) != set(LABELS):
        raise ValueError("Unexpected labels")

    y = np.array(
        [LABELS.index(label) for label in labels],
        dtype=int,
    )

    texts = data["NormalizedText"].astype(str).to_numpy()
    projects = data["ProjectID"].to_numpy()

    groups = np.asarray(
        phase4.create_connected_groups(data)
    )

    manifest = pd.read_csv(
        MANIFEST,
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
        raise ValueError("Fold manifest mismatch")

    aligned = manifest.set_index("RecordID").loc[
        data["RecordID"]
    ]

    if not np.array_equal(
        aligned["ActualClass"].to_numpy(), labels
    ):
        raise ValueError("Actual class mismatch")

    if not np.array_equal(
        aligned["ProjectID"].to_numpy(), projects
    ):
        raise ValueError("Project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Unexpected folds")

    if (
        pd.DataFrame({
            "Group": groups,
            "Fold": folds,
        })
        .groupby("Group")["Fold"].nunique() > 1
    ).any():
        raise ValueError("Outer group overlap")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nPHASE 8 EXPERIMENT 22")
    print("=" * 70)
    print("Device:", device)
    print("Batch composition: 12 ordinary + 4 focused")
    print("Paired inner comparisons: 9")
    print("New models to train: 9")
    print("Outer-fold evaluation: No")
    print("Original test accessed: No")

    exp19.set_seed()

    tokenizer = exp19.AutoTokenizer.from_pretrained(
        exp19.MODEL_NAME
    )

    all_rows = []
    sampling_rows = []

    for outer in [1, 2, 3]:
        outer_train = np.flatnonzero(folds != outer)

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

            if (
                set(groups[train_idx])
                & set(groups[valid_idx])
            ):
                raise ValueError("Inner group leakage")

            if (
                set(projects[train_idx])
                & set(projects[valid_idx])
            ):
                raise ValueError("Inner project leakage")

            print(
                f"\n  INNER FOLD {inner_number}: "
                f"Train={len(train_idx)} "
                f"Validation={len(valid_idx)}",
                flush=True,
            )

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

            project_codes, _ = pd.factorize(
                projects[selected],
                sort=True,
            )

            train_dataset = exp19.RequirementDataset(
                {
                    key: val[:n_train]
                    for key, val in encoded.items()
                },
                y[train_idx],
                project_codes[:n_train],
            )

            valid_dataset = exp19.RequirementDataset(
                {
                    key: val[n_train:]
                    for key, val in encoded.items()
                },
                y[valid_idx],
                project_codes[n_train:],
            )

            seed = SEED + outer * 100 + inner_number

            history, counts = train_mixed(
                exp19,
                exp21,
                train_dataset,
                valid_dataset,
                y[train_idx],
                project_codes[:n_train],
                device,
                seed,
            )

            for row in history:
                row.update({
                    "OuterFold": outer,
                    "InnerFold": inner_number,
                    "TrainRecords": len(train_idx),
                    "ValidationRecords": len(valid_idx),
                })

            for row in counts:
                row.update({
                    "OuterFold": outer,
                    "InnerFold": inner_number,
                })

            all_rows.extend(history)
            sampling_rows.extend(counts)

            # Preserve completed work between training runs.
            pd.DataFrame(all_rows).to_csv(
                TABLES / "22_mixed_epoch_history.csv",
                index=False,
            )

            pd.DataFrame(sampling_rows).to_csv(
                TABLES / "22_mixed_sampling_counts.csv",
                index=False,
            )

    history = pd.DataFrame(all_rows)

    final_mixed = history[
        history["Epoch"] == EPOCHS
    ].copy()

    if len(final_mixed) != 9:
        raise ValueError("Incomplete mixed-model results")

    comparison = pd.concat(
        [previous, final_mixed],
        ignore_index=True,
    )

    summary = comparison.groupby(
        "Model", as_index=False
    ).agg(
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

    paired = comparison.pivot(
        index=["OuterFold", "InnerFold"],
        columns="Model",
        values=["Accuracy", "MacroF1"],
    )

    paired_rows = []

    for (outer, inner), values in paired.iterrows():
        paired_rows.append({
            "OuterFold": outer,
            "InnerFold": inner,
            "MixedMinusShuffledAccuracy": float(
                values[("Accuracy", "Mixed75_25CE")]
                - values[("Accuracy", "ShuffledCE")]
            ),
            "MixedMinusShuffledMacroF1": float(
                values[("MacroF1", "Mixed75_25CE")]
                - values[("MacroF1", "ShuffledCE")]
            ),
            "MixedMinusClassAwareAccuracy": float(
                values[("Accuracy", "Mixed75_25CE")]
                - values[("Accuracy", "ClassAwareCE")]
            ),
            "MixedMinusClassAwareMacroF1": float(
                values[("MacroF1", "Mixed75_25CE")]
                - values[("MacroF1", "ClassAwareCE")]
            ),
        })

    paired_df = pd.DataFrame(paired_rows)

    comparison.to_csv(
        TABLES / "22_three_strategy_fold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        TABLES / "22_three_strategy_summary.csv",
        index=False,
    )

    paired_df.to_csv(
        TABLES / "22_mixed_paired_differences.csv",
        index=False,
    )

    metadata = {
        "experiment": "Phase 8 Experiment 22",
        "method": "75/25 mixed class-aware sampling",
        "batch_size": BATCH_SIZE,
        "ordinary_samples_per_batch": ORDINARY_COUNT,
        "focused_samples_per_batch": FOCUSED_COUNT,
        "focused_classes": "Two non-F classes per batch",
        "focused_pair_rule":
            "Prefer two different source projects",
        "epochs": EPOCHS,
        "weight_exponent": exp21.WEIGHT_EXPONENT,
        "new_training_runs": 9,
        "paired_inner_comparisons": 9,
        "outer_validation_evaluated": False,
        "original_test_accessed": False,
        "original_labels_modified": False,
        "limitations": [
            "The ratio was chosen after earlier development results.",
            "Inner folds are not independent test samples.",
            "Fewer unique natural records may be seen per epoch.",
            "Focused sampling uses replacement.",
            "Model selection is exploratory.",
        ],
    }

    (
        REPORTS / "22_mixed_sampling_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 22 SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\nPAIRED DIFFERENCES")
    print(paired_df.to_string(index=False))

    print("\nMixed macro-F1 wins vs shuffled:",
          int((paired_df[
              "MixedMinusShuffledMacroF1"
          ] > 0).sum()),
          "of 9")

    print("Mixed macro-F1 wins vs class-aware:",
          int((paired_df[
              "MixedMinusClassAwareMacroF1"
          ] > 0).sum()),
          "of 9")

    print("\nOriginal test accessed: No")
    print("Outer validation evaluated: No")
    print("Original datasets modified: No")
    print("Experiment 22 completed.")


if __name__ == "__main__":
    main()
