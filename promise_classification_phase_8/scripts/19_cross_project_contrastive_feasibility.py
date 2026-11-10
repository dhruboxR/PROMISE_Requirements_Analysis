
"""
Phase 8 Experiment 19
Cross-project supervised contrastive learning feasibility.

Compares:
A. Fine-tuned MiniLM with weighted cross-entropy.
B. Same classifier plus cross-project SupCon loss.

Both use identical class-aware batches and inner data.

No outer-fold evaluation.
No original test access.
No original labels modified.
"""

from pathlib import Path
import importlib.util
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

P4_SCRIPT = (
    THESIS / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

MANIFEST = (
    THESIS / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO",
]

SEED = 42
OUTER_FOLD = 1
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_EXPONENT = 0.25
CONTRASTIVE_WEIGHT = 0.1
TEMPERATURE = 0.1


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")

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
    def __init__(self, encodings, labels, projects):
        self.encodings = encodings
        self.labels = torch.as_tensor(
            labels, dtype=torch.long
        )
        self.projects = torch.as_tensor(
            projects, dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        result = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        result["labels"] = self.labels[index]
        result["projects"] = self.projects[index]
        return result


class CrossProjectBatchSampler(Sampler):
    """
    Creates 8 same-class pairs per batch.

    When possible, paired records come from
    distinct projects.

    This sampler deliberately changes the class
    distribution. The comparison baseline uses
    the exact same sampling policy.
    """

    def __init__(self, labels, projects, batch_size=16, seed=42):
        if batch_size != 16:
            raise ValueError("This experiment expects batch size 16")

        self.labels = np.asarray(labels)
        self.projects = np.asarray(projects)
        self.seed = seed
        self.epoch = 0
        self.steps = math.ceil(len(labels) / batch_size)

        self.pools = {}

        for label in sorted(np.unique(self.labels)):
            positions = np.flatnonzero(self.labels == label)

            by_project = {}

            for position in positions:
                project = int(self.projects[position])
                by_project.setdefault(project, []).append(
                    int(position)
                )

            self.pools[int(label)] = by_project

        if len(self.pools) < 8:
            raise ValueError("Insufficient classes for batch construction")

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self.epoch
        )
        self.epoch += 1

        available_labels = sorted(self.pools)

        for _ in range(self.steps):
            selected_classes = rng.choice(
                available_labels,
                size=8,
                replace=False,
            )

            batch = []

            for label in selected_classes:
                project_pools = self.pools[int(label)]
                project_ids = list(project_pools)

                if len(project_ids) >= 2:
                    projects = rng.choice(
                        project_ids, size=2, replace=False
                    )

                    for project in projects:
                        batch.append(int(rng.choice(
                            project_pools[int(project)]
                        )))
                else:
                    only_pool = next(iter(project_pools.values()))
                    chosen = rng.choice(
                        only_pool,
                        size=2,
                        replace=len(only_pool) < 2,
                    )
                    batch.extend(int(v) for v in chosen)

            rng.shuffle(batch)
            yield batch


class MiniLMContrastiveClassifier(nn.Module):
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

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(self, input_ids, attention_mask,
                token_type_ids=None):

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

        logits = self.classifier(
            self.dropout(pooled)
        )

        embeddings = F.normalize(
            self.projection(pooled),
            p=2,
            dim=1,
        )

        return logits, embeddings


def cross_project_supcon(embeddings, labels, projects):
    """
    Positives: same class, different project.
    Negatives: different class.

    Same-class, same-project records are excluded
    from the contrastive denominator to avoid
    treating them as negative examples.

    Anchors without positives contribute zero.
    """
    labels = labels.view(-1)
    projects = projects.view(-1)

    same_class = labels[:, None] == labels[None, :]
    same_project = projects[:, None] == projects[None, :]

    eye = torch.eye(
        len(labels),
        dtype=torch.bool,
        device=labels.device,
    )

    positive = same_class & ~same_project & ~eye

    denominator_mask = (
        (~same_class) | positive
    ) & ~eye

    logits = (
        embeddings @ embeddings.T
    ) / TEMPERATURE

    # Calculate in float32 for numerical stability.
    logits = logits.float()

    masked_logits = logits.masked_fill(
        ~denominator_mask, -1e4
    )

    log_denominator = torch.logsumexp(
        masked_logits, dim=1, keepdim=True
    )

    log_probabilities = logits - log_denominator

    positive_count = positive.sum(dim=1)
    valid_anchors = positive_count > 0

    if not valid_anchors.any():
        return embeddings.sum() * 0.0, 0

    positive_log_probability = (
        log_probabilities * positive.float()
    ).sum(dim=1) / positive_count.clamp(min=1)

    loss = -positive_log_probability[
        valid_anchors
    ].mean()

    return loss, int(valid_anchors.sum().item())


def make_class_weights(y):
    counts = np.bincount(
        y, minlength=len(LABELS)
    ).astype(float)

    if (counts == 0).any():
        raise ValueError("Missing class in inner training data")

    raw = (
        len(y) / (len(LABELS) * counts)
    ) ** WEIGHT_EXPONENT

    average = (counts * raw).sum() / len(y)

    return torch.tensor(
        raw / average, dtype=torch.float32
    )


def evaluate(model, loader, device):
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in ("labels", "projects")
            }

            logits, _ = model(**inputs)

            actual.extend(
                batch["labels"].tolist()
            )
            predicted.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    recalls = recall_score(
        actual,
        predicted,
        labels=list(range(len(LABELS))),
        average=None,
        zero_division=0,
    )

    results = {
        "Accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "MacroF1": float(
            f1_score(
                actual,
                predicted,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
    }

    for label in ["F", "L", "O", "SC", "PO"]:
        results[f"Recall_{label}"] = float(
            recalls[LABELS.index(label)]
        )

    return results


def run_candidate(
    mode, train_dataset, validation_loader,
    train_labels, train_projects, device,
):
    set_seed()

    sampler = CrossProjectBatchSampler(
        train_labels,
        train_projects,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
    )

    model = MiniLMContrastiveClassifier().to(device)

    weights = make_class_weights(
        train_labels
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
    )

    rows = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_ce = 0.0
        total_contrastive = 0.0
        total_valid_anchors = 0

        for batch in train_loader:
            labels = batch["labels"].to(device)
            projects = batch["projects"].to(device)

            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in ("labels", "projects")
            }

            optimizer.zero_grad(set_to_none=True)

            logits, embeddings = model(**inputs)

            ce_loss = criterion(logits, labels)

            if mode == "CE_SupCon":
                contrastive_loss, valid_anchors = (
                    cross_project_supcon(
                        embeddings, labels, projects
                    )
                )

                loss = (
                    ce_loss
                    + CONTRASTIVE_WEIGHT * contrastive_loss
                )
            else:
                contrastive_loss = ce_loss.new_zeros(())
                valid_anchors = 0
                loss = ce_loss

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            total_ce += float(ce_loss.item())
            total_contrastive += float(
                contrastive_loss.item()
            )
            total_valid_anchors += valid_anchors

        scores = evaluate(
            model, validation_loader, device
        )

        row = {
            "Model": mode,
            "Epoch": epoch,
            "MeanBatchCELoss": total_ce / len(train_loader),
            "MeanBatchSupConLoss":
                total_contrastive / len(train_loader),
            "ValidContrastiveAnchors":
                total_valid_anchors,
            **scores,
        }

        rows.append(row)

        print(
            f"{mode} Epoch {epoch}: "
            f"CE={row['MeanBatchCELoss']:.4f} "
            f"SupCon={row['MeanBatchSupConLoss']:.4f} "
            f"Accuracy={scores['Accuracy']:.4f} "
            f"MacroF1={scores['MacroF1']:.4f}",
            flush=True,
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    phase4 = load_module(
        P4_SCRIPT, "phase4_exp19"
    )

    if phase4.SEED != SEED or phase4.LABELS != LABELS:
        raise ValueError("Phase 4 configuration mismatch")

    data = phase4.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected dataset size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate record IDs")

    labels = data["_class_"].astype(str).to_numpy()

    if set(labels) != set(LABELS):
        raise ValueError("Unexpected labels")

    y = np.asarray([
        LABELS.index(label) for label in labels
    ])

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
        len(manifest) != 3127
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"])
            != set(data["RecordID"])
    ):
        raise ValueError("Invalid fold manifest")

    aligned = manifest.set_index("RecordID").loc[
        data["RecordID"]
    ]

    if not np.array_equal(
        aligned["ActualClass"].to_numpy(), labels
    ):
        raise ValueError("Manifest label mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        data["ProjectID"].to_numpy()
    ):
        raise ValueError("Manifest project mismatch")

    folds = aligned["Fold"].astype(int).to_numpy()

    if set(folds) != {1, 2, 3}:
        raise ValueError("Unexpected folds")

    if (
        pd.DataFrame({"Group": groups, "Fold": folds})
        .groupby("Group")["Fold"].nunique() > 1
    ).any():
        raise ValueError("Outer connected-group leakage")

    outer_training = np.flatnonzero(
        folds != OUTER_FOLD
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    chosen_split = None

    for tr, va in inner_cv.split(
        np.zeros(len(outer_training)),
        y[outer_training],
        groups[outer_training],
    ):
        train_idx = outer_training[tr]
        valid_idx = outer_training[va]

        if len(np.unique(y[train_idx])) == len(LABELS):
            chosen_split = train_idx, valid_idx
            break

    if chosen_split is None:
        raise ValueError("No valid inner training split")

    train_idx, valid_idx = chosen_split

    if (
        set(groups[train_idx]) & set(groups[valid_idx])
    ):
        raise ValueError("Inner connected-group leakage")

    if (
        set(data.iloc[train_idx]["ProjectID"])
        & set(data.iloc[valid_idx]["ProjectID"])
    ):
        raise ValueError("Inner project leakage")

    print("\nPHASE 8 EXPERIMENT 19")
    print("=" * 70)

    print("Inner training records:", len(train_idx))
    print("Inner validation records:", len(valid_idx))
    print("Inner training projects:",
          data.iloc[train_idx]["ProjectID"].nunique())
    print("Inner validation projects:",
          data.iloc[valid_idx]["ProjectID"].nunique())

    print("Original test accessed: No")
    print("Outer-fold evaluation: No")

    # Select only inner-training and inner-validation texts.
    # The outer-held-out examples are never tokenized here.
    selected_idx = np.concatenate(
        [train_idx, valid_idx]
    )

    selected_texts = (
        data.iloc[selected_idx]["NormalizedText"]
        .astype(str).tolist()
    )

    selected_y = y[selected_idx]

    selected_projects_raw = (
        data.iloc[selected_idx]["ProjectID"]
        .astype(str).to_numpy()
    )

    project_codes, _ = pd.factorize(
        selected_projects_raw, sort=True
    )

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    encodings = tokenizer(
        selected_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_size = len(train_idx)

    train_encodings = {
        key: value[:train_size]
        for key, value in encodings.items()
    }

    valid_encodings = {
        key: value[train_size:]
        for key, value in encodings.items()
    }

    train_dataset = RequirementDataset(
        train_encodings,
        selected_y[:train_size],
        project_codes[:train_size],
    )

    valid_dataset = RequirementDataset(
        valid_encodings,
        selected_y[train_size:],
        project_codes[train_size:],
    )

    validation_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = []

    for mode in ["CE_Only", "CE_SupCon"]:
        print(f"\nTRAINING: {mode}", flush=True)

        result = run_candidate(
            mode,
            train_dataset,
            validation_loader,
            selected_y[:train_size],
            project_codes[:train_size],
            device,
        )

        results.extend(result)

    history = pd.DataFrame(results)

    history.to_csv(
        TABLES / "19_contrastive_inner_history.csv",
        index=False,
    )

    final_epoch = history[
        history["Epoch"] == EPOCHS
    ].copy()

    final_epoch.to_csv(
        TABLES / "19_contrastive_final_comparison.csv",
        index=False,
    )

    best_epoch = (
        history.sort_values(
            ["Model", "MacroF1", "Epoch"],
            ascending=[True, False, True],
        )
        .groupby("Model", as_index=False)
        .head(1)
        .sort_values("Model")
    )

    best_epoch.to_csv(
        TABLES / "19_contrastive_best_inner_epochs.csv",
        index=False,
    )

    metadata = {
        "experiment": "Phase 8 Experiment 19",
        "method": "Cross-project supervised contrastive feasibility",
        "training_records": int(len(train_idx)),
        "validation_records": int(len(valid_idx)),
        "modes": ["CE_Only", "CE_SupCon"],
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_exponent": WEIGHT_EXPONENT,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "temperature": TEMPERATURE,
        "sampling": "8 classes x 2 examples",
        "positive_pair_rule": "same class, different project",
        "heldout_outer_evaluated": False,
        "original_test_accessed": False,
        "labels_modified": False,
        "experiment_exploratory": True,
        "selection_bias_warning":
            "The inner split has been examined previously. "
            "Results are feasibility findings, not independent evidence.",
    }

    (
        REPORTS / "19_contrastive_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 19 FINAL-EPOCH COMPARISON")
    print("=" * 70)

    print(
        final_epoch[
            [
                "Model", "Epoch",
                "Accuracy", "MacroF1",
                "Recall_F", "Recall_L",
                "Recall_O", "Recall_SC", "Recall_PO",
            ]
        ].to_string(index=False)
    )

    print("\nBEST INNER EPOCHS")
    print(
        best_epoch[
            ["Model", "Epoch", "Accuracy", "MacroF1"]
        ].to_string(index=False)
    )

    print("\nOriginal test accessed: No")
    print("Outer fold evaluated: No")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
