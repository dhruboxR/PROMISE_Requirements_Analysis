
"""Shared utilities for PROMISE+ Phase 3 model development."""

from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

PHASE3_DIR = Path(__file__).resolve().parents[1]
THESIS_DIR = PHASE3_DIR.parent

PHASE2_DIR = THESIS_DIR / "promise_preprocessing_phase_2"

SPLITS_DIR = PHASE2_DIR / "outputs" / "splits"
MANIFEST_PATH = (
    PHASE2_DIR / "outputs" / "reports"
    / "08_final_split_manifest.json"
)

OUTPUT_DIR = PHASE3_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

SEED = 42


def prepare_outputs():
    for folder in [
        MODELS_DIR, METRICS_DIR,
        FIGURES_DIR, REPORTS_DIR, TABLES_DIR
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(name):
    """Read only the explicitly requested frozen split."""

    if name not in {"train", "validation"}:
        raise ValueError(
            "During development, load only train or validation."
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    path = SPLITS_DIR / f"final_{name}.csv"
    expected = manifest["splits"][name]["sha256"]

    if sha256(path) != expected:
        raise ValueError(
            f"Checksum mismatch for {name} split."
        )

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False,
    )

    if len(df) != manifest["splits"][name]["records"]:
        raise ValueError("Unexpected split size.")

    required = {
        "RecordID", "NormalizedText", "_class_", "ProjectID"
    }

    if not required.issubset(df.columns):
        raise ValueError("Missing required dataset columns.")

    if df["NormalizedText"].eq("").any():
        raise ValueError("Blank normalized requirements found.")

    if not set(df["_class_"]).issubset(set(LABELS)):
        raise ValueError("Unexpected classification label.")

    return df


def evaluate(y_true, y_pred):
    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    _, _, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


def full_report(y_true, y_pred):
    return classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )


def confusion_table(y_true, y_pred):
    matrix = confusion_matrix(
        y_true, y_pred, labels=LABELS
    )

    return pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    )
