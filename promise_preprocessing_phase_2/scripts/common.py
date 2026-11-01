
"""Shared utilities for PROMISE+ Phase 2 preprocessing."""

from pathlib import Path
import arff
import pandas as pd

# Directory locations
PHASE2_DIR = Path(__file__).resolve().parents[1]
THESIS_DIR = PHASE2_DIR.parent
PHASE1_DIR = THESIS_DIR / "promise_exploration_phase_1"

# Input from Phase 1
INPUT_PATH = (
    PHASE1_DIR
    / "outputs"
    / "validated"
    / "Promise+_repaired.arff"
)

# Output locations
OUTPUT_DIR = PHASE2_DIR / "outputs"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
SPLITS_DIR = OUTPUT_DIR / "splits"
REPORTS_DIR = OUTPUT_DIR / "reports"
TABLES_DIR = OUTPUT_DIR / "tables"

# Dataset columns
TEXT_COLUMN = "RequirementText"
LABEL_COLUMN = "_class_"
PROJECT_COLUMN = "ProjectID"

EXPECTED_COLUMNS = [
    PROJECT_COLUMN,
    TEXT_COLUMN,
    LABEL_COLUMN,
]


def prepare_outputs():
    """Create all output directories."""
    for directory in [
        CLEANED_DIR,
        SPLITS_DIR,
        REPORTS_DIR,
        TABLES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Load the validated PROMISE+ dataset."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Validated Phase 1 dataset not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        dataset = arff.load(file)

    columns = [name for name, _ in dataset["attributes"]]
    df = pd.DataFrame(dataset["data"], columns=columns)

    if columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema: {columns}"
        )

    return df
