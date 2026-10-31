
"""Shared utilities for PROMISE+ exploratory analysis."""

from pathlib import Path
import arff
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
THESIS_DIR = PROJECT_DIR.parent

# Original dataset (preserved without modification)
ORIGINAL_DATA_PATH = THESIS_DIR / "dataset" / "Promise+.arff"

# Validated and repaired dataset used for exploration
DATA_PATH = PROJECT_DIR / "outputs" / "validated" / "Promise+_repaired.arff"

# Output directories
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"


def prepare_outputs():
    for p in (FIGURES_DIR, TABLES_DIR, REPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}\nKeep Promise+.arff in the thesis dataset/ folder.')
    with DATA_PATH.open('r', encoding='utf-8-sig', errors='replace') as f:
        dataset = arff.load(f)
    columns = [a[0] for a in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=columns)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(lambda x: x.decode('utf-8', errors='replace') if isinstance(x, bytes) else x)
    return df


def text_candidates(df: pd.DataFrame):
    """Heuristic ranking only; validate actual schema before concluding."""
    result = []
    for col in df.select_dtypes(include=['object', 'string']).columns:
        values = df[col].dropna().astype(str)
        avg_length = values.str.len().mean() if not values.empty else 0
        result.append((col, float(avg_length)))
    return sorted(result, key=lambda item: item[1], reverse=True)


def choose_text_column(df: pd.DataFrame):
    candidates = text_candidates(df)
    return candidates[0][0] if candidates else None


def choose_label_column(df: pd.DataFrame, text_col=None):
    # Favor known class names; otherwise choose a low-cardinality column.
    likely = ('class', 'label', 'type', 'category', 'requirement_type', 'classification')
    lookup = {str(c).lower().strip(): c for c in df.columns}
    for name in likely:
        if name in lookup and lookup[name] != text_col:
            return lookup[name]
    options = []
    for col in df.columns:
        if col == text_col:
            continue
        n = df[col].nunique(dropna=True)
        if 1 < n <= min(30, max(2, len(df) // 3)):
            options.append((n, col))
    return sorted(options, key=lambda x: x[0])[0][1] if options else None
