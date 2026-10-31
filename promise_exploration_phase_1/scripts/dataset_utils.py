"""Shared PROMISE+ dataset loading and column discovery utilities."""
from pathlib import Path
import arff
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / 'dataset' / 'Promise+.arff'
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'outputs'


def load_dataset():
    """Load ARFF (including string attributes) without modifying original data."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f'Missing dataset: {DATA_FILE}')
    with DATA_FILE.open('r', encoding='utf-8-sig', errors='replace') as f:
        obj = arff.load(f)
    columns = [name for name, _ in obj['attributes']]
    df = pd.DataFrame(obj['data'], columns=columns)
    return df, obj


def discover_columns(df, metadata):
    """Suggest text and label columns; never silently assume column order."""
    names = list(df.columns)
    attrs = dict(metadata['attributes'])
    text_hints = ('requirement', 'sentence', 'description', 'text', 'statement', 'content')
    label_hints = ('class', 'label', 'type', 'category')
    text_candidates = [x for x in names if any(h in x.lower() for h in text_hints)]
    label_candidates = [x for x in names if any(h in x.lower() for h in label_hints)]
    # ARFF nominal attributes often represent class labels.
    nominal_candidates = [x for x in names if isinstance(attrs.get(x), list)]
    return text_candidates, label_candidates or nominal_candidates


def output_path(folder, filename):
    destination = OUTPUT_ROOT / folder / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
