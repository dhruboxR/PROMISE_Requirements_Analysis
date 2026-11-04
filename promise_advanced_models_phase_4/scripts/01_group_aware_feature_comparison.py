
"""PROMISE+ Phase 4: Group-aware text feature comparison.

Uses ONLY Phase 2 train + validation data.
The original test set is never loaded.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


SEED = 42

ROOT = Path(__file__).resolve().parents[2]
PHASE4 = Path(__file__).resolve().parents[1]

PHASE2 = ROOT / "promise_preprocessing_phase_2"
SPLITS = PHASE2 / "outputs" / "splits"
MANIFEST = PHASE2 / "outputs" / "reports" / "08_final_split_manifest.json"

METRICS = PHASE4 / "outputs" / "metrics"
REPORTS = PHASE4 / "outputs" / "reports"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_development_data():
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    frames = []

    for name in ("train", "validation"):
        info = manifest["splits"][name]
        path = SPLITS / info["file"]

        if sha256(path) != info["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}")

        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={
                "RecordID": str,
                "ProjectID": str,
                "_class_": str,
            },
            keep_default_na=False,
        )

        if len(df) != info["records"]:
            raise ValueError(f"Wrong record count: {name}")

        if df["ProjectID"].nunique() != info["projects"]:
            raise ValueError(f"Wrong project count: {name}")

        if df["_class_"].nunique() != info["classes"]:
            raise ValueError(f"Wrong class count: {name}")

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    if len(data) != 3127:
        raise ValueError("Unexpected development dataset size")

    if data["RecordID"].duplicated().any():
        raise ValueError("Duplicate RecordIDs")

    if data["NormalizedText"].astype(str).str.strip().eq("").any():
        raise ValueError("Blank requirement text")

    if set(data["_class_"]) != set(LABELS):
        raise ValueError("Unexpected class labels")

    return data


def create_connected_groups(df):
    """Connect rows sharing project or normalized text."""

    n = len(df)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)

        if ra == rb:
            return

        if rank[ra] < rank[rb]:
            ra, rb = rb, ra

        parent[rb] = ra

        if rank[ra] == rank[rb]:
            rank[ra] += 1

    first_project = {}
    first_text = {}

    for i, row in enumerate(df.itertuples(index=False)):
        project = str(row.ProjectID)
        text = " ".join(
            str(row.NormalizedText).casefold().split()
        )

        if project in first_project:
            union(i, first_project[project])
        else:
            first_project[project] = i

        if text in first_text:
            union(i, first_text[text])
        else:
            first_text[text] = i

    roots = [find(i) for i in range(n)]
    group_ids = {
        root: j
        for j, root in enumerate(sorted(set(roots)))
    }

    return np.array([group_ids[root] for root in roots])


def make_features(kind):
    word = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        max_features=25000,
    )

    if kind == "word":
        return word

    character = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=40000,
    )

    return FeatureUnion([
        ("word", word),
        ("character", character),
    ])


def make_classifier(kind):
    if kind == "Linear SVM":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )

    return LogisticRegression(
        C=10.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=SEED,
    )


def main():
    METRICS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    data = load_development_data()

    X = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()
    groups = create_connected_groups(data)

    print("\nPROMISE+ PHASE 4: FEATURE COMPARISON")
    print("=" * 65)
    print("Development records:", len(data))
    print("Classes:", len(set(y)))
    print("Connected groups:", len(set(groups)))

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=SEED,
    )

    # Fix the same folds for every experiment.
    folds = list(cv.split(X, y, groups))

    fold_info = []

    for fold_no, (tr_idx, va_idx) in enumerate(folds, 1):
        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise ValueError("Connected-group leakage")

        train_labels = set(y[tr_idx])
        val_labels = set(y[va_idx])

        if train_labels != set(LABELS):
            raise ValueError(
                f"Fold {fold_no}: missing training classes"
            )

        missing_val = sorted(set(LABELS) - val_labels)

        info = {
            "fold": fold_no,
            "train_size": len(tr_idx),
            "validation_size": len(va_idx),
            "validation_classes": len(val_labels),
            "missing_validation_classes": missing_val,
        }

        fold_info.append(info)

        print(
            f"Fold {fold_no}: "
            f"train={len(tr_idx)}, "
            f"validation={len(va_idx)}, "
            f"missing validation classes={missing_val}"
        )

    configs = [
        ("Linear SVM", "word"),
        ("Linear SVM", "word+character"),
        ("Logistic Regression", "word"),
        ("Logistic Regression", "word+character"),
    ]

    results = []

    for model_name, feature_kind in configs:
        print(
            f"\nMODEL: {model_name} | FEATURES: {feature_kind}"
        )

        for fold_no, (tr_idx, va_idx) in enumerate(folds, 1):
            pipeline = Pipeline([
                ("features", make_features(feature_kind)),
                ("classifier", make_classifier(model_name)),
            ])

            # Entire pipeline fits on this fold's training data.
            pipeline.fit(X[tr_idx], y[tr_idx])
            predicted = pipeline.predict(X[va_idx])

            row = {
                "model": model_name,
                "features": feature_kind,
                "fold": fold_no,
                "train_records": len(tr_idx),
                "validation_records": len(va_idx),
                "accuracy": accuracy_score(
                    y[va_idx], predicted
                ),
                "macro_f1": f1_score(
                    y[va_idx],
                    predicted,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                ),
                "balanced_accuracy": balanced_accuracy_score(
                    y[va_idx], predicted
                ),
                "weighted_f1": f1_score(
                    y[va_idx],
                    predicted,
                    average="weighted",
                    zero_division=0,
                ),
            }

            results.append(row)

            print(
                f"Fold {fold_no} | "
                f"Accuracy={row['accuracy']:.4f} | "
                f"Macro-F1={row['macro_f1']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        METRICS / "01_feature_comparison_folds.csv",
        index=False,
    )

    summary = (
        results_df.groupby(
            ["model", "features"],
            as_index=False
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=(
                "balanced_accuracy", "mean"
            ),
            mean_weighted_f1=("weighted_f1", "mean"),
        )
        .sort_values("mean_macro_f1", ascending=False)
    )

    summary.to_csv(
        METRICS / "01_feature_comparison_summary.csv",
        index=False,
    )

    metadata = {
        "phase": 4,
        "experiment": "Group-aware feature comparison",
        "development_records": len(data),
        "connected_groups": len(set(groups)),
        "folds": fold_info,
        "random_seed": SEED,
        "selection_metric": "mean_macro_f1",
        "original_test_used": False,
        "scikit_learn_version": sklearn.__version__,
        "python_version": sys.version,
    }

    (
        REPORTS / "01_feature_comparison_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 65)
    print(summary.round(4).to_string(index=False))

    print("\nSaved experiment results.")
    print("Original 550-record test partition not accessed.")


if __name__ == "__main__":
    main()
