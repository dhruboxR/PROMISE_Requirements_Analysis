
"""
Phase 8 Experiment 07
Nested lexical-semantic feature-fusion comparison.

Outer: Phase 5 fixed project-aware folds.
Inner: group-aware weight selection using macro-F1.

Reuses Phase 4 TF-IDF and frozen MiniLM encoding.

No original test data accessed.
"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import torch

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
P4 = THESIS / "promise_advanced_models_phase_4"
P5 = THESIS / "promise_improved_model_phase_5"

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO",
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_for(y):
    counts = pd.Series(y).value_counts()
    n = len(y)
    k = len(counts)

    raw = {
        c: np.sqrt(n / (k * counts[c]))
        for c in counts.index
    }

    normalizer = sum(
        counts[c] * raw[c] for c in counts.index
    ) / n

    class_weights = {
        c: float(raw[c] / normalizer)
        for c in counts.index
    }

    return LogisticRegression(
        C=10.0,
        class_weight=class_weights,
        max_iter=2000,
        random_state=SEED,
    )


def features(lexical, embeddings, weight):
    if weight == 0:
        return lexical

    return hstack(
        [lexical, csr_matrix(embeddings * weight)],
        format="csr",
    )


def metrics(actual, pred):
    recalls = recall_score(
        actual, pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    result = {
        "Accuracy": accuracy_score(actual, pred),
        "MacroF1": f1_score(
            actual, pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "BalancedAccuracy": balanced_accuracy_score(
            actual, pred
        ),
    }

    for c in ["F", "O", "SC", "PO"]:
        result[f"Recall_{c}"] = recalls[
            LABELS.index(c)
        ]

    return {
        key: float(value)
        for key, value in result.items()
    }


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    cv = load_module(
        P4 / "scripts/01_group_aware_feature_comparison.py",
        "phase4_cv07",
    )
    semantic = load_module(
        P4 / "scripts/17_semantic_embedding_logistic_cv.py",
        "phase4_semantic07",
    )

    if (
        cv.SEED != SEED
        or cv.LABELS != LABELS
        or semantic.MODEL_NAME != MODEL_NAME
        or semantic.MAX_LENGTH != 128
    ):
        raise ValueError("Previous model settings mismatch")

    data = cv.load_development_data().copy()

    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    texts = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()
    groups = np.asarray(cv.create_connected_groups(data))

    manifest = pd.read_csv(
        P5 / "outputs/tables/01_fixed_fold_assignments.csv",
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or data["RecordID"].duplicated().any()
        or manifest["RecordID"].duplicated().any()
        or set(manifest["RecordID"]) != set(data["RecordID"])
    ):
        raise ValueError("Invalid fixed fold manifest")

    aligned = manifest.set_index("RecordID").loc[
        data["RecordID"]
    ]

    if not np.array_equal(
        aligned["ActualClass"].to_numpy(), y
    ):
        raise ValueError("Class mismatch")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        data["ProjectID"].to_numpy(),
    ):
        raise ValueError("Project mismatch")

    outer_folds = aligned["Fold"].astype(int).to_numpy()

    if set(outer_folds) != {1, 2, 3}:
        raise ValueError("Unexpected outer folds")

    group_fold_count = pd.DataFrame({
        "Group": groups,
        "Fold": outer_folds,
    }).groupby("Group")["Fold"].nunique()

    if (group_fold_count > 1).any():
        raise ValueError("Connected group leakage")

    semantic.set_seed()

    from transformers import AutoTokenizer, AutoModel

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Generating frozen MiniLM embeddings...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)
    encoder.eval()

    embeddings = semantic.generate_embeddings(
        texts.tolist(), tokenizer, encoder, device
    )

    if (
        embeddings.shape != (3127, 384)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid embeddings")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fold_rows = []
    inner_rows = []
    prediction_rows = []

    for outer in [1, 2, 3]:
        print(f"\nOUTER FOLD {outer}", flush=True)

        tr_idx = np.flatnonzero(outer_folds != outer)
        va_idx = np.flatnonzero(outer_folds == outer)

        if set(groups[tr_idx]) & set(groups[va_idx]):
            raise ValueError("Outer group overlap")

        inner_cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED,
        )

        inner_scores = {w: [] for w in WEIGHTS}

        for number, (inner_tr, inner_va) in enumerate(
            inner_cv.split(
                texts[tr_idx],
                y[tr_idx],
                groups[tr_idx],
            ), start=1
        ):
            print(f"  Inner fold {number}", flush=True)

            train_ids = tr_idx[inner_tr]
            valid_ids = tr_idx[inner_va]

            if set(groups[train_ids]) & set(groups[valid_ids]):
                raise ValueError("Inner group overlap")

            vectorizer = cv.make_features("word+character")

            lexical_train = vectorizer.fit_transform(
                texts[train_ids]
            )
            lexical_valid = vectorizer.transform(
                texts[valid_ids]
            )

            for w in WEIGHTS:
                x_train = features(
                    lexical_train, embeddings[train_ids], w
                )
                x_valid = features(
                    lexical_valid, embeddings[valid_ids], w
                )

                clf = model_for(y[train_ids])
                clf.fit(x_train, y[train_ids])
                pred = clf.predict(x_valid)

                result = metrics(y[valid_ids], pred)
                inner_scores[w].append(result)

                inner_rows.append({
                    "OuterFold": outer,
                    "InnerFold": number,
                    "Weight": w,
                    **result,
                })

        mean_f1 = {
            w: float(np.mean([
                r["MacroF1"] for r in inner_scores[w]
            ]))
            for w in WEIGHTS
        }

        # Tie-break toward the original weight 1.0.
        selected = sorted(
            WEIGHTS,
            key=lambda w: (-mean_f1[w], abs(w - 1.0)),
        )[0]

        print("  Inner mean macro-F1:", mean_f1)
        print("  Selected weight:", selected)

        vectorizer = cv.make_features("word+character")
        lex_train = vectorizer.fit_transform(texts[tr_idx])
        lex_valid = vectorizer.transform(texts[va_idx])

        for name, w in [
            ("BaselineWeight1", 1.0),
            ("NestedFusion", selected),
        ]:
            x_train = features(
                lex_train, embeddings[tr_idx], w
            )
            x_valid = features(
                lex_valid, embeddings[va_idx], w
            )

            clf = model_for(y[tr_idx])
            clf.fit(x_train, y[tr_idx])
            pred = clf.predict(x_valid)
            result = metrics(y[va_idx], pred)

            fold_rows.append({
                "Model": name,
                "Fold": outer,
                "Weight": w,
                **result,
            })

            for idx, prediction in zip(va_idx, pred):
                prediction_rows.append({
                    "RecordID": data.iloc[idx]["RecordID"],
                    "Fold": outer,
                    "Model": name,
                    "ActualClass": y[idx],
                    "PredictedClass": prediction,
                })

            print(
                f"  {name}: Weight={w} "
                f"Accuracy={result['Accuracy']:.4f} "
                f"MacroF1={result['MacroF1']:.4f}",
                flush=True,
            )

    results = pd.DataFrame(fold_rows)

    summary = results.groupby(
        "Model", as_index=False
    ).agg(
        MeanAccuracy=("Accuracy", "mean"),
        MeanMacroF1=("MacroF1", "mean"),
        MeanBalancedAccuracy=("BalancedAccuracy", "mean"),
        MeanRecallO=("Recall_O", "mean"),
        MeanRecallSC=("Recall_SC", "mean"),
        MeanRecallPO=("Recall_PO", "mean"),
    )

    baseline = summary.loc[
        summary["Model"] == "BaselineWeight1"
    ].iloc[0]

    reproduced = (
        abs(baseline["MeanAccuracy"] - 0.6550551723865207) < 1e-5
        and abs(baseline["MeanMacroF1"] - 0.32003390467151266) < 1e-5
    )

    results.to_csv(
        TABLES / "07_fusion_fold_metrics.csv", index=False
    )
    pd.DataFrame(inner_rows).to_csv(
        TABLES / "07_fusion_inner_metrics.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        TABLES / "07_fusion_oof_predictions.csv", index=False
    )
    summary.to_csv(
        TABLES / "07_fusion_summary.csv", index=False
    )

    metadata = {
        "experiment": "Phase 8 Experiment 07",
        "method": "Nested lexical-semantic fusion weighting",
        "candidate_weights": WEIGHTS,
        "inner_folds": 3,
        "outer_folds": 3,
        "seed": SEED,
        "baseline_reproduced": bool(reproduced),
        "outer_validation_used_for_selection": False,
        "development_results_exploratory": True,
        "prior_outer_results_informed_research_direction": True,
        "original_test_accessed": False,
    }

    (
        REPORTS / "07_fusion_metadata.json"
    ).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nEXPERIMENT 07 SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))
    print(
        "\nBaseline reproduction:",
        "PASS" if reproduced else "FAILED"
    )
    print("Original test accessed: No")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
