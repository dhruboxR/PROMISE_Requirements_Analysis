
"""
Phase 8 Experiment 05
Nested Confusion-Aware Specialist Correction (CASC)

Outer folds: original Phase 5 fixed group-aware folds.
Inner folds: 3-fold StratifiedGroupKFold on outer training data.

Inner OOF predictions select specialist thresholds.
Outer validation labels are never used to select thresholds.

Exploratory development experiment, not independent
test validation. No original test records are accessed.
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
    accuracy_score, f1_score, balanced_accuracy_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
P4 = THESIS / "promise_advanced_models_phase_4"
P5 = THESIS / "promise_improved_model_phase_5"

TABLES = ROOT / "outputs/tables"
REPORTS = ROOT / "outputs/reports"

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO",
]
TARGETS = ["O", "SC", "PO"]
SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 1.01 means no corrections for a specialist.
THRESHOLDS = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.01,
]

MAX_INNER_ACCURACY_DROP = 0.02


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weights(y):
    counts = pd.Series(y).value_counts()
    classes = sorted(counts.index)
    n = len(y)
    k = len(classes)

    raw = {
        c: np.sqrt(n / (k * counts[c]))
        for c in classes
    }
    avg = sum(counts[c] * raw[c] for c in classes) / n

    return {c: float(raw[c] / avg) for c in classes}


def classifier(y):
    return LogisticRegression(
        C=10.0,
        class_weight=weights(y),
        max_iter=2000,
        random_state=SEED,
    )


def fit_models(x, y):
    if set(y) != set(LABELS):
        raise ValueError("Training partition missing 12-class labels")

    base = classifier(y)
    base.fit(x, y)

    specialists = {}

    for target in TARGETS:
        mask = np.isin(y, ["F", target])
        binary_y = (y[mask] == target).astype(int)

        if set(binary_y) != {0, 1}:
            raise ValueError(f"Missing binary class: {target}")

        model = classifier(binary_y)
        model.fit(x[mask], binary_y)
        specialists[target] = model

    return base, specialists


def get_predictions(base, specialists, x):
    baseline = base.predict(x)
    probabilities = {}

    for target, model in specialists.items():
        proba = model.predict_proba(x)
        positive_column = list(model.classes_).index(1)
        probabilities[target] = proba[:, positive_column]

    return baseline, probabilities


def correct_predictions(baseline, probabilities, thresholds):
    result = np.asarray(baseline, dtype=object).copy()
    original_f = np.asarray(baseline) == "F"

    # Competing specialists are ranked by score minus threshold.
    margin = np.full(len(result), -np.inf)
    chosen = np.asarray(["F"] * len(result), dtype=object)

    for target in TARGETS:
        threshold = thresholds[target]
        candidate_margin = probabilities[target] - threshold

        eligible = (
            original_f
            & (probabilities[target] >= threshold)
            & (candidate_margin > margin)
        )

        margin[eligible] = candidate_margin[eligible]
        chosen[eligible] = target

    changed = original_f & (margin > -np.inf)
    result[changed] = chosen[changed]

    invalid = set(result) - set(LABELS)
    if invalid:
        raise ValueError(
            f"Invalid predicted classes: {invalid}"
        )

    return result


def score(y, predicted):
    return {
        "Accuracy": float(accuracy_score(y, predicted)),
        "MacroF1": float(f1_score(
            y, predicted, labels=LABELS,
            average="macro", zero_division=0,
        )),
        "BalancedAccuracy": float(
            balanced_accuracy_score(y, predicted)
        ),
    }


def choose_thresholds(y, baseline, probabilities):
    disabled = {target: 1.01 for target in TARGETS}
    selected = disabled.copy()
    baseline_scores = score(y, baseline)

    accuracy_floor = (
        baseline_scores["Accuracy"] - MAX_INNER_ACCURACY_DROP
    )
    best_score = baseline_scores["MacroF1"]

    # Deterministic coordinate search. Begin with no corrections.
    for _ in range(2):
        improved = False

        for target in TARGETS:
            current = selected.copy()
            candidate_best = best_score
            candidate_threshold = current[target]

            for threshold in THRESHOLDS:
                trial = current.copy()
                trial[target] = threshold

                predicted = correct_predictions(
                    baseline, probabilities, trial
                )
                metrics = score(y, predicted)

                if metrics["Accuracy"] < accuracy_floor:
                    continue

                if metrics["MacroF1"] > candidate_best + 1e-6:
                    candidate_best = metrics["MacroF1"]
                    candidate_threshold = threshold

            if candidate_threshold != selected[target]:
                selected[target] = candidate_threshold
                best_score = candidate_best
                improved = True

        if not improved:
            break

    return selected, baseline_scores, score(
        y, correct_predictions(baseline, probabilities, selected)
    )


def make_features(vectorizer_factory, train_text, val_text,
                  train_emb, val_emb):
    vectorizer = vectorizer_factory("word+character")

    lex_train = vectorizer.fit_transform(train_text)
    lex_val = vectorizer.transform(val_text)

    x_train = hstack([
        lex_train,
        csr_matrix(train_emb),
    ], format="csr")

    x_val = hstack([
        lex_val,
        csr_matrix(val_emb),
    ], format="csr")

    return x_train, x_val


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    p4cv = load_module(
        P4 / "scripts/01_group_aware_feature_comparison.py",
        "p4cv05",
    )
    semantic = load_module(
        P4 / "scripts/17_semantic_embedding_logistic_cv.py",
        "semantic05",
    )

    if (
        p4cv.LABELS != LABELS
        or p4cv.SEED != SEED
        or semantic.MODEL_NAME != MODEL_NAME
        or semantic.MAX_LENGTH != 128
    ):
        raise ValueError("Previous configuration mismatch")

    data = p4cv.load_development_data().copy()
    if len(data) != 3127:
        raise ValueError("Unexpected development size")

    data["RecordID"] = data["RecordID"].astype(str)
    data["ProjectID"] = data["ProjectID"].astype(str)

    text = data["NormalizedText"].astype(str).to_numpy()
    y = data["_class_"].astype(str).to_numpy()
    groups = np.asarray(p4cv.create_connected_groups(data))

    manifest = pd.read_csv(
        P5 / "outputs/tables/01_fixed_fold_assignments.csv",
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if (
        len(manifest) != len(data)
        or manifest["RecordID"].duplicated().any()
        or data["RecordID"].duplicated().any()
        or set(manifest["RecordID"]) != set(data["RecordID"])
    ):
        raise ValueError("Invalid fold manifest")

    aligned = manifest.set_index("RecordID").loc[data["RecordID"]]

    if not np.array_equal(aligned["ActualClass"].to_numpy(), y):
        raise ValueError("Class labels differ from manifest")

    if not np.array_equal(
        aligned["ProjectID"].astype(str).to_numpy(),
        data["ProjectID"].to_numpy(),
    ):
        raise ValueError("Project IDs differ from manifest")

    outer_folds = aligned["Fold"].astype(int).to_numpy()

    if set(outer_folds) != {1, 2, 3}:
        raise ValueError("Unexpected outer fold IDs")

    if (
        pd.DataFrame({"Group": groups, "Fold": outer_folds})
        .groupby("Group")["Fold"].nunique() > 1
    ).any():
        raise ValueError("Outer fold group leakage")

    from transformers import AutoTokenizer, AutoModel

    semantic.set_seed()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Generating frozen MiniLM embeddings...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    encoder.eval()

    embeddings = semantic.generate_embeddings(
        text.tolist(), tokenizer, encoder, device
    )

    if (
        embeddings.shape != (3127, 384)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid MiniLM embeddings")

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows = []
    prediction_rows = []
    threshold_rows = []

    for outer in [1, 2, 3]:
        print(f"\nOUTER FOLD {outer}", flush=True)

        train = np.flatnonzero(outer_folds != outer)
        val = np.flatnonzero(outer_folds == outer)

        if set(groups[train]) & set(groups[val]):
            raise ValueError("Outer group leakage")

        inner_cv = StratifiedGroupKFold(
            n_splits=3,
            shuffle=True,
            random_state=SEED,
        )

        inner_baseline = np.empty(len(train), dtype=object)
        inner_probs = {
            target: np.full(len(train), np.nan)
            for target in TARGETS
        }
        covered = np.zeros(len(train), dtype=bool)

        for inner_fold, (tr, va) in enumerate(
            inner_cv.split(text[train], y[train], groups[train]),
            start=1,
        ):
            if set(groups[train][tr]) & set(groups[train][va]):
                raise ValueError("Inner group leakage")

            print(f"  Inner fold {inner_fold}", flush=True)

            x_tr, x_va = make_features(
                p4cv.make_features,
                text[train][tr],
                text[train][va],
                embeddings[train][tr],
                embeddings[train][va],
            )

            base, specialists = fit_models(x_tr, y[train][tr])
            pred, probs = get_predictions(base, specialists, x_va)

            inner_baseline[va] = pred
            covered[va] = True

            for target in TARGETS:
                inner_probs[target][va] = probs[target]

        if not covered.all():
            raise ValueError("Incomplete inner OOF predictions")

        if any(
            not np.isfinite(inner_probs[target]).all()
            for target in TARGETS
        ):
            raise ValueError("Missing inner specialist probabilities")

        thresholds, inner_before, inner_after = choose_thresholds(
            y[train], inner_baseline, inner_probs
        )

        print("  Selected thresholds:", thresholds, flush=True)
        print(
            "  Inner Macro-F1:",
            round(inner_before["MacroF1"], 4),
            "->",
            round(inner_after["MacroF1"], 4),
            flush=True,
        )

        x_train, x_val = make_features(
            p4cv.make_features,
            text[train],
            text[val],
            embeddings[train],
            embeddings[val],
        )

        base, specialists = fit_models(x_train, y[train])
        baseline, probs = get_predictions(base, specialists, x_val)

        corrected = correct_predictions(
            baseline, probs, thresholds
        )

        for model_name, pred in [
            ("BaselineTempered", baseline),
            ("CASC", corrected),
        ]:
            metrics = score(y[val], pred)
            recalls = recall_score(
                y[val], pred, labels=LABELS,
                average=None, zero_division=0,
            )
            rows.append({
                "Model": model_name,
                "Fold": outer,
                "ValidationRecords": len(val),
                **metrics,
                **{
                    f"Recall_{name}": float(recalls[LABELS.index(name)])
                    for name in ["F", "O", "SC", "PO"]
                },
            })

            print(
                f"  {model_name}: "
                f"Acc={metrics['Accuracy']:.4f} "
                f"MacroF1={metrics['MacroF1']:.4f}",
                flush=True,
            )

        for position, index in enumerate(val):
            prediction_rows.append({
                "RecordID": data.iloc[index]["RecordID"],
                "Fold": outer,
                "ActualClass": y[index],
                "BaselinePrediction": baseline[position],
                "CASCPrediction": corrected[position],
                **{
                    f"Probability_{target}": float(probs[target][position])
                    for target in TARGETS
                },
            })

        threshold_rows.append({
            "OuterFold": outer,
            **{f"Threshold_{t}": thresholds[t] for t in TARGETS},
            "InnerBaselineAccuracy": inner_before["Accuracy"],
            "InnerBaselineMacroF1": inner_before["MacroF1"],
            "InnerCAS CAccuracy".replace(" ", ""): inner_after["Accuracy"],
            "InnerCASCMacroF1": inner_after["MacroF1"],
        })

    metrics_df = pd.DataFrame(rows)
    summary = (
        metrics_df.groupby("Model", as_index=False)
        .agg(
            MeanAccuracy=("Accuracy", "mean"),
            MeanMacroF1=("MacroF1", "mean"),
            MeanBalancedAccuracy=("BalancedAccuracy", "mean"),
            MeanRecallO=("Recall_O", "mean"),
            MeanRecallSC=("Recall_SC", "mean"),
            MeanRecallPO=("Recall_PO", "mean"),
        )
    )

    baseline = summary.loc[
        summary["Model"] == "BaselineTempered"
    ].iloc[0]

    reproduced = (
        abs(baseline["MeanAccuracy"] - 0.6550551723865207) < 1e-5
        and abs(baseline["MeanMacroF1"] - 0.32003390467151266) < 1e-5
    )

    metrics_df.to_csv(
        TABLES / "05_nested_casc_fold_metrics.csv", index=False
    )
    summary.to_csv(
        TABLES / "05_nested_casc_summary.csv", index=False
    )
    pd.DataFrame(prediction_rows).to_csv(
        TABLES / "05_nested_casc_oof_predictions.csv", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        TABLES / "05_nested_casc_thresholds.csv", index=False
    )

    metadata = {
        "experiment": "Phase 8 Experiment 05",
        "method": "Nested Confusion-Aware Specialist Correction",
        "inner_folds": 3,
        "outer_folds": 3,
        "threshold_candidates": THRESHOLDS,
        "maximum_inner_accuracy_drop": MAX_INNER_ACCURACY_DROP,
        "baseline_reproduced": bool(reproduced),
        "outer_validation_used_for_threshold_selection": False,
        "previous_outer_results_informed_method_design": True,
        "original_test_accessed": False,
        "independent_validation_established": False,
    }

    (REPORTS / "05_nested_casc_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nPHASE 8 EXPERIMENT 05 RESULTS")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("\nBaseline reproduction:", "PASS" if reproduced else "FAILED")
    print("Original test accessed: No")
    print("Original datasets modified: No")


if __name__ == "__main__":
    main()
