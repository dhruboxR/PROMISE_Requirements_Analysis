
"""Phase 4 Experiment 09: Three-model prediction overlap.

Compares Logistic Regression, DistilBERT and RoBERTa
on the SAME validation records.

No model training. No test partition access.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent
TABLES = ROOT / "outputs" / "tables"
REPORTS = ROOT / "outputs" / "reports"

VALIDATION = (
    THESIS / "promise_preprocessing_phase_2"
    / "outputs/splits/final_validation.csv"
)

LR_MODEL = (
    THESIS / "promise_model_development_phase_3"
    / "outputs/models/07_tuned_logistic_regression.joblib"
)

DISTIL = (
    TABLES / "05_extended_weighted_distilbert_predictions.csv"
)

ROBERTA = (
    TABLES / "07_weighted_roberta_predictions.csv"
)

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

EXPECTED_LR_ACCURACY = 0.669078


def load_transformer_predictions(path, name):
    df = pd.read_csv(
        path,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    required = {
        "RecordID", "ProjectID",
        "_class_", "PredictedLabel"
    }

    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}")

    if df["RecordID"].duplicated().any():
        raise ValueError(f"Duplicate IDs: {name}")

    if len(df) != 553:
        raise ValueError(f"Unexpected record count: {name}")

    return df[
        ["RecordID", "ProjectID", "_class_", "PredictedLabel"]
    ].rename(
        columns={"PredictedLabel": f"{name}_Prediction"}
    )


def load_logistic_predictions(validation):
    """Use the saved Phase 3 pipeline without fitting it."""

    model = joblib.load(LR_MODEL)

    if not hasattr(model, "predict"):
        raise TypeError(
            "Saved Logistic Regression artifact has no "
            "predict method. Inspect its structure before "
            "continuing."
        )

    # The exact Phase 3 text input is verified against its
    # previously reported validation accuracy.
    for column in ["NormalizedText", "RequirementText"]:
        if column not in validation.columns:
            continue

        try:
            predictions = np.asarray(
                model.predict(
                    validation[column].fillna("").astype(str)
                )
            ).astype(str)

        except (ValueError, TypeError, AttributeError) as exc:
            print(f"{column} input failed: {exc}")
            continue

        accuracy = accuracy_score(
            validation["_class_"].astype(str),
            predictions,
        )

        print(
            f"Logistic Regression using {column}: "
            f"accuracy={accuracy:.6f}"
        )

        if abs(accuracy - EXPECTED_LR_ACCURACY) < 0.00001:
            print("Phase 3 validation accuracy verified.")
            return predictions

    raise ValueError(
        "Could not reproduce Phase 3 validation accuracy. "
        "Check the original Phase 3 preprocessing and "
        "prediction code before comparing models."
    )


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    for path in [VALIDATION, LR_MODEL, DISTIL, ROBERTA]:
        if not path.exists():
            raise FileNotFoundError(path)

    validation = pd.read_csv(
        VALIDATION,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if len(validation) != 553:
        raise ValueError("Invalid validation record count")

    if validation["RecordID"].duplicated().any():
        raise ValueError("Duplicate validation RecordIDs")

    validation["LR_Prediction"] = (
        load_logistic_predictions(validation)
    )

    distil = load_transformer_predictions(
        DISTIL, "DistilBERT"
    )

    roberta = load_transformer_predictions(
        ROBERTA, "RoBERTa"
    )

    # One-to-one joins prevent accidental row misalignment.
    merged = validation[
        ["RecordID", "ProjectID", "_class_", "LR_Prediction"]
    ].merge(
        distil,
        on="RecordID",
        suffixes=("", "_distil"),
        validate="one_to_one",
    ).merge(
        roberta,
        on="RecordID",
        suffixes=("", "_roberta"),
        validate="one_to_one",
    )

    if len(merged) != 553:
        raise ValueError("Records lost during merge")

    for suffix in ["distil", "roberta"]:
        if not (
            merged["ProjectID"]
            == merged[f"ProjectID_{suffix}"]
        ).all():
            raise ValueError("ProjectID mismatch")

        if not (
            merged["_class_"]
            == merged[f"_class__{suffix}"]
        ).all():
            raise ValueError("Ground-truth label mismatch")

    models = {
        "LR": "LR_Prediction",
        "DistilBERT": "DistilBERT_Prediction",
        "RoBERTa": "RoBERTa_Prediction",
    }

    print("\nTHREE-MODEL COMPLEMENTARITY ANALYSIS")
    print("=" * 65)

    for name, column in models.items():
        merged[f"{name}_Correct"] = (
            merged[column] == merged["_class_"]
        )

        accuracy = accuracy_score(
            merged["_class_"], merged[column]
        )

        macro_f1 = f1_score(
            merged["_class_"],
            merged[column],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        print(
            f"{name:<12} Accuracy: {accuracy:.4f} "
            f"Macro-F1: {macro_f1:.4f}"
        )

    a = merged["LR_Correct"]
    b = merged["DistilBERT_Correct"]
    c = merged["RoBERTa_Correct"]

    overlap = {
        "All three correct": int((a & b & c).sum()),
        "Only LR correct": int((a & ~b & ~c).sum()),
        "Only DistilBERT correct": int((~a & b & ~c).sum()),
        "Only RoBERTa correct": int((~a & ~b & c).sum()),
        "LR + DistilBERT correct": int((a & b & ~c).sum()),
        "LR + RoBERTa correct": int((a & ~b & c).sum()),
        "DistilBERT + RoBERTa correct": int((~a & b & c).sum()),
        "All three wrong": int((~a & ~b & ~c).sum()),
    }

    print("\nCORRECTNESS OVERLAP")
    print("-" * 65)

    for category, count in overlap.items():
        print(f"{category:<35}: {count:>4}")

    assert sum(overlap.values()) == len(merged)

    oracle_correct = int((a | b | c).sum())
    oracle_accuracy = oracle_correct / len(merged)

    print("\nORACLE ANALYSIS")
    print("-" * 65)
    print("Records:", len(merged))
    print("At least one model correct:", oracle_correct)
    print("All models wrong:", overlap["All three wrong"])
    print(f"Oracle accuracy upper bound: {oracle_accuracy:.4f}")

    print("\nDIFFICULT CLASS ANALYSIS")
    print("-" * 65)

    for label in ["FT", "PO", "SC", "MN", "LF"]:
        subset = merged[merged["_class_"] == label]

        print(
            f"{label}: total={len(subset)}, "
            f"LR={int(subset['LR_Correct'].sum())}, "
            f"DistilBERT={int(subset['DistilBERT_Correct'].sum())}, "
            f"RoBERTa={int(subset['RoBERTa_Correct'].sum())}"
        )

    merged.to_csv(
        TABLES / "09_three_model_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([
        {"Category": key, "Count": value}
        for key, value in overlap.items()
    ]).to_csv(
        TABLES / "09_three_model_overlap.csv",
        index=False,
    )

    print("\nResults saved successfully.")
    print("No model retraining performed.")
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
