
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"

DISTIL_PATH = TABLES / "05_extended_weighted_distilbert_predictions.csv"
ROBERTA_PATH = TABLES / "07_weighted_roberta_predictions.csv"


def load_predictions(path, model_name):
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
        raise ValueError(f"Duplicate IDs in {path}")

    return df[
        ["RecordID", "ProjectID", "_class_", "PredictedLabel"]
    ].rename(columns={
        "PredictedLabel": f"{model_name}_Prediction"
    })


def main():
    distil = load_predictions(DISTIL_PATH, "DistilBERT")
    roberta = load_predictions(ROBERTA_PATH, "RoBERTa")

    if set(distil["RecordID"]) != set(roberta["RecordID"]):
        raise ValueError("Models evaluated different records")

    merged = distil.merge(
        roberta,
        on="RecordID",
        suffixes=("_distil", "_roberta"),
        validate="one_to_one",
    )

    if len(merged) != 553:
        raise ValueError("Unexpected validation record count")

    if not (
        merged["_class__distil"]
        == merged["_class__roberta"]
    ).all():
        raise ValueError("True labels do not match")

    if not (
        merged["ProjectID_distil"]
        == merged["ProjectID_roberta"]
    ).all():
        raise ValueError("Project IDs do not match")

    merged["Actual"] = merged["_class__distil"]

    merged["Distil_Correct"] = (
        merged["DistilBERT_Prediction"]
        == merged["Actual"]
    )

    merged["RoBERTa_Correct"] = (
        merged["RoBERTa_Prediction"]
        == merged["Actual"]
    )

    both_correct = (
        merged["Distil_Correct"]
        & merged["RoBERTa_Correct"]
    ).sum()

    distil_only = (
        merged["Distil_Correct"]
        & ~merged["RoBERTa_Correct"]
    ).sum()

    roberta_only = (
        ~merged["Distil_Correct"]
        & merged["RoBERTa_Correct"]
    ).sum()

    both_wrong = (
        ~merged["Distil_Correct"]
        & ~merged["RoBERTa_Correct"]
    ).sum()

    print("\nMODEL COMPLEMENTARITY ANALYSIS")
    print("=" * 55)

    print("Both correct:", both_correct)
    print("Only DistilBERT correct:", distil_only)
    print("Only RoBERTa correct:", roberta_only)
    print("Both wrong:", both_wrong)

    print("\nTotal records:", len(merged))

    disagreements = (
        merged["DistilBERT_Prediction"]
        != merged["RoBERTa_Prediction"]
    ).sum()

    print("Prediction disagreements:", disagreements)

    # Diagnostic upper bound; requires an oracle to choose
    # the correct model for each record.
    oracle = (
        both_correct + distil_only + roberta_only
    ) / len(merged)

    print(f"Oracle accuracy upper bound: {oracle:.4f}")

    merged.to_csv(
        TABLES / "08_model_complementarity.csv",
        index=False,
    )

    print("\nSaved: outputs/tables/08_model_complementarity.csv")


if __name__ == "__main__":
    main()
