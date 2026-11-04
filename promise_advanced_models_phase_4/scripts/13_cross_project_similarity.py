
"""Phase 4 Experiment 13: Cross-project similarity analysis.

Uses training and validation requirements only.
Does not change labels, train a classifier, or read test data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT.parent

SPLITS = (
    THESIS
    / "promise_preprocessing_phase_2"
    / "outputs/splits"
)

TABLES = ROOT / "outputs/tables"
TABLES.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = ["PO", "FT", "SC"]


def read_split(name):
    path = SPLITS / f"final_{name}.csv"

    df = pd.read_csv(
        path,
        dtype={"RecordID": str, "ProjectID": str},
        keep_default_na=False,
    )

    if df["RecordID"].duplicated().any():
        raise ValueError(f"Duplicate RecordIDs in {name}")

    return df


def main():
    train = read_split("train")
    validation = read_split("validation")

    if len(train) != 2574 or len(validation) != 553:
        raise ValueError("Unexpected split sizes")

    if set(train["ProjectID"]) & set(validation["ProjectID"]):
        raise ValueError("Training/validation project overlap")

    target = validation[
        validation["_class_"].isin(TARGET_CLASSES)
    ].copy()

    # Fit vocabulary and IDF on training text ONLY.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        norm="l2",
    )

    train_matrix = vectorizer.fit_transform(
        train["NormalizedText"].astype(str)
    )

    target_matrix = vectorizer.transform(
        target["NormalizedText"].astype(str)
    )

    similarities = cosine_similarity(
        target_matrix,
        train_matrix
    )

    rows = []

    for i, (_, requirement) in enumerate(target.iterrows()):
        # Best match among all training examples.
        best_idx = int(np.argmax(similarities[i]))

        # Best match with the same known class.
        same_class_indices = np.flatnonzero(
            train["_class_"].to_numpy()
            == requirement["_class_"]
        )

        if len(same_class_indices) == 0:
            raise ValueError("Missing training class")

        same_idx = int(
            same_class_indices[
                np.argmax(similarities[i, same_class_indices])
            ]
        )

        best = train.iloc[best_idx]
        same = train.iloc[same_idx]

        rows.append({
            "ValidationRecordID": requirement["RecordID"],
            "ValidationProject": requirement["ProjectID"],
            "ActualClass": requirement["_class_"],
            "ValidationText": requirement["RequirementText"],

            "NearestTrainingClass": best["_class_"],
            "NearestTrainingProject": best["ProjectID"],
            "NearestTrainingText": best["RequirementText"],
            "NearestSimilarity": float(
                similarities[i, best_idx]
            ),

            "SameClassTrainingProject": same["ProjectID"],
            "SameClassTrainingText": same["RequirementText"],
            "SameClassSimilarity": float(
                similarities[i, same_idx]
            ),
        })

    results = pd.DataFrame(rows)

    output = (
        TABLES / "13_cross_project_similarity.csv"
    )

    results.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nEXPERIMENT 13: CROSS-PROJECT SIMILARITY")
    print("=" * 65)

    for label in TARGET_CLASSES:
        subset = results[
            results["ActualClass"] == label
        ]

        matching = (
            subset["NearestTrainingClass"] == label
        ).sum()

        print(f"\nCLASS: {label}")
        print("Validation requirements:", len(subset))
        print(
            "Nearest example has matching label:",
            int(matching),
        )
        print(
            "Mean nearest similarity:",
            round(subset["NearestSimilarity"].mean(), 4),
        )
        print(
            "Mean same-class similarity:",
            round(subset["SameClassSimilarity"].mean(), 4),
        )

        print("\nNearest training labels:")
        print(
            subset["NearestTrainingClass"]
            .value_counts()
            .to_string()
        )

    print("\nSaved:", output)
    print("Original test partition not accessed.")


if __name__ == "__main__":
    main()
