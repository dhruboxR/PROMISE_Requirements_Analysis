
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs/tables"

DATA_SCRIPT = (
    ROOT.parent
    / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

PREDICTIONS = TABLES / "07_fusion_oof_predictions.csv"

MANIFEST = (
    ROOT.parent
    / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

# Project/class pairs identified in Experiment 14.
TARGETS = [
    ("82", "L"),
    ("80", "SC"),
    ("60", "SC"),
    ("59", "PO"),
    ("52", "PO"),
    ("74", "O"),
    ("6", "O"),
]

spec = importlib.util.spec_from_file_location(
    "phase4_exp15", DATA_SCRIPT
)

if spec is None or spec.loader is None:
    raise RuntimeError("Phase 4 loader unavailable")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

data = module.load_development_data().copy()
data["RecordID"] = data["RecordID"].astype(str)
data["ProjectID"] = data["ProjectID"].astype(str)

pred = pd.read_csv(
    PREDICTIONS,
    dtype={"RecordID": str},
)

pred = pred[
    pred["Model"] == "BaselineWeight1"
].copy()

manifest = pd.read_csv(
    MANIFEST,
    dtype={"RecordID": str, "ProjectID": str},
    keep_default_na=False,
)

if (
    len(data) != 3127
    or len(pred) != 3127
    or data["RecordID"].duplicated().any()
    or pred["RecordID"].duplicated().any()
    or manifest["RecordID"].duplicated().any()
):
    raise ValueError("Invalid development or prediction data")

df = data.merge(
    pred[
        ["RecordID", "Fold", "ActualClass", "PredictedClass"]
    ],
    on="RecordID",
    validate="one_to_one",
)

df = df.merge(
    manifest[["RecordID", "ProjectID", "Fold"]],
    on="RecordID",
    how="left",
    suffixes=("", "_Manifest"),
    validate="one_to_one",
)

if len(df) != 3127:
    raise ValueError("Unexpected merged record count")

if not (df["_class_"] == df["ActualClass"]).all():
    raise ValueError("Actual class mismatch")

if not (
    df["ProjectID"] == df["ProjectID_Manifest"]
).all():
    raise ValueError("Project mismatch")

if not (df["Fold"] == df["Fold_Manifest"]).all():
    raise ValueError("Fold mismatch")

df["Correct"] = (
    df["ActualClass"] == df["PredictedClass"]
)

review = []
rows = []

print("\nPHASE 8 EXPERIMENT 15")
print("=" * 75)

for project, target in TARGETS:
    subset = df[
        (df["ProjectID"] == project)
        & (df["ActualClass"] == target)
    ].copy()

    if subset.empty:
        raise ValueError(
            f"Missing expected project/class: {project}/{target}"
        )

    fold_ids = sorted(subset["Fold"].unique())

    if len(fold_ids) != 1:
        raise ValueError(
            f"Project {project} spans multiple folds"
        )

    fold = fold_ids[0]

    train = df[df["Fold"] != fold]

    # Confirm there is no project overlap.
    if project in set(train["ProjectID"]):
        raise ValueError("Project isolation violation")

    train_class = train[train["ActualClass"] == target]

    print(f"\nPROJECT {project} | CLASS {target}")
    print("-" * 75)

    print("Held-out fold:", fold)
    print("Project/class support:", len(subset))
    print("Correct:", int(subset["Correct"].sum()))

    print(
        "Training class examples:",
        len(train_class),
    )
    print(
        "Training projects containing class:",
        train_class["ProjectID"].nunique(),
    )

    print("\nPredicted class distribution:")
    print(
        subset["PredictedClass"]
        .value_counts()
        .to_string()
    )

    rows.append({
        "ProjectID": project,
        "Class": target,
        "Fold": int(fold),
        "Support": len(subset),
        "Correct": int(subset["Correct"].sum()),
        "Recall": float(subset["Correct"].mean()),
        "TrainingClassCount": len(train_class),
        "TrainingClassProjects":
            train_class["ProjectID"].nunique(),
    })

    # Deterministic samples: errors first, then correct cases.
    subset = subset.sort_values(
        ["Correct", "RecordID"],
        ascending=[True, True],
    )

    samples = subset.head(8)

    print("\nSample requirements:")

    for _, item in samples.iterrows():
        requirement = str(item["NormalizedText"])

        print(
            f"\nID: {item['RecordID']} | "
            f"Actual: {target} | "
            f"Predicted: {item['PredictedClass']}"
        )
        print("Text:", requirement[:250])

        review.append({
            "ProjectID": project,
            "Fold": int(fold),
            "RecordID": item["RecordID"],
            "ActualClass": target,
            "PredictedClass": item["PredictedClass"],
            "Correct": bool(item["Correct"]),
            "RequirementText": requirement,
        })

TABLES.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_csv(
    TABLES / "15_project_failure_summary.csv",
    index=False,
)

pd.DataFrame(review).to_csv(
    TABLES / "15_requirement_review_samples.csv",
    index=False,
)

print("\nEXPERIMENT 15 COMPLETED")
print("=" * 75)
print("Project failure summary: Saved")
print("Requirement review samples: Saved")
print("Model retraining: No")
print("Original test accessed: No")
print("Original datasets modified: No")
