
"""
Phase 8 Experiment 16
Cross-project class consistency review.

Creates side-by-side samples of training and held-out
requirements for difficult project/class combinations.

No training, relabeling, or original test access.
"""

from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_SCRIPT = (
    ROOT.parent
    / "promise_advanced_models_phase_4"
    / "scripts/01_group_aware_feature_comparison.py"
)

MANIFEST = (
    ROOT.parent
    / "promise_improved_model_phase_5"
    / "outputs/tables/01_fixed_fold_assignments.csv"
)

OUTPUT = ROOT / "outputs/tables"
OUTPUT.mkdir(parents=True, exist_ok=True)

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
    "phase4_exp16", DATA_SCRIPT
)

if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load Phase 4 dataset module")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

data = module.load_development_data().copy()

data["RecordID"] = data["RecordID"].astype(str)
data["ProjectID"] = data["ProjectID"].astype(str)

manifest = pd.read_csv(
    MANIFEST,
    dtype={"RecordID": str, "ProjectID": str},
    keep_default_na=False,
)

if (
    len(data) != 3127
    or len(manifest) != 3127
    or data["RecordID"].duplicated().any()
    or manifest["RecordID"].duplicated().any()
    or set(data["RecordID"]) != set(manifest["RecordID"])
):
    raise ValueError("Invalid development dataset or fold manifest")

aligned = manifest.set_index("RecordID").loc[
    data["RecordID"]
]

if not (
    aligned["ProjectID"].to_numpy()
    == data["ProjectID"].to_numpy()
).all():
    raise ValueError("Project mismatch")

if not (
    aligned["ActualClass"].to_numpy()
    == data["_class_"].astype(str).to_numpy()
).all():
    raise ValueError("Label mismatch")

data["Fold"] = aligned["Fold"].astype(int).to_numpy()

if set(data["Fold"]) != {1, 2, 3}:
    raise ValueError("Unexpected outer folds")

rows = []

for project, label in TARGETS:
    heldout = data[
        (data["ProjectID"] == project)
        & (data["_class_"] == label)
    ].copy()

    if heldout.empty or heldout["Fold"].nunique() != 1:
        raise ValueError(f"Invalid target: {project}/{label}")

    fold = int(heldout["Fold"].iloc[0])

    training = data[
        (data["Fold"] != fold)
        & (data["_class_"] == label)
    ].copy()

    if project in set(training["ProjectID"]):
        raise ValueError("Project leakage detected")

    print(f"\nCLASS {label} | HELD-OUT PROJECT {project}")
    print("=" * 70)
    print("Held-out examples:", len(heldout))
    print("Training examples:", len(training))
    print("Training projects:", training["ProjectID"].nunique())

    # Sample different training projects before repeating a project.
    training = training.sort_values(
        ["ProjectID", "RecordID"]
    )

    training_sample = (
        training.groupby("ProjectID", group_keys=False)
        .head(1)
        .head(8)
    )

    heldout_sample = heldout.sort_values(
        "RecordID"
    ).head(8)

    for source, subset in [
        ("Training", training_sample),
        ("HeldOut", heldout_sample),
    ]:
        print(f"\n{source.upper()} SAMPLES")

        for _, record in subset.iterrows():
            text = str(record["NormalizedText"])

            print(
                f"\nProject={record['ProjectID']} "
                f"Record={record['RecordID']}"
            )
            print(text[:350])

            rows.append({
                "TargetClass": label,
                "TargetProject": project,
                "OuterFold": fold,
                "Source": source,
                "RecordID": record["RecordID"],
                "ProjectID": record["ProjectID"],
                "OriginalClass": record["_class_"],
                "RequirementText": text,
            })

result = pd.DataFrame(rows)

path = OUTPUT / "16_cross_project_class_review.csv"
result.to_csv(path, index=False)

print("\nEXPERIMENT 16 COMPLETED")
print("Review examples saved:", len(result))
print("Output:", path)
print("Original labels changed: No")
print("Original test accessed: No")
