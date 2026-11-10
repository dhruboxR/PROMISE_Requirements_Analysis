
"""
Phase 8: Experiment 00
Inspect previous datasets, fold assignments, model outputs,
and installed dependencies before reproducing the baseline.

Read-only inspection. No source files are modified.
"""

from pathlib import Path
import importlib.metadata
import pandas as pd

THESIS = Path.home() / "Unofficial Thesis"

PHASE_2 = THESIS / "promise_preprocessing_phase_2"
PHASE_5 = THESIS / "promise_improved_model_phase_5"
PHASE_6 = THESIS / "promise_advanced_improvement_phase_6"
PHASE_8 = THESIS / "promise_classification_phase_8"

files = {
    "Training split": (
        PHASE_2 / "outputs/splits/final_train.csv"
    ),
    "Validation split": (
        PHASE_2 / "outputs/splits/final_validation.csv"
    ),
    "Test split (metadata only)": (
        PHASE_2 / "outputs/splits/final_test.csv"
    ),
    "Phase 5 fixed folds": (
        PHASE_5 / "outputs/tables/01_fixed_fold_assignments.csv"
    ),
    "Phase 6 verified folds": (
        PHASE_6 / "outputs/tables/01_verified_folds.csv"
    ),
}

def inspect_csv(name, path):
    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)
    print("Path:", path)

    if not path.exists():
        print("STATUS: NOT FOUND")
        return

    df = pd.read_csv(path)

    print("STATUS: FOUND")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False, max_colwidth=50))

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nColumn uniqueness:")
    for col in df.columns:
        print(
            f"{col}: {df[col].nunique(dropna=False)}"
        )


def inspect_source_scripts():
    print("\n" + "=" * 65)
    print("PHASE 5 PYTHON SCRIPTS")
    print("=" * 65)

    scripts_dir = PHASE_5 / "scripts"

    if not scripts_dir.exists():
        print("Scripts directory not found")
        return

    for path in sorted(scripts_dir.glob("*.py")):
        print(path.name)


def inspect_model_outputs():
    print("\n" + "=" * 65)
    print("PHASE 5 OUTPUT FILES")
    print("=" * 65)

    output_dir = PHASE_5 / "outputs"

    if not output_dir.exists():
        print("Output directory not found")
        return

    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".csv", ".json", ".md"
        }:
            print(path.relative_to(PHASE_5))


def inspect_packages():
    print("\n" + "=" * 65)
    print("PYTHON DEPENDENCIES")
    print("=" * 65)

    packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "sentence-transformers",
        "torch",
        "transformers",
    ]

    for package in packages:
        try:
            version = importlib.metadata.version(package)
            print(f"{package}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: NOT INSTALLED")


def main():
    print("\nPHASE 8: INITIAL EXPERIMENT INSPECTION")

    for name, path in files.items():
        inspect_csv(name, path)

    inspect_source_scripts()
    inspect_model_outputs()
    inspect_packages()

    print("\n" + "=" * 65)
    print("INSPECTION FINISHED")
    print("=" * 65)
    print("No datasets or previous experiments modified.")


if __name__ == "__main__":
    main()
