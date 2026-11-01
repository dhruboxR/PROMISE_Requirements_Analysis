
"""Create leakage-aware PROMISE+ train/validation/test splits."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import random

import pandas as pd

from common import (
    CLEANED_DIR, SPLITS_DIR, REPORTS_DIR,
    TABLES_DIR, prepare_outputs
)

INPUT = CLEANED_DIR / "04_labeled_requirements.csv"
SEED = 42
TARGETS = {"train": 0.70, "validation": 0.15, "test": 0.15}
N_TRIALS = 3000


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def text_key(text):
    """Consistent duplicate matching with Phase 2 Step 3."""
    return " ".join(str(text).casefold().split())


def build_groups(df):
    """Connect records sharing a project or normalized text."""
    uf = UnionFind(len(df))
    seen_projects = {}
    seen_texts = {}

    for i, row in df.iterrows():
        project = row["ProjectID"]
        key = text_key(row["NormalizedText"])

        if project in seen_projects:
            uf.union(i, seen_projects[project])
        else:
            seen_projects[project] = i

        if key in seen_texts:
            uf.union(i, seen_texts[key])
        else:
            seen_texts[key] = i

    groups = defaultdict(list)
    for i in range(len(df)):
        groups[uf.find(i)].append(i)

    return list(groups.values())


def score_split(assignment, components, df, labels):
    """Assess size balance and class-distribution differences."""
    total = len(df)
    global_counts = df["_class_"].value_counts(normalize=True)

    score = 0.0

    for name, component_ids in assignment.items():
        indices = [
            idx
            for component_id in component_ids
            for idx in components[component_id]
        ]

        size = len(indices)
        score += 5 * abs(size / total - TARGETS[name])

        if size == 0:
            score += 100
            continue

        subset = df.iloc[indices]
        proportions = subset["_class_"].value_counts(
            normalize=True
        )

        score += sum(
            abs(proportions.get(label, 0) -
                global_counts.get(label, 0))
            for label in labels
        )

        if name != "train":
            missing = set(labels) - set(subset["_class_"])
            score += 2 * len(missing)

    return score


def main():
    prepare_outputs()

    df = pd.read_csv(
        INPUT,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False,
    ).reset_index(drop=True)

    assert len(df) == 3677
    assert df["RecordID"].is_unique

    components = build_groups(df)
    total = len(df)
    component_sizes = [len(g) for g in components]
    labels = sorted(df["_class_"].unique())

    print("\nPROMISE+ LEAKAGE-AWARE SPLITTING")
    print("=" * 52)
    print("Total records:", total)
    print("Connected components:", len(components))
    print("Largest component:", max(component_sizes))
    print("Target split:", TARGETS)

    # A large connected component cannot be divided
    # without violating at least one isolation constraint.
    if len(components) < 3:
        raise RuntimeError(
            "Fewer than three independent components exist. "
            "A three-way leakage-free split is impossible."
        )

    rng = random.Random(SEED)
    best = None
    best_score = float("inf")
    names = list(TARGETS)

    for trial in range(N_TRIALS):
        order = list(range(len(components)))
        rng.shuffle(order)

        # Prioritize larger groups, randomizing equal sizes.
        order.sort(key=lambda i: len(components[i]), reverse=True)

        assignment = {name: [] for name in names}
        sizes = Counter()

        # Reserve one component for each split.
        for name, component_id in zip(names, order[:3]):
            assignment[name].append(component_id)
            sizes[name] += len(components[component_id])

        for component_id in order[3:]:
            size = len(components[component_id])

            # Choose the split with the lowest target
            # occupancy relative to its desired size.
            name = min(
                names,
                key=lambda n: (
                    (sizes[n] + size) /
                    max(1, TARGETS[n] * total),
                    rng.random(),
                ),
            )
            assignment[name].append(component_id)
            sizes[name] += size

        score = score_split(
            assignment, components, df, labels
        )

        if score < best_score:
            best_score = score
            best = assignment

    result = df.copy()
    result["Split"] = ""

    for name, component_ids in best.items():
        indices = [
            idx
            for component_id in component_ids
            for idx in components[component_id]
        ]
        result.loc[indices, "Split"] = name

    assert result["Split"].ne("").all()

    split_frames = {
        name: result[result["Split"] == name].copy()
        for name in names
    }

    # Strict leakage checks.
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a = split_frames[first]
            b = split_frames[second]

            assert set(a["ProjectID"]).isdisjoint(
                set(b["ProjectID"])
            ), "Project leakage detected"

            keys_a = set(a["NormalizedText"].map(text_key))
            keys_b = set(b["NormalizedText"].map(text_key))

            assert keys_a.isdisjoint(keys_b), (
                "Duplicate-text leakage detected"
            )

    distribution = []

    for name in names:
        part = split_frames[name]

        part.to_csv(
            SPLITS_DIR / f"05_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        distribution.append({
            "Split": name,
            "Records": len(part),
            "Percentage": round(100 * len(part) / total, 2),
            "Projects": part["ProjectID"].nunique(),
            "ClassesPresent": part["_class_"].nunique(),
        })

    distribution_df = pd.DataFrame(distribution)
    distribution_df.to_csv(
        TABLES_DIR / "05_split_summary.csv",
        index=False,
    )

    class_table = pd.crosstab(
        result["_class_"], result["Split"]
    ).reindex(
        index=labels, columns=names, fill_value=0
    )

    class_table.to_csv(
        TABLES_DIR / "05_split_class_distribution.csv"
    )

    report = {
        "seed": SEED,
        "trials": N_TRIALS,
        "selection_score": best_score,
        "requested_proportions": TARGETS,
        "connected_components": len(components),
        "largest_component_size": max(component_sizes),
        "split_summary": distribution,
        "project_overlap": False,
        "duplicate_text_overlap": False,
        "class_counts": {
            name: {
                label: int(class_table.loc[label, name])
                for label in labels
            }
            for name in names
        },
        "limitation": (
            "Connected project/text components constrain "
            "achievable split proportions. The chosen split "
            "minimizes a heuristic score and is not guaranteed "
            "to be globally optimal or fully stratified."
        ),
    }

    report_path = REPORTS_DIR / "05_split_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nSplit summary:")
    print(distribution_df.to_string(index=False))

    print("\nClass distribution:")
    print(class_table.to_string())

    print("\nChecks:")
    print("Project overlap: 0")
    print("Duplicate-text overlap: 0")

    print("\nFiles saved to:")
    print(SPLITS_DIR)
    print(TABLES_DIR)
    print(report_path)

    if any(
        item["ClassesPresent"] < len(labels)
        for item in distribution
    ):
        print(
            "\nWARNING: Some partitions lack one or more classes. "
            "Review class coverage before model training."
        )

    print("\nSplit generation completed.")


if __name__ == "__main__":
    main()

