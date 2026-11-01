
"""Optimize PROMISE+ splits while preserving project/text isolation.

Candidate 1 is never overwritten.
Optimization uses labels and component membership only, not model results.
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from common import (
    CLEANED_DIR, SPLITS_DIR, REPORTS_DIR,
    TABLES_DIR, prepare_outputs
)

SEED = 42
ITERATIONS = 20000

NAMES = ["train", "validation", "test"]
TARGET = np.array([0.70, 0.15, 0.15])
MASTER = CLEANED_DIR / "04_labeled_requirements.csv"


def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"ProjectID": str, "_class_": str},
        keep_default_na=False,
    )


def key(text):
    return " ".join(str(text).casefold().split())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_components(df):
    uf = UnionFind(len(df))
    projects = {}
    texts = {}

    for i, row in enumerate(df.itertuples(index=False)):
        # Explicit dataframe access avoids dependency
        # on how pandas renames unusual column names.
        project = str(df.at[i, "ProjectID"])
        text = key(df.at[i, "NormalizedText"])

        if project in projects:
            uf.union(i, projects[project])
        else:
            projects[project] = i

        if text in texts:
            uf.union(i, texts[text])
        else:
            texts[text] = i

    groups = defaultdict(list)
    for i in range(len(df)):
        groups[uf.find(i)].append(i)

    return list(groups.values())


def make_statistics(df, components, labels):
    label_index = {label: i for i, label in enumerate(labels)}

    counts = np.zeros(
        (len(components), len(labels)), dtype=np.int64
    )
    sizes = np.zeros(len(components), dtype=np.int64)

    for component_id, indices in enumerate(components):
        sizes[component_id] = len(indices)

        for label in df.iloc[indices]["_class_"]:
            counts[component_id, label_index[label]] += 1

    return counts, sizes


def objective(assignment, component_counts, sizes):
    """Lower score means better size and class balance."""
    n = sizes.sum()
    all_counts = component_counts.sum(axis=0)
    global_prop = all_counts / n

    split_counts = np.zeros((3, len(all_counts)), dtype=float)
    split_sizes = np.zeros(3, dtype=float)

    for split_id in range(3):
        mask = assignment == split_id
        split_sizes[split_id] = sizes[mask].sum()
        split_counts[split_id] = component_counts[mask].sum(axis=0)

    if np.any(split_sizes == 0):
        return 1e9

    size_error = np.abs(split_sizes / n - TARGET).sum()

    # Total variation distance for each partition.
    class_error = 0.0
    minority_error = 0.0

    for split_id in range(3):
        actual = split_counts[split_id] / split_sizes[split_id]
        class_error += 0.5 * np.abs(actual - global_prop).sum()

        expected = all_counts * TARGET[split_id]

        # Penalize large relative deficits, especially for
        # underrepresented classes.
        deficit = np.maximum(0, expected - split_counts[split_id])
        minority_error += np.mean(
            deficit / np.maximum(expected, 1)
        )

    missing = (split_counts == 0).sum()

    return float(
        6.0 * size_error
        + 2.0 * class_error
        + 0.4 * minority_error
        + 20.0 * missing
    )


def verify(df, assignments):
    parts = {
        name: df.iloc[
            np.flatnonzero(assignments == split_id)
        ].copy()
        for split_id, name in enumerate(NAMES)
    }

    assert sum(len(p) for p in parts.values()) == len(df)
    assert all(len(p) > 0 for p in parts.values())

    for a, b in combinations(NAMES, 2):
        assert set(parts[a]["ProjectID"]).isdisjoint(
            set(parts[b]["ProjectID"])
        )

        keys_a = set(parts[a]["NormalizedText"].map(key))
        keys_b = set(parts[b]["NormalizedText"].map(key))

        assert keys_a.isdisjoint(keys_b)

    all_labels = set(df["_class_"])
    assert all(
        set(part["_class_"]) == all_labels
        for part in parts.values()
    )

    return parts


def summarize(parts, full):
    global_prop = full["_class_"].value_counts(normalize=True)

    rows = []
    for name, part in parts.items():
        p = part["_class_"].value_counts(normalize=True)

        tv = 0.5 * sum(
            abs(p.get(label, 0) - global_prop.get(label, 0))
            for label in global_prop.index
        )

        rows.append({
            "Split": name,
            "Records": len(part),
            "Percentage": round(100 * len(part) / len(full), 2),
            "Projects": part["ProjectID"].nunique(),
            "ClassesPresent": part["_class_"].nunique(),
            "DistributionDistance": round(tv, 4),
            "FunctionalPercentage": round(
                100 * p.get("F", 0), 2
            ),
        })

    return pd.DataFrame(rows)


def main():
    prepare_outputs()

    df = read_csv(MASTER).reset_index(drop=True)

    if len(df) != 3677 or not df["RecordID"].is_unique:
        raise ValueError("Unexpected master dataset.")

    original_paths = {
        name: SPLITS_DIR / f"05_{name}.csv"
        for name in NAMES
    }

    original_hashes = {
        name: sha256(path)
        for name, path in original_paths.items()
    }

    original_parts = {
        name: read_csv(path)
        for name, path in original_paths.items()
    }

    record_to_split = {}
    for split_id, name in enumerate(NAMES):
        for record_id in original_parts[name]["RecordID"]:
            if record_id in record_to_split:
                raise ValueError("Duplicate Candidate 1 RecordID")
            record_to_split[record_id] = split_id

    if set(record_to_split) != set(df["RecordID"]):
        raise ValueError("Candidate 1 does not cover master records")

    components = build_components(df)
    labels = sorted(df["_class_"].unique())
    counts, sizes = make_statistics(df, components, labels)

    # Start optimization from the previously validated split.
    initial = np.empty(len(components), dtype=np.int64)

    for component_id, indices in enumerate(components):
        split_ids = {
            record_to_split[df.at[i, "RecordID"]]
            for i in indices
        }
        if len(split_ids) != 1:
            raise ValueError("Candidate 1 splits a connected component")
        initial[component_id] = split_ids.pop()

    rng = np.random.default_rng(SEED)

    current = initial.copy()
    current_score = objective(current, counts, sizes)

    best = current.copy()
    best_score = current_score
    initial_score = current_score

    # Simulated annealing: move or swap entire components.
    for iteration in range(ITERATIONS):
        proposal = current.copy()

        if rng.random() < 0.65:
            comp = int(rng.integers(len(components)))
            new_split = int(rng.integers(3))
            proposal[comp] = new_split
        else:
            a, b = rng.choice(
                len(components), size=2, replace=False
            )
            proposal[a], proposal[b] = proposal[b], proposal[a]

        if len(set(proposal.tolist())) < 3:
            continue

        proposed_score = objective(proposal, counts, sizes)

        # Gradually reduce acceptance of worse solutions.
        progress = iteration / ITERATIONS
        temperature = 0.15 * (0.001 / 0.15) ** progress

        delta = proposed_score - current_score

        if delta < 0 or rng.random() < math.exp(
            -delta / temperature
        ):
            current = proposal
            current_score = proposed_score

        if current_score < best_score:
            best = current.copy()
            best_score = current_score

    # Expand component assignments to row assignments.
    row_assignments = np.empty(len(df), dtype=np.int64)
    for component_id, indices in enumerate(components):
        row_assignments[indices] = best[component_id]

    candidate2 = verify(df, row_assignments)

    # Reconstruct Candidate 1 in the same column layout.
    candidate1 = {
        name: original_parts[name]
        for name in NAMES
    }

    summary1 = summarize(candidate1, df)
    summary2 = summarize(candidate2, df)

    comparison = pd.concat([
        summary1.assign(Candidate="Candidate 1"),
        summary2.assign(Candidate="Candidate 2"),
    ], ignore_index=True)

    comparison.to_csv(
        TABLES_DIR / "07_candidate_comparison.csv",
        index=False,
    )

    class_counts = pd.DataFrame({
        name: candidate2[name]["_class_"]
        .value_counts()
        .reindex(labels, fill_value=0)
        for name in NAMES
    })

    class_counts.index.name = "Class"
    class_counts.to_csv(
        TABLES_DIR / "07_candidate2_class_distribution.csv"
    )

    for name, part in candidate2.items():
        part = part.copy()
        part["Split"] = name
        part.to_csv(
            SPLITS_DIR / f"07_candidate2_{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # Ensure Candidate 1 has not been altered.
    assert all(
        sha256(original_paths[name]) == original_hashes[name]
        for name in NAMES
    )

    report = {
        "seed": SEED,
        "iterations": ITERATIONS,
        "target_proportions": dict(zip(
            NAMES, TARGET.tolist()
        )),
        "component_count": len(components),
        "largest_component": int(sizes.max()),
        "candidate1_score": initial_score,
        "candidate2_score": best_score,
        "objective_improved": best_score < initial_score,
        "candidate1_sha256": original_hashes,
        "candidate1": summary1.to_dict(orient="records"),
        "candidate2": summary2.to_dict(orient="records"),
        "validation": {
            "project_overlap": 0,
            "normalized_text_overlap": 0,
            "all_classes_present": True,
        },
        "note": (
            "Heuristic optimization, not a proof of global "
            "optimality. Split selection must be frozen before "
            "model training and test evaluation."
        ),
    }

    report_path = REPORTS_DIR / "07_optimization_report.json"
    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8",
    )

    print("\nPROMISE+ PHASE 2: SPLIT OPTIMIZATION")
    print("=" * 55)
    print("Components:", len(components))
    print("Largest component:", sizes.max())
    print("Candidate 1 score:", round(initial_score, 5))
    print("Candidate 2 score:", round(best_score, 5))

    print("\nCandidate comparison:")
    print(comparison.to_string(index=False))

    print("\nCandidate 2 class counts:")
    print(class_counts.to_string())

    print("\nProject overlap: 0")
    print("Duplicate-text overlap: 0")
    print("All classes present: Yes")
    print("Candidate 1 files unchanged: Yes")
    print("\nReport:", report_path)


if __name__ == "__main__":
    main()
