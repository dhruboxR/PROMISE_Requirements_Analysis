# PROMISE+ Dataset: Phase 2 Preprocessing Report

Generated (UTC): 2026-10-31T20:54:18.801193+00:00

**Selected split:** Candidate 2

## 1. Dataset Verification

The Phase 2 input contains 3,677 requirements from 88 projects with 12 classification categories.

Phase 1 input comparison: passed.

## 2. Text Normalization

Conservative normalization applies Unicode NFC, whitespace consolidation, and trimming. Original requirement texts and labels are retained.

Modified records: 1,563.

Unchanged records: 2,114.

Unique normalized texts: 3,564.

## 3. Duplicate and Annotation Review

Exact duplicate excess records: 21.

Normalized duplicate excess records: 113.

Cross-project repeated-text groups: 89.

Conflicting-label groups: 4 (8 records).

All records and original annotations are preserved. The four annotation-conflict groups remain flagged for review.

## 4. Label Preparation

Functional requirements: 2,258.

Non-functional requirements: 1,419.

A fixed 12-class integer mapping was created. Binary labels use 0 for functional requirements and 1 for all other categories.

## 5. Split Optimization

An initial connected-component split was compared with a class-aware optimized allocation. Connected components link records sharing a project or casefolded, whitespace-normalized text.

Candidate 1 optimization score: 0.75295.

Candidate 2 optimization score: 0.26588.

Candidate 2 was selected using predefined partition-size, class-balance, and isolation criteria.

### Candidate Comparison

| Split | Records | Percentage | Projects | ClassesPresent | DistributionDistance | FunctionalPercentage | Candidate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2575 | 70.03 | 58 | 12 | 0.0317 | 59.18 | Candidate 1 |
| validation | 551 | 14.99 | 16 | 12 | 0.0955 | 60.07 | Candidate 1 |
| test | 551 | 14.99 | 14 | 12 | 0.1303 | 73.14 | Candidate 1 |
| train | 2574 | 70.0 | 50 | 12 | 0.0128 | 61.5 | Candidate 2 |
| validation | 553 | 15.04 | 19 | 12 | 0.0472 | 61.48 | Candidate 2 |
| test | 550 | 14.96 | 19 | 12 | 0.0291 | 60.91 | Candidate 2 |

## 6. Final Dataset Partitions

| Split | Records | Percentage | Projects | Classes | DistributionDistance |
| --- | --- | --- | --- | --- | --- |
| train | 2574 | 70.0 | 50 | 12 | 0.0128 |
| validation | 553 | 15.04 | 19 | 12 | 0.0472 |
| test | 550 | 14.96 | 19 | 12 | 0.0291 |

### Final Class Distribution

| Class | test | train | validation |
| --- | --- | --- | --- |
| A | 11 | 49 | 11 |
| F | 335 | 1583 | 340 |
| FT | 5 | 20 | 10 |
| L | 35 | 165 | 9 |
| LF | 10 | 67 | 12 |
| MN | 11 | 55 | 9 |
| O | 21 | 111 | 25 |
| PE | 25 | 113 | 25 |
| PO | 10 | 42 | 24 |
| SC | 9 | 70 | 16 |
| SE | 35 | 163 | 39 |
| US | 43 | 136 | 33 |

## 7. Integrity and Leakage Validation

All 3,677 records were retained exactly once.

- Project overlap across partitions: zero.
- Normalized duplicate-text overlap: zero.
- Observed classes in every partition: 12.
- Final split checksums: match frozen manifest.
- Original prepared record content: unchanged.

## 8. Reproducibility

Random seed: 42.

The frozen split checksums are stored in `08_final_split_manifest.json`. Final splits are `final_train.csv`, `final_validation.csv`, and `final_test.csv`.

Prepared master SHA-256: `f8b14f89d7b9e1d586b9a786a5d989ebe78356fb9fdb221beaf63012083b3bf5`.

## 9. Limitations

- Four label-disagreement groups remain unresolved.
- Project-level differences can influence class distributions.
- Minority categories have limited evaluation examples.
- Optimization is heuristic, not globally optimal.
- The selected split was optimized using class labels, but without model predictions.
- Contradictory requirements have not been independently annotated; label conflicts should not be treated as semantic contradictions.

## 10. Guidance for Model Development

Use the frozen train partition for fitting learned preprocessing and model parameters. Use the validation partition for tuning and model selection. Reserve the test partition for final evaluation.

Report macro-F1, balanced accuracy, per-class precision and recall, confusion matrices, and overall accuracy. Document the handling of duplicate and conflicting labels.
