# Phase 2: PROMISE+ Dataset Preprocessing

This directory contains the preprocessing, label preparation, duplicate analysis, and leakage-aware dataset splitting workflow for the PROMISE+ requirements dataset.

## Objectives

- Verify the validated dataset from Phase 1.
- Normalize requirement text without changing its meaning.
- Identify repeated requirements and conflicting annotations.
- Prepare consistent multiclass and binary labels.
- Generate project- and duplicate-aware dataset partitions.
- Optimize class balance while maintaining data isolation.
- Validate and freeze the final partitions.

## Dataset

| Property | Value |
|---|---:|
| Total requirements | 3,677 |
| Projects | 88 |
| Requirement classes | 12 |
| Modified texts during normalization | 1,563 |
| Normalized duplicate excess records | 113 |
| Conflicting-label groups | 4 |
| Records removed | 0 |

The original requirement texts, project identifiers, and classification labels are preserved.

## Scripts

| Script | Purpose |
|---|---|
| `01_verify_input.py` | Verify Phase 1 dataset integrity |
| `02_normalize_text.py` | Unicode NFC and whitespace normalization |
| `03_review_duplicates.py` | Duplicate and annotation-conflict analysis |
| `04_prepare_labels.py` | Multiclass and binary label preparation |
| `05_create_splits.py` | Initial project- and duplicate-aware splitting |
| `06_validate_splits.py` | Independent validation of Candidate 1 |
| `07_optimize_splits.py` | Class-aware allocation optimization |
| `08_finalize_splits.py` | Validate and freeze Candidate 2 |
| `09_generate_preprocessing_report.py` | Consolidated preprocessing report |

## Final Dataset Split

Candidate 2 was selected using predefined partition-size, class-balance, and isolation criteria, without model-performance results.

| Split | Records | Percentage | Projects | Classes |
|---|---:|---:|---:|---:|
| Training | 2,574 | 70.00% | 50 | 12 |
| Validation | 553 | 15.04% | 19 | 12 |
| Test | 550 | 14.96% | 19 | 12 |

### Class-distribution distance

| Split | Total variation distance |
|---|---:|
| Training | 0.0128 |
| Validation | 0.0472 |
| Test | 0.0291 |

## Leakage Prevention

The splitting procedure groups records that share a project identifier or normalized requirement text.

Validation confirmed:

- No project overlap between partitions.
- No normalized-text overlap between partitions.
- No missing or duplicated RecordIDs.
- All original records preserved.
- All 12 classes represented in every partition.

## Final Output Files

The authoritative datasets are stored in `outputs/splits/`:

```text
final_train.csv
final_validation.csv
final_test.csv
```

The corresponding checksums and metadata are stored in:

`outputs/reports/08_final_split_manifest.json`

The complete preprocessing report is available at:

`outputs/reports/09_final_preprocessing_report.md`

## Reproducing the Workflow

From the repository root, activate the shared Python environment:

```bash
source .venv/bin/activate
cd promise_preprocessing_phase_2
```

Install the dependencies from the Phase 1 requirements file if needed:

```bash
python -m pip install -r ../promise_exploration_phase_1/requirements.txt
```

Run the scripts in numerical order:

```bash
python scripts/01_verify_input.py
python scripts/02_normalize_text.py
python scripts/03_review_duplicates.py
python scripts/04_prepare_labels.py
python scripts/05_create_splits.py
python scripts/06_validate_splits.py
python scripts/07_optimize_splits.py
python scripts/08_finalize_splits.py
python scripts/09_generate_preprocessing_report.py
```

The finalization script refuses to overwrite frozen split files when their contents differ. Preserve the approved datasets and manifest when reproducing later experiments.

## Methodological Limitations

- Four annotation-conflict groups remain unresolved.
- Minority classes have relatively few validation and test examples.
- Connected components restrict possible dataset partitions.
- The optimization procedure is heuristic.
- Class labels were used for balancing partitions, but model results were not used to choose the split.
- Classification-label disagreement does not establish semantic contradiction between requirements.

## Model Development Policy

Future experiments should fit learned preprocessing transformations and model parameters using the training partition. The validation partition should support model selection and tuning.

The test partition must remain fixed and should be used for final evaluation without repeatedly selecting models based on test performance.

Recommended classification metrics include macro-F1, balanced accuracy, per-class precision, per-class recall, and confusion matrices.

**Status: Phase 2 preprocessing completed. Phase 3 model development is planned.**