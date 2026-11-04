# Phase 4 Experiment Log

## Purpose

This file records the objectives, methods, results, and decisions of the Phase 4 PROMISE+ requirements classification experiments.

All model evaluation experiments use development data only. The original 550-record test partition is excluded.

## Experiment register

| ID | Experiment | Main finding |
|---|---|---|
| 01 | Group-aware TF-IDF feature comparison | Word + character Logistic Regression achieved 0.2816 mean CV macro-F1 |
| 02 | TF-IDF feature-weight tuning | No tested configuration clearly exceeded the selected baseline |
| 03 | Standard DistilBERT | Validation macro-F1: 0.1887 |
| 04 | Class-weighted DistilBERT | Validation macro-F1 improved to 0.2735 |
| 05 | Extended weighted DistilBERT | Six-epoch validation macro-F1 reached 0.3174 |
| 06 | Transformer error analysis | Identified FT and PO as unrecognized categories |
| 07 | Weighted RoBERTa | Validation macro-F1: 0.2966 |
| 08 | DistilBERT/RoBERTa complementarity | 161 prediction disagreements on 553 records |
| 09 | Three-model complementarity | 126 requirements were wrong under all three models |
| 10 | Three-model majority voting | Validation macro-F1: 0.3449, below Logistic Regression |
| 11 | Voting error analysis | Voting corrected 22 LR errors but introduced 51 new errors |
| 12 | Shared-error analysis | 126 validation requirements misclassified by all three models |
| 13 | Cross-project textual similarity | Difficult classes showed low same-class lexical similarity |
| 14 | Weighted DistilBERT group-aware CV | Mean macro-F1: 0.2442 |
| 15 | Logistic Regression/DistilBERT group CV comparison | 260 LR-only and 237 DistilBERT-only correct predictions |
| 16 | Group-aware error analysis | PO, SC, L, MN, FT, and LF had high shared-error rates |
| 17 | Balanced MiniLM + LR | Mean macro-F1: 0.2557; balanced accuracy: 0.3217 |
| 18 | MiniLM prediction diagnostics | Found excessive minority-class predictions and poor F recall |
| 19 | Unweighted MiniLM + LR | Mean accuracy: 0.6284; mean macro-F1: 0.2621 |

## Evaluation protocols

**Protocol A: Original validation**

Training: 2,574 records.

Validation: 553 records.

The models were evaluated on the original Phase 2 validation partition.

**Protocol B: Group-aware cross-validation**

Development records: 3,127.

Connected groups: 66.

Folds: 3.

Random seed: 42.

Grouping: ProjectID and matching normalized requirement text.

The same folds were reused for classical, transformer, and semantic embedding comparisons.

## Best observed results

| Criterion | Model | Score | Protocol |
|---|---|---:|---|
| Highest validation macro-F1 | Tuned Logistic Regression | 0.3894 | A |
| Highest mean CV accuracy | Unweighted MiniLM + LR | 0.6284 | B |
| Highest mean CV macro-F1 | TF-IDF + LR | 0.2816 | B |
| Highest mean CV balanced accuracy | Balanced MiniLM + LR | 0.3217 | B |

These values represent observed development results, not independent final performance claims.

## Experiment record requirements

For reproducibility, each model-development experiment should retain its script, dataset source, model identifier, hyperparameters, seed, evaluation procedure, results, and output paths.

Checkpoint selection and class-weight calculations should be documented explicitly.

Experiments using pretrained models should identify their public model checkpoints and applicable licenses.

## Final decision

Phase 4 concludes without establishing a transformer or ensemble that reliably outperforms the TF-IDF Logistic Regression baseline in project-aware macro-F1.

Phase 5 will investigate hybrid lexical-semantic representations and improved minority-class handling.

The original test partition will not be used for Phase 5 model selection.
