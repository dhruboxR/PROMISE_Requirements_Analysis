# Phase 6: Advanced Model Improvement and Annotation Consistency Analysis

## 1. Overview

Phase 6 investigates advanced classification methods, cross-project generalization difficulties, and potential annotation inconsistencies in the PROMISE+ software requirements dataset.

The primary objective was to improve the classification performance achieved in Phase 5 while maintaining a leakage-controlled evaluation protocol. Experiments included DeBERTa-v3-base fine-tuning, class-weighted learning, Linear SVM, classification error analysis, and an annotation consistency audit.

The experiments showed that the Phase 5 hybrid model remained the strongest evaluated development model. Further analysis identified significant class imbalance, project-specific distribution differences, and potential annotation inconsistencies, particularly in the Scalability (SC), Look and Feel (LF), and Portability (PO) categories.

Phase 6 concludes with a structured annotation audit and a prioritized list of requirements requiring independent source verification.

## 2. Dataset

The research uses the PROMISE+ dataset for software requirements classification.

| Property | Value |
|---|---|
| Dataset | PROMISE+ |
| Total requirements | 3,677 |
| Total software projects | 88 |
| Classification categories | 12 |
| Development requirements | 3,127 |
| Original test requirements | 550 |
| Evaluation method | Three-fold project-aware cross-validation |

The classification categories are:

| Label | Category |
|---|---|
| F | Functional |
| A | Availability |
| L | Legal |
| LF | Look and Feel |
| MN | Maintainability |
| O | Operational |
| PE | Performance |
| SC | Scalability |
| SE | Security |
| US | Usability |
| FT | Fault Tolerance |
| PO | Portability |

The original test partition was not accessed during Phase 6.

**Evaluation limitation:** The original test partition had already been examined in Phase 3, so it should not be described as a completely untouched independent test set for the overall research project.

## 3. Evaluation Protocol

Phase 6 reused the verified project-aware evaluation protocol established in Phase 4.

Three-fold cross-validation was performed on the 3,127 development requirements.

| Fold | Training requirements | Held-out requirements |
|---|---:|---:|
| 1 | 2,159 | 968 |
| 2 | 2,024 | 1,103 |
| 3 | 2,071 | 1,056 |

The evaluation protocol ensured:

- Training and held-out partitions contained no overlapping ProjectIDs.
- Identical normalized requirements did not appear across training and held-out partitions.
- The same fold assignments were reused for comparable experiments.
- Feature extraction and class weighting were fitted using training-fold data.
- The original 550-record test partition was excluded from Phase 6 experiments.

This protocol evaluates generalization to software projects not represented during model training.

## 4. Experimental Work

### Experiment 01: Evaluation Protocol Verification

Verified the development dataset, group assignments, cross-validation folds, and leakage constraints.

**Result:** All protocol checks passed.

Generated files:

- `outputs/tables/01_verified_folds.csv`
- `outputs/reports/01_protocol_audit.json`

### Experiment 02: Classification Error Analysis

Analyzed the Phase 5 hybrid model's out-of-fold predictions to identify difficult requirement categories, frequently confused labels, and projects with poor classification performance.

The analysis identified recurring problems involving SC, PO, L, MN, FT, and LF.

Frequent classification errors included:

- L predicted as F
- SC predicted as F
- LF predicted as F
- PO predicted as F
- MN predicted as F

The experiment also evaluated differences between unweighted, tempered, and balanced class-weight configurations.

### Experiment 03: DeBERTa-v3-base Fine-Tuning

Fine-tuned `microsoft/deberta-v3-base` on Fold 1 using three configurations:

1. Unweighted cross-entropy
2. Tempered class-weighted cross-entropy
3. Balanced class-weighted cross-entropy

The training environment included:

- NVIDIA GeForce RTX 4050 Laptop GPU
- FP32 model parameters
- BF16 mixed-precision computation
- Gradient accumulation
- Gradient checkpointing
- Learning rate of 5e-6

Initial numerical instability was addressed before completing the training experiments.

#### Fold 1 results

| Configuration | Accuracy | Macro-F1 | Balanced Accuracy | Predicted Categories |
|---|---:|---:|---:|---:|
| Unweighted | 65.70% | 6.61% | 8.33% | 1/12 |
| Tempered | 67.05% | 12.40% | 15.26% | 3/12 |
| Balanced | 56.82% | 12.89% | 19.26% | 5/12 |

Unweighted DeBERTa classified every held-out requirement as Functional (F).

Class weighting improved minority-category recognition, but none of the DeBERTa configurations surpassed the Phase 5 hybrid benchmark.

Only Fold 1 was evaluated for DeBERTa. These results must not be presented as three-fold cross-validation performance.

### Experiment 04: Hybrid and DeBERTa Prediction Comparison

Compared Phase 5 hybrid predictions against balanced DeBERTa predictions on the same 968 held-out Fold 1 requirements.

| Prediction outcome | Requirements |
|---|---:|
| Both models correct | 494 |
| Only hybrid correct | 173 |
| Only DeBERTa correct | 56 |
| Both models incorrect | 245 |

DeBERTa correctly classified 56 requirements missed by the hybrid, but 42 of those belonged to the Functional category.

The limited complementary performance did not provide sufficient evidence to prioritize an ensemble.

### Experiment 05: Balanced Linear SVM

Evaluated a balanced Linear Support Vector Classifier using combined word-level and character-level TF-IDF features.

The experiment used the same three project-aware folds.

| Fold | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| 1 | 63.53% | 28.86% | 31.78% |
| 2 | 57.84% | 23.31% | 25.70% |
| 3 | 61.74% | 30.82% | 30.22% |
| **Mean** | **61.04%** | **27.66%** | **29.23%** |

The Linear SVM underperformed the Phase 5 hybrid model.

### Experiment 06: Project Difficulty and Class Coverage Analysis

Compared out-of-fold predictions from the Phase 5 hybrid and Linear SVM.

Across 3,127 development requirements:

| Prediction outcome | Count |
|---|---:|
| Both models correct | 1,800 |
| Only hybrid correct | 243 |
| Only SVM correct | 105 |
| Both models incorrect | 979 |

Several projects showed consistently poor performance.

| ProjectID | Requirements | Both Models Incorrect | Both-Wrong Rate |
|---|---:|---:|---:|
| 80 | 54 | 54 | 100% |
| 74 | 16 | 16 | 100% |
| 59 | 52 | 48 | 92.31% |
| 52 | 26 | 24 | 92.31% |
| 86 | 45 | 38 | 84.44% |
| 76 | 78 | 56 | 71.79% |
| 82 | 243 | 152 | 62.55% |

Class coverage analysis found substantial differences between training and held-out projects.

Examples:

- Project 80 contained 54 SC requirements, but the corresponding training fold contained only 24 SC requirements.
- Project 82 contained 153 L requirements, compared with only 18 L requirements in the corresponding training fold.
- Projects 52 and 59 contained large proportions of PO requirements, with comparatively limited PO training examples.

These findings suggest that class imbalance and differences in requirement distributions across software projects contribute to classification difficulty.

### Experiment 07: Requirement Text Investigation

Examined examples from difficult projects and compared them against requirements with the same labels in the training folds.

Several requirements appeared semantically different from typical training examples of their assigned categories.

Examples included:

- SC-labeled requirements describing library reports and operations.
- LF-labeled requirements describing encryption, XML file naming, and data transmission.
- PO-labeled requirements describing response times, processing constraints, and hardware capacity.
- L-labeled requirements describing software compilation and data processing.

These observations motivated an annotation consistency audit.

The findings were treated as suspected annotation inconsistencies, not confirmed dataset errors.

### Experiment 08: Annotation Consistency Audit

Prepared a targeted audit containing 96 development requirements:

- 72 requirements from six difficult projects.
- 24 requirements randomly sampled from other development projects.

The AI-assisted blind review produced provisional classifications for all 96 requirements.

Original labels were preserved separately, and the original dataset was not modified.

#### Experiment 08B: Annotation Agreement Analysis

Compared the AI-assisted classifications against the original PROMISE+ labels.

| Metric | Result |
|---|---:|
| Total reviewed | 96 |
| Agreements | 36 |
| Disagreements | 60 |
| Exact agreement | 37.50% |
| Cohen's Kappa | 0.2847 |

Agreement differed between sampling groups:

| Sampling Group | Requirements | Exact Agreement |
|---|---:|---:|
| Priority projects | 72 | 30.56% |
| Control projects | 24 | 58.33% |

The most frequent disagreements were:

| Original Label | AI-Suggested Label | Count |
|---|---|---:|
| PO | PE | 11 |
| SC | F | 8 |
| LF | F | 6 |
| PO | F | 3 |

The results indicate substantial AI-to-dataset disagreement, especially among the deliberately selected difficult projects.

**Important limitation:** These results do not measure the dataset's true annotation error rate. AI-generated annotations are not independently verified ground truth, and the sample is not representative of the complete dataset.

#### Experiment 08C: Priority Source Verification Preparation

Identified 11 requirements for priority source-document verification.

The selected records primarily involve Projects 80, 86, and 52.

Generated file:

`outputs/annotation_audit/08_priority_source_verification.csv`

The verification file includes fields for source documents, section headings, source-label evidence, contextual interpretation, and independent reviewer decisions.

Source-document verification and expert adjudication remain pending.

## 5. Model Performance Summary

The strongest selected development model remains the Phase 5 hybrid classifier, combining TF-IDF lexical representations and frozen MiniLM embeddings.

| Model | Evaluation | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Phase 5 Hybrid | Three-fold mean | **65.51%** | **32.00%** |
| Phase 6 Linear SVM | Three-fold mean | 61.04% | 27.66% |
| DeBERTa Unweighted | Fold 1 only | 65.70% | 6.61% |
| DeBERTa Tempered | Fold 1 only | 67.05% | 12.40% |
| DeBERTa Balanced | Fold 1 only | 56.82% | 12.89% |

The DeBERTa results are shown for reference but are not directly equivalent to three-fold mean performance.

No Phase 6 model demonstrated sufficient evidence of improvement over the Phase 5 hybrid.

## 6. Project Structure

```text
promise_advanced_improvement_phase_6/
├── README.md
├── EXPERIMENT_LOG.md
├── scripts/
│   ├── 01_verify_evaluation_protocol.py
│   ├── 02_analyze_classification_errors.py
│   ├── 03_train_deberta_cv.py
│   ├── 03_train_deberta_tempered.py
│   ├── 03_train_deberta_balanced.py
│   ├── 04_compare_deberta_hybrid.py
│   ├── 05_linear_svm_cv.py
│   ├── 06_project_difficulty_analysis.py
│   ├── 07_prepare_annotation_audit.py
│   └── 08_analyze_annotation_agreement.py
└── outputs/
    ├── metrics/
    ├── tables/
    ├── reports/
    ├── models/
    └── annotation_audit/
        ├── 08_reviewer_1.csv
        ├── 08_reviewer_2.csv
        ├── 08_ai_assisted_annotations.csv
        ├── 08_original_label_key.csv
        ├── 08_agreement_summary.json
        ├── 08_class_agreement.csv
        ├── 08_project_agreement.csv
        ├── 08_disagreements_for_review.csv
        └── 08_priority_source_verification.csv
```

The structure lists the principal scripts and outputs. Additional generated CSV and JSON files are retained within their respective directories.

## 7. Reproducibility

### Environment

Experiments were conducted on Ubuntu with a Python virtual environment and an NVIDIA GeForce RTX 4050 Laptop GPU.

Major libraries included:

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- NumPy
- SentencePiece

### Running the experiments

From the Phase 6 directory:

```bash
python scripts/01_verify_evaluation_protocol.py
python scripts/02_analyze_classification_errors.py
python scripts/05_linear_svm_cv.py
python scripts/06_project_difficulty_analysis.py
python scripts/07_prepare_annotation_audit.py
python scripts/08_analyze_annotation_agreement.py
```

DeBERTa experiments can be run separately:

```bash
python scripts/03_train_deberta_cv.py --fold 1 --epochs 3
python scripts/03_train_deberta_tempered.py --fold 1 --epochs 3
python scripts/03_train_deberta_balanced.py --fold 1 --epochs 3
```

The scripts depend on the Phase 4 and Phase 5 project directories and their generated artifacts.

Training the transformer models also requires the relevant pretrained model files and sufficient computational resources.

## 8. Limitations

The following limitations should be considered when interpreting Phase 6:

1. DeBERTa experiments were evaluated only on Fold 1.
2. All performance measurements depend on the available labeled requirements.
3. Severe class imbalance and project-level distribution differences complicate cross-project evaluation.
4. The annotation audit deliberately oversampled difficult projects.
5. AI-to-dataset disagreements do not establish annotation errors.
6. Independent expert verification and source-document checks remain incomplete.
7. No new independent test evaluation was performed during Phase 6.
8. The original test partition was previously examined during Phase 3.

## 9. Main Findings

Phase 6 produced five major findings:

1. The Phase 5 hybrid remained the strongest selected model under the established development evaluation.
2. Unweighted DeBERTa exhibited severe majority-class prediction collapse; class weighting improved category coverage but did not surpass the hybrid.
3. Linear SVM also underperformed the hybrid under three-fold project-aware cross-validation.
4. Class imbalance and project-specific requirement distributions contributed to challenging generalization conditions.
5. A targeted annotation audit identified repeated AI-to-dataset disagreements involving SC, LF, and PO, providing motivation for independent source verification.

The analysis suggests that improving PROMISE+ classification requires attention to dataset characteristics and annotation reliability alongside model architecture and feature representation.
