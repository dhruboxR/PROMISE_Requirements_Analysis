# PROMISE+ Requirements Analysis

**Exploratory Data Analysis, Machine Learning, and Cross-Project Software Requirements Classification**

This repository contains research on automated software requirements classification using the PROMISE+ dataset. The project focuses on dataset analysis, leakage-controlled preprocessing, traditional machine learning, transformer-based models, hybrid classification, cross-project generalization, and annotation consistency analysis.

**Research Status: Phases 1–6 Completed | Phase 7 Planned**

## 1. Research Objectives

- Analyze the PROMISE+ dataset and identify data-quality issues.
- Develop reproducible preprocessing and dataset-splitting procedures.
- Evaluate traditional machine learning and transformer-based classifiers.
- Improve classification performance using lexical and semantic features.
- Investigate minority-class performance and cross-project generalization.
- Develop and verify a reusable requirements classification model.
- Investigate potential annotation inconsistencies in difficult software projects.
- Examine original requirements documents to assess the reliability of selected dataset labels.

The classification task predicts one of 12 requirement categories from natural-language software requirements.

## 2. Dataset Information

| Property | Value |
|---|---|
| Dataset | PROMISE+ |
| Original format | ARFF |
| Total requirements | 3,677 |
| Software projects | 88 |
| Requirement categories | 12 |
| Dataset attributes | 3 |
| Missing values | 0 |
| Exact duplicate records | 21 |
| Duplicate normalized requirement texts | 112 |

**Dataset attributes:** `ProjectID`, `RequirementText`, `_class_`

**Requirement categories:** `F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`

| Code | Category |
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

The original dataset is preserved in `dataset/Promise+.arff`.

## 3. Repository Structure

```text
PROMISE_Requirements_Analysis/
├── dataset/
├── Docs/
├── promise_exploration_phase_1/
├── promise_preprocessing_phase_2/
├── promise_model_development_phase_3/
├── promise_advanced_models_phase_4/
├── promise_improved_model_phase_5/
│   ├── scripts/
│   ├── outputs/
│   ├── EXPERIMENT_LOG.md
│   └── README.md
├── promise_advanced_improvement_phase_6/
│   ├── scripts/
│   ├── outputs/
│   │   ├── metrics/
│   │   ├── reports/
│   │   ├── tables/
│   │   ├── models/
│   │   └── annotation_audit/
│   ├── EXPERIMENT_LOG.md
│   └── README.md
├── README.md
├── requirements.txt
└── requirements.lock.txt
```

Each phase contains its own scripts, outputs, and documentation. Large model checkpoints and virtual environments may be excluded from Git tracking.

Phase 7 will be maintained separately when source-document verification begins.

## 4. Research Phases

| Phase | Research Focus | Status |
|---|---|---|
| Phase 1 | Dataset validation and exploratory analysis | Completed |
| Phase 2 | Preprocessing and leakage-controlled splitting | Completed |
| Phase 3 | Traditional machine learning models | Completed |
| Phase 4 | Advanced models and comparative evaluation | Completed |
| Phase 5 | Hybrid classification and model verification | Completed |
| Phase 6 | Advanced modeling, generalization analysis, and annotation audit | Completed |
| Phase 7 | Original source-document verification and annotation quality analysis | Planned |

### Phase 1: Dataset Exploration

Validated and repaired the original ARFF dataset, examined class distributions, identified duplicates, and analyzed requirement-text characteristics.

**Documentation:** [Phase 1 README](promise_exploration_phase_1/README.md)

### Phase 2: Dataset Preprocessing

Implemented conservative text normalization, duplicate inspection, and project-aware splitting.

| Partition | Requirements |
|---|---:|
| Training | 2,574 |
| Validation | 553 |
| Test | 550 |

The partitions were checked for project and normalized-text overlap.

**Documentation:** [Phase 2 README](promise_preprocessing_phase_2/README.md)

### Phase 3: Traditional Machine Learning

Evaluated TF-IDF with Logistic Regression, Linear SVM, Naive Bayes, Random Forest, and baseline classifiers.

The selected tuned Logistic Regression model achieved:

| Metric | Validation | Original Test |
|---|---:|---:|
| Accuracy | 66.91% | 68.00% |
| Macro-F1 | 38.94% | 37.55% |
| Balanced accuracy | 41.31% | 34.65% |

The original test partition was accessed during Phase 3.

**Documentation:** [Phase 3 README](promise_model_development_phase_3/README.md)

### Phase 4: Advanced Models

Completed 19 experiments involving DistilBERT, RoBERTa, MiniLM embeddings, feature combinations, class weighting, ensembles, and group-aware cross-validation.

The strongest Phase 4 model under three-fold group-aware cross-validation was word and character TF-IDF with Logistic Regression.

- Mean accuracy: **62.54%**
- Mean macro-F1: **28.16%**
- Mean balanced accuracy: **28.27%**

**Documentation:** [Phase 4 README](promise_advanced_models_phase_4/README.md)

### Phase 5: Improved Classification Framework

Completed 10 experiments investigating hybrid lexical-semantic representations, class weighting, minority-class errors, model training, and reproducible inference.

The selected framework combines:

- Word and character TF-IDF features.
- Frozen `all-MiniLM-L6-v2` embeddings.
- Logistic Regression with tempered class weighting.

**Development cross-validation results:**

| Metric | Phase 4 Baseline | Phase 5 Hybrid |
|---|---:|---:|
| Mean accuracy | 62.54% | **65.51%** |
| Mean macro-F1 | 28.16% | **32.00%** |
| Mean balanced accuracy | 28.27% | **31.98%** |
| Mean weighted-F1 | 58.78% | **61.16%** |

The hybrid improved mean macro-F1 by **3.84 percentage points** compared with the Phase 4 baseline.

Experiments 08–10 completed training and serialization of the selected model, implemented raw-text inference, and verified artifact integrity, normalization consistency, feature dimensions, and repeated predictions.

The consolidated classifier was trained on **3,127 development requirements** using **48,993 combined features**.

**Documentation:** [Phase 5 README](promise_improved_model_phase_5/README.md)

**Experiment records:** [Phase 5 Experiment Log](promise_improved_model_phase_5/EXPERIMENT_LOG.md)

### Phase 6: Advanced Model Improvement and Annotation Consistency Analysis

Phase 6 investigated advanced classification approaches, project-level generalization difficulties, and potential annotation inconsistencies in PROMISE+.

The phase included eight principal experiments and additional analysis stages.

#### Experiment 01: Evaluation Protocol Verification

Verified the existing three-fold project-aware evaluation protocol.

| Fold | Training | Held-out |
|---|---:|---:|
| 1 | 2,159 | 968 |
| 2 | 2,024 | 1,103 |
| 3 | 2,071 | 1,056 |

All project and normalized-text leakage checks passed.

The original test partition was excluded from Phase 6.

#### Experiment 02: Classification Error Analysis

Analyzed Phase 5 out-of-fold predictions, class-specific performance, and project-level classification difficulties.

Major weaknesses involved:

- Scalability (SC)
- Portability (PO)
- Legal (L)
- Maintainability (MN)
- Fault Tolerance (FT)
- Look and Feel (LF)

Frequent errors involved minority-class requirements being classified as Functional (F).

#### Experiment 03: DeBERTa-v3-base Fine-Tuning

Fine-tuned `microsoft/deberta-v3-base` using three configurations:

- Unweighted cross-entropy.
- Tempered class-weighted cross-entropy.
- Balanced class-weighted cross-entropy.

All completed DeBERTa experiments used Fold 1.

| Configuration | Accuracy | Macro-F1 | Balanced Accuracy | Categories Predicted |
|---|---:|---:|---:|---:|
| Unweighted | 65.70% | 6.61% | 8.33% | 1/12 |
| Tempered | 67.05% | 12.40% | 15.26% | 3/12 |
| Balanced | 56.82% | 12.89% | 19.26% | 5/12 |

The unweighted model predicted only Functional requirements.

Class weighting improved minority-category coverage but did not surpass the Phase 5 hybrid.

These results are Fold 1 measurements and should not be interpreted as three-fold cross-validation results.

#### Experiment 04: Hybrid and DeBERTa Comparison

Compared Phase 5 hybrid predictions with balanced DeBERTa predictions on the same 968 held-out requirements.

| Prediction Outcome | Requirements |
|---|---:|
| Both correct | 494 |
| Only hybrid correct | 173 |
| Only DeBERTa correct | 56 |
| Both incorrect | 245 |

DeBERTa corrected 56 requirements missed by the hybrid, but 42 of those corrections belonged to the Functional category.

The results did not provide sufficient evidence to prioritize a DeBERTa-based ensemble.

#### Experiment 05: Balanced Linear SVM

Evaluated balanced Linear SVM using combined word and character TF-IDF features under three-fold project-aware cross-validation.

| Fold | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| 1 | 63.53% | 28.86% | 31.78% |
| 2 | 57.84% | 23.31% | 25.70% |
| 3 | 61.74% | 30.82% | 30.22% |
| **Mean** | **61.04%** | **27.66%** | **29.23%** |

Linear SVM underperformed the Phase 5 hybrid across all primary mean cross-validation metrics.

#### Experiment 06: Cross-Project Difficulty Analysis

Compared Phase 5 hybrid and Linear SVM out-of-fold predictions across 3,127 development requirements.

| Prediction Outcome | Count |
|---|---:|
| Both correct | 1,800 |
| Only hybrid correct | 243 |
| Only SVM correct | 105 |
| Both incorrect | 979 |

Several software projects showed severe classification difficulties.

Examples:

- Project 80: 54 of 54 requirements misclassified by both models.
- Project 74: 16 of 16 requirements misclassified by both models.
- Project 59: 48 of 52 requirements misclassified by both models.
- Project 52: 24 of 26 requirements misclassified by both models.
- Project 86: 38 of 45 requirements misclassified by both models.

Further analysis identified substantial differences in class coverage across training and held-out projects.

For example, Project 82 contained 153 Legal requirements, while the corresponding training fold contained only 18 Legal requirements.

#### Experiment 07: Requirement Text Investigation

Examined difficult-project requirements and compared them with same-class training examples.

Several requirements appeared semantically different from typical examples of their original classes.

Examples included:

- SC-labeled requirements describing library reporting and accessibility.
- LF-labeled requirements describing XML formatting, encryption, and data transmission.
- PO-labeled requirements describing response times and hardware constraints.
- L-labeled requirements describing software compilation.

These findings motivated an investigation into annotation consistency.

They were treated as suspected inconsistencies, not confirmed labeling errors.

#### Experiment 08: Annotation Consistency Audit

Prepared a targeted audit of **96 development requirements**:

- 72 sampled from six difficult projects.
- 24 sampled from other development projects as controls.

All 96 requirements received provisional AI-assisted annotations without consulting the original labels during the initial review.

The original labels were preserved separately.

**Experiment 08B: Agreement Analysis**

| Metric | Result |
|---|---:|
| Requirements reviewed | 96 |
| Agreements | 36 |
| Disagreements | 60 |
| Exact agreement | 37.50% |
| Cohen's Kappa | 0.2847 |

Agreement by sampling group:

| Sampling Group | Requirements | Agreement |
|---|---:|---:|
| Priority projects | 72 | 30.56% |
| Control projects | 24 | 58.33% |

Frequent disagreement patterns:

| Original Label | AI-Suggested Label | Count |
|---|---|---:|
| PO | PE | 11 |
| SC | F | 8 |
| LF | F | 6 |
| PO | F | 3 |

The results indicate substantial disagreement between the provisional AI annotations and the existing dataset classifications.

**These disagreements are not confirmed annotation errors.** The audit used a targeted, nonrepresentative sample and did not include completed independent expert adjudication.

**Experiment 08C: Priority Source Verification Preparation**

Identified 11 suspected mismatches for source-document verification, primarily from Projects 80, 86, and 52.

Prepared:

`outputs/annotation_audit/08_priority_source_verification.csv`

Verification against the original requirements documents and independent expert review remain pending and are planned for Phase 7.

**Documentation:** [Phase 6 README](promise_advanced_improvement_phase_6/README.md)

**Experiment records:** [Phase 6 Experiment Log](promise_advanced_improvement_phase_6/EXPERIMENT_LOG.md)

## 5. Overall Model Comparison

The strongest selected development model remains the Phase 5 hybrid classifier.

| Model | Evaluation | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|
| Phase 4 TF-IDF + LR | Three-fold mean | 62.54% | 28.16% | 28.27% |
| **Phase 5 Hybrid** | **Three-fold mean** | **65.51%** | **32.00%** | **31.98%** |
| Phase 6 Linear SVM | Three-fold mean | 61.04% | 27.66% | 29.23% |
| DeBERTa Unweighted | Fold 1 only | 65.70% | 6.61% | 8.33% |
| DeBERTa Tempered | Fold 1 only | 67.05% | 12.40% | 15.26% |
| DeBERTa Balanced | Fold 1 only | 56.82% | 12.89% | 19.26% |

The DeBERTa results are included as exploratory Fold 1 comparisons and are not directly equivalent to three-fold mean results.

No Phase 6 model demonstrated a validated improvement over the Phase 5 hybrid.

## 6. Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | pandas, NumPy, SciPy |
| Machine learning | scikit-learn |
| Deep learning | PyTorch |
| Language models | Hugging Face Transformers, Sentence Transformers |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Accuracy, Macro-F1, Balanced Accuracy, Cohen's Kappa |
| Experiment records | CSV, JSON, Python scripts |
| Version control | Git, GitHub |

The project uses separate Python environments for traditional machine learning and transformer experiments.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/dhruboxR/PROMISE_Requirements_Analysis.git
cd PROMISE_Requirements_Analysis
```

Create the main environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Transformer experiments use a separate `.venv-transformers` environment. Its recorded dependencies are available in `promise_advanced_models_phase_4/requirements-transformers.lock.txt`.

Refer to each phase's README for its execution instructions.

## 8. Phase 5 Model Inference

After activating the transformer environment and making the locally trained Phase 5 model artifacts available:

```bash
cd promise_improved_model_phase_5

python scripts/09_predict_requirement.py \
  --text "The system shall allow users to reset their passwords." \
  --top-k 3
```

The inference pipeline normalizes the input, generates TF-IDF and MiniLM features, and predicts one of the 12 requirement categories.

The pretrained MiniLM checkpoint and saved classifier artifacts are required.

## 9. Research Limitations

- The dataset is imbalanced, with Functional requirements representing approximately 61.41% of records.
- Minority categories such as SC, PO, MN, LF, and FT remain difficult to classify.
- Model performance varies substantially between software projects.
- Some project-aware training folds contain limited examples of particular requirement classes.
- Repeated development cross-validation may influence model-selection decisions.
- DeBERTa experiments in Phase 6 were conducted only on Fold 1.
- The original test partition was previously evaluated in Phase 3 and is not completely untouched.
- Phase 5 improvements have not been confirmed through a new independent test evaluation.
- The annotation audit used a targeted, nonrepresentative sample.
- AI-assisted annotation disagreements do not establish incorrect ground-truth labels.
- Independent expert review and original source-document verification remain incomplete.
- Requirements classification does not directly establish semantic inconsistency between requirements.

## 10. Next Research Direction: Phase 7

**Phase 7: Original Source Document Verification and Annotation Quality Analysis**

Phase 7 will investigate the original requirements documents associated with selected PROMISE+ records.

Planned activities include:

1. Inspecting original PURE requirements documents.
2. Matching selected PROMISE+ requirements to their source specifications.
3. Identifying source-document sections and original classification evidence.
4. Verifying 11 priority annotation disagreements.
5. Documenting whether the surrounding context supports the original labels.
6. Preparing an evidence-based annotation quality report.
7. Considering independently verified corrections in a separate experimental dataset, if justified.

The existing PROMISE+ dataset, Phase 5 classifier, and Phase 6 results will remain unchanged during source verification.

## 11. Research Documentation

The thesis introduction and literature review are available in `Docs/Thesis_Intro+LiteratureReview.pdf`.

Detailed methodologies, scripts, experimental results, and reproducibility records are provided in the corresponding phase directories.

Relevant documentation:

- [Phase 1 README](promise_exploration_phase_1/README.md)
- [Phase 2 README](promise_preprocessing_phase_2/README.md)
- [Phase 3 README](promise_model_development_phase_3/README.md)
- [Phase 4 README](promise_advanced_models_phase_4/README.md)
- [Phase 5 README](promise_improved_model_phase_5/README.md)
- [Phase 6 README](promise_advanced_improvement_phase_6/README.md)

## 12. Dataset Attribution

This repository is intended for academic research.

The PROMISE+ dataset was introduced by Silva et al. in *Promise+: Expandindo a base de dados de requisitos de software Promise_exp*, published at SBES 2024.

**PROMISE+ Dataset:** https://zenodo.org/records/12805484

**PURE Requirements Dataset:** https://zenodo.org/records/7118517

The original datasets, pretrained language models, and third-party software remain subject to their respective licenses and attribution requirements.

---

**Current Research Status:** Phases 1–6 Completed. Phase 7 Planned.

**Best Selected Development Model:** Phase 5 Hybrid, 65.51% mean project-aware cross-validation accuracy and 32.00% mean macro-F1.

**Next Research Activity:** Original source-document verification and annotation quality analysis.
