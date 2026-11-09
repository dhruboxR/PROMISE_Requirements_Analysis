# PROMISE+ Requirements Analysis

**Exploratory Data Analysis, Machine Learning, and Cross-Project Software Requirements Classification**

This repository contains research on automated software requirements classification using the PROMISE+ dataset. The project focuses on dataset analysis, leakage-controlled preprocessing, traditional machine learning, transformer-based models, hybrid classification, and cross-project generalization.

**Research Status: Phases 1–5 Completed | Phase 6 Planned**

## 1. Research Objectives

- Analyze the PROMISE+ dataset and identify data-quality issues.
- Develop reproducible preprocessing and dataset-splitting procedures.
- Evaluate traditional machine learning and transformer-based classifiers.
- Improve classification performance using lexical and semantic features.
- Investigate minority-class performance and cross-project generalization.
- Develop and verify a reusable requirements classification model.

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
│   │   ├── metrics/
│   │   ├── reports/
│   │   ├── tables/
│   │   └── models/
│   ├── EXPERIMENT_LOG.md
│   └── README.md
├── README.md
├── requirements.txt
└── requirements.lock.txt
```

Each phase contains its own scripts, outputs, and documentation. Large model files and virtual environments may be excluded from Git tracking.

## 4. Research Phases

| Phase | Research Focus | Status |
|---|---|---|
| Phase 1 | Dataset validation and exploratory analysis | Completed |
| Phase 2 | Preprocessing and leakage-controlled splitting | Completed |
| Phase 3 | Traditional machine learning models | Completed |
| Phase 4 | Advanced models and comparative evaluation | Completed |
| Phase 5 | Hybrid classification and model verification | Completed |
| Phase 6 | Independent evaluation and further improvements | Planned |

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

The hybrid improved mean macro-F1 by **3.84 percentage points**.

Experiments 08–10 completed the training and serialization of the selected model, implemented raw-text inference, and verified artifact integrity, normalization consistency, feature dimensions, and repeated predictions.

The consolidated classifier was trained on **3,127 development requirements** using **48,993 combined features**.

**Documentation:** [Phase 5 README](promise_improved_model_phase_5/README.md)

**Experiment records:** [Phase 5 Experiment Log](promise_improved_model_phase_5/EXPERIMENT_LOG.md)

## 5. Technologies

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | pandas, NumPy, SciPy |
| Machine learning | scikit-learn |
| Deep learning | PyTorch |
| Language models | Hugging Face Transformers |
| Visualization | Matplotlib, Seaborn |
| Experiment records | CSV, JSON, Python scripts |
| Version control | Git, GitHub |

The project uses separate Python environments for traditional machine learning and transformer experiments.

## 6. Installation

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

## 7. Phase 5 Model Inference

After activating the transformer environment and making the locally trained Phase 5 model artifacts available:

```bash
cd promise_improved_model_phase_5

python scripts/09_predict_requirement.py \
  --text "The system shall allow users to reset their passwords." \
  --top-k 3
```

The inference pipeline normalizes the input, generates TF-IDF and MiniLM features, and predicts one of the 12 requirement categories.

The pretrained MiniLM checkpoint and saved classifier artifacts are required.

## 8. Research Limitations

- The dataset is imbalanced, with Functional requirements representing approximately 61.41% of records.
- Minority categories such as SC and PO remain difficult to classify.
- Model performance varies between software projects.
- Repeated cross-validation may influence model-selection decisions.
- The original test partition was previously evaluated in Phase 3 and is not completely untouched.
- Phase 5 improvements have not yet been confirmed through an independent evaluation.
- Requirements classification does not directly establish semantic inconsistency between requirements.

## 9. Next Research Direction

Phase 6 will focus on independent evaluation, additional robustness studies, and improving classification of difficult minority categories.

The selected Phase 5 model will serve as a reference configuration for subsequent experiments.

## 10. Research Documentation

The thesis introduction and literature review are available in `Docs/Thesis_Intro+LiteratureReview.pdf`.

Detailed methodologies, experiment results, and reproducibility records are provided in the corresponding phase directories.

## 11. Dataset Attribution

This repository is intended for academic research. The original PROMISE+ dataset, pretrained language models, and third-party software remain subject to their respective licenses and attribution requirements.

**Current Status: Phase 5 Completed. Phase 6 Planned.**