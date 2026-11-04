# PROMISE+ Requirements Analysis

**Exploratory Data Analysis, Machine Learning, and Cross-Project Software Requirements Classification**

This repository contains research on the PROMISE+ software requirements dataset, focusing on dataset quality, preprocessing, automated requirements classification, and model generalization across software projects.

The research follows a structured, reproducible workflow covering dataset exploration, leakage-controlled preprocessing, traditional machine learning, transformer-based classification, semantic embeddings, and comparative evaluation.

**Current research status:** Phases 1, 2, 3, and 4 have been completed. Phase 5 will focus on developing improved requirements classification methods based on the findings from previous experiments.

---

## 1. Research Objectives

The primary objectives of this research are:

- Explore and analyze the PROMISE+ requirements dataset.
- Identify data-quality issues, duplicate requirements, and potential annotation inconsistencies.
- Examine requirement-class distributions and class imbalance.
- Analyze linguistic characteristics and requirement lengths.
- Establish a reproducible dataset preprocessing and validation workflow.
- Develop traditional machine learning models for automated requirements classification.
- Investigate transformer-based models and pretrained semantic embeddings.
- Evaluate classification performance on previously unseen software projects.
- Identify difficult requirement classes and investigate cross-project generalization challenges.
- Develop improved classification methods based on experimental findings.
- Investigate potential requirements inconsistencies as a related research direction.

The classification task predicts one of 12 predefined requirement categories from the natural-language text of a software requirement.

---

## 2. Dataset Information

The project uses the PROMISE+ software requirements dataset, provided in ARFF format.

| Property | Description |
|---|---|
| Dataset | PROMISE+ |
| Original format | ARFF |
| Total requirements | 3,677 |
| Unique projects | 88 |
| Dataset attributes | 3 |
| Requirement categories | 12 |
| Missing values | 0 |
| Blank requirement texts | 0 |
| Exact duplicate records | 21 |
| Duplicate requirement texts (normalized) | 112 |

### Dataset Attributes

| Attribute | Description |
|---|---|
| `ProjectID` | Identifier of the software project |
| `RequirementText` | Natural-language software requirement |
| `_class_` | Classification label assigned to the requirement |

The dataset contains 12 requirement categories:

`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`

### Dataset Preservation

The original dataset is preserved at:

`dataset/Promise+.arff`

Phase 1 produces a structurally repaired working copy and an audit report documenting ARFF formatting corrections.

The repair process preserves the original records and classification labels. Any unresolved annotation conflicts are documented rather than silently modified.

---

## 3. Repository Structure

```text
PROMISE_Requirements_Analysis/
├── dataset/
│   ├── Lista dos tipos de software.xlsx
│   └── Promise+.arff
│
├── Docs/
│   └── Thesis_Intro+LiteratureReview.pdf
│
├── promise_exploration_phase_1/
│   ├── code_outputs_phase1.txt
│   ├── outputs/
│   ├── README.md
│   └── scripts/
│
├── promise_preprocessing_phase_2/
│   ├── code_outputs_phase2.txt
│   ├── outputs/
│   ├── README.md
│   └── scripts/
│
├── promise_model_development_phase_3/
│   ├── code_outputs_phase3.txt
│   ├── outputs/
│   ├── PROMISE_Phase_3_Final_Report.pdf
│   ├── README.md
│   └── scripts/
│
├── promise_advanced_models_phase_4/
│   ├── code_outputs_phase4.txt
│   ├── configs/
│   ├── EXPERIMENT_LOG.md
│   ├── FINDINGS_AND_LIMITATIONS.md
│   ├── outputs/
│   ├── PHASE4_FILE_INVENTORY.txt
│   ├── README.md
│   ├── requirements-transformers.lock.txt
│   ├── requirements.txt
│   └── scripts/
│
├── README.md
├── requirements.lock.txt
└── requirements.txt
```

The structure above reflects the repository workspace at the completion of Phase 4.

Large generated model checkpoints and local virtual environments may be excluded from Git tracking. As a result, some local outputs may not be available directly from the GitHub repository.

---

## 4. Research Workflow and Progress

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Dataset validation and exploratory data analysis | Completed |
| Phase 2 | Data preprocessing and leakage-controlled splitting | Completed |
| Phase 3 | Traditional machine learning model development | Completed |
| Phase 4 | Advanced models, comparative evaluation, and error analysis | Completed |
| Phase 5 | Improved requirements classification framework | Planned |
| Phase 6 | Final evaluation and thesis reporting | Planned |

### Phase 1: Dataset Exploration

Phase 1 investigates the original PROMISE+ dataset and prepares it for subsequent processing.

The main activities include:

- ARFF validation and repair.
- Dataset schema and attribute inspection.
- Missing-value and duplicate analysis.
- Requirement class-distribution analysis.
- Requirement length and linguistic analysis.
- Documentation of potential annotation conflicts.

The original dataset contains 3,677 requirements across 88 software projects.

The dataset is highly imbalanced, with Functional (`F`) requirements representing approximately 61.41% of all records.

**Documentation:** `promise_exploration_phase_1/README.md`

### Phase 2: Dataset Preprocessing and Splitting

Phase 2 prepares the dataset for model training and evaluation while minimizing data leakage.

The preprocessing workflow includes:

- Requirement text normalization.
- Preservation of original requirement records and labels.
- Duplicate and annotation-conflict inspection.
- Project-aware dataset splitting.
- Separation of identical normalized texts across partitions.
- Dataset integrity checks and checksum verification.

The final selected split contains:

| Partition | Requirements | Projects |
|---|---:|---:|
| Training | 2,574 | 50 |
| Validation | 553 | 19 |
| Test | 550 | 19 |
| **Total** | **3,677** | **88** |

The partitions have no project overlap or normalized-text overlap.

The split was selected using predefined data-quality and class-balance criteria, without model-performance-based selection.

**Documentation:** `promise_preprocessing_phase_2/README.md`

### Phase 3: Traditional Machine Learning Models

Phase 3 establishes baseline models for automated requirements classification.

The evaluated approaches include:

- Majority-class baseline.
- TF-IDF with Logistic Regression.
- TF-IDF with Linear Support Vector Machine.
- Multinomial Naive Bayes.
- Random Forest.
- Hyperparameter-tuned Logistic Regression and Linear SVM.

The best selected Phase 3 model was tuned Logistic Regression.

**Original validation results:**

| Metric | Score |
|---|---:|
| Accuracy | 66.91% |
| Macro-F1 | 38.94% |
| Balanced accuracy | 41.31% |
| Weighted-F1 | 65.05% |

**Phase 3 test evaluation:**

| Metric | Score |
|---|---:|
| Accuracy | 68.00% |
| Macro-F1 | 37.55% |
| Balanced accuracy | 34.65% |
| Weighted-F1 | 63.78% |

The selected model was evaluated on the original test partition in Phase 3. Because those test results and class-level errors have already been examined, this partition is not treated as a completely untouched independent test for future model-selection decisions.

**Documentation:** `promise_model_development_phase_3/README.md`

**Report:** `promise_model_development_phase_3/PROMISE_Phase_3_Final_Report.pdf`

### Phase 4: Advanced Models and Comparative Evaluation

Phase 4 investigates whether advanced text representations and model architectures can improve classification, particularly for minority classes and unseen software projects.

The investigated approaches include:

- Word and character TF-IDF feature combinations.
- DistilBERT fine-tuning.
- Class-weighted DistilBERT.
- Extended DistilBERT training.
- Class-weighted RoBERTa.
- Frozen MiniLM sentence embeddings.
- Weighted and unweighted Logistic Regression with semantic embeddings.
- Model complementarity analysis.
- Majority-voting ensembles.
- Class-level and project-level error analysis.
- Cross-project lexical similarity analysis.
- Group-aware cross-validation.

Experiments 01–19 were completed and documented.

#### Original validation comparison

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Tuned Logistic Regression | **66.91%** | **38.94%** | **41.31%** |
| Standard DistilBERT | 55.70% | 18.87% | 22.28% |
| Weighted DistilBERT, 3 epochs | 59.86% | 27.35% | 32.11% |
| Weighted DistilBERT, 6 epochs | 56.78% | 31.74% | 37.58% |
| Weighted RoBERTa | 52.44% | 29.66% | 33.76% |
| Three-model majority voting | 61.66% | 34.49% | 38.29% |

These experiments use the original 553-record validation partition.

#### Three-fold group-aware cross-validation

For cross-project evaluation, the original training and validation partitions were combined into 3,127 development requirements.

The evaluation used three-fold `StratifiedGroupKFold` with random seed 42.

Connected grouping ensured that requirements belonging to the same software project or sharing identical normalized text remained within the same group.

| Model | Mean accuracy | Mean macro-F1 | Mean balanced accuracy |
|---|---:|---:|---:|
| Word + character TF-IDF Logistic Regression | 62.54% | **28.16%** | 28.27% |
| Weighted DistilBERT | 61.72% | 24.42% | 27.93% |
| Balanced MiniLM + Logistic Regression | 46.99% | 25.57% | **32.17%** |
| Unweighted MiniLM + Logistic Regression | **62.84%** | 26.21% | 26.47% |

These values are mean scores across three folds, not pooled out-of-fold scores.

The word and character TF-IDF Logistic Regression model achieved the highest mean cross-validation macro-F1 among the evaluated approaches.

#### Main Phase 4 Findings

1. Class weighting substantially improved DistilBERT's minority-class recognition, but did not consistently outperform Logistic Regression.
2. Longer transformer fine-tuning improved macro-F1 in the original validation experiment but reduced overall accuracy.
3. RoBERTa did not outperform the strongest DistilBERT configuration in the tested setup.
4. Majority voting performed worse than Logistic Regression alone.
5. MiniLM embeddings offered an alternative semantic representation, but performance was sensitive to class weighting.
6. PO, SC, L, MN, FT, and LF remained particularly difficult under cross-project evaluation.
7. Several difficult requirements showed low lexical similarity to same-class training examples.
8. Large performance differences between software projects indicate persistent generalization challenges.

These findings motivate further work on combined lexical and semantic representations and improved handling of minority classes.

**Phase 4 documentation:**

- `promise_advanced_models_phase_4/README.md`
- `promise_advanced_models_phase_4/EXPERIMENT_LOG.md`
- `promise_advanced_models_phase_4/FINDINGS_AND_LIMITATIONS.md`

---

## 5. Technologies and Libraries

The research uses Python and the following tools and libraries.

| Category | Technologies |
|---|---|
| Programming | Python |
| Data processing | pandas, NumPy, liac-arff |
| Visualization | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Deep learning | PyTorch |
| Pretrained language models | Hugging Face Transformers |
| Experiment tracking | CSV, JSON, Python scripts |
| Version control | Git and GitHub |

### Pretrained Models Investigated

- `distilbert-base-uncased`
- `roberta-base`
- `sentence-transformers/all-MiniLM-L6-v2`

These models are publicly pretrained architectures adapted or evaluated for requirements classification. The pretrained architectures themselves are not claimed as original contributions of this research.

### Dependency Files

The repository contains:

- `requirements.txt`
- `requirements.lock.txt`
- `promise_advanced_models_phase_4/requirements.txt`
- `promise_advanced_models_phase_4/requirements-transformers.lock.txt`

The lock files record package versions used in the respective research environments. Installation compatibility may depend on the Python version and operating system.

---

## 6. Installation and Setup

### Clone the repository

```bash
git clone https://github.com/dhruboxR/PROMISE_Requirements_Analysis.git
cd PROMISE_Requirements_Analysis
```

### Create an environment for dataset analysis and classical ML

On Ubuntu or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
python -m pip install -r requirements.txt
```

For exact recorded package versions, use `requirements.lock.txt` where compatible.

### Create an environment for transformer experiments

A separate Python environment is recommended for Phase 4 transformer experiments.

```bash
python3 -m venv .venv-transformers
source .venv-transformers/bin/activate
```

Install dependencies according to:

`promise_advanced_models_phase_4/requirements.txt`

GPU-enabled PyTorch installation depends on the operating system, NVIDIA driver, and supported CUDA wheel. Follow the official PyTorch installation instructions for the target machine.

The recorded transformer environment is documented in:

`promise_advanced_models_phase_4/requirements-transformers.lock.txt`

GPU acceleration is recommended for transformer fine-tuning. Traditional machine learning and saved-output analysis can generally run on CPU.

---

## 7. Running the Research Experiments

Each phase contains its own `scripts/` directory and documentation.

### Phase 1: Exploration

```bash
cd promise_exploration_phase_1
```

Read its `README.md` for the documented validation and exploration sequence.

### Phase 2: Preprocessing

```bash
cd promise_preprocessing_phase_2
```

Run the scripts in the documented order to reproduce the preprocessing outputs and dataset splits.

### Phase 3: Traditional ML

```bash
cd promise_model_development_phase_3
```

The model-development scripts and saved evaluation results are documented in the phase README.

### Phase 4: Advanced Models

```bash
cd promise_advanced_models_phase_4
```

The phase contains numbered experiment scripts covering model development, diagnostic analysis, and cross-validation.

For example, to reproduce the group-aware classical baseline comparison:

```bash
python scripts/01_group_aware_feature_comparison.py
```

The corresponding output directories contain experiment metrics, tables, reports, and saved models where available.

Refer to `EXPERIMENT_LOG.md` for the full experiment sequence.

Reproducing transformer experiments requires the appropriate PyTorch and Transformers environment.

---

## 8. Research Considerations and Limitations

The following considerations are important for interpreting the results.

### Class imbalance

Functional requirements represent approximately 61.41% of the dataset.

Minority categories contain substantially fewer examples, affecting class-level recall, precision, and F1 scores.

### Cross-project generalization

Different software projects use different terminology, domains, and requirement-writing conventions.

Performance on familiar projects does not necessarily represent performance on unseen projects.

### Potential annotation conflicts

Four normalized-text groups contain conflicting classification labels.

These groups require careful inspection, and their labels should not be changed solely to improve model performance.

### Evaluation separation

Phase 4 experiments excluded the original test partition and used training and validation development data.

However, repeated experimentation with the same development folds can influence model selection. Consequently, improvements found on these folds should be treated as exploratory until independently confirmed.

### Requirements classification versus inconsistency detection

The 12 existing labels support requirements classification.

They do not directly establish whether two requirements contradict each other or contain other forms of semantic inconsistency.

Any future inconsistency-detection task requires its own definition, annotation strategy, and evaluation procedure.

---

## 9. Planned Phase 5: Improved Classification Framework

Phase 5 will investigate approaches to improving generalization and minority-class performance.

The proposed research direction includes:

- Combining lexical TF-IDF features with pretrained semantic embeddings.
- Investigating controlled class-weighting strategies.
- Improving recognition of difficult requirement categories.
- Comparing hybrid models against existing Phase 4 baselines.
- Examining robustness across previously unseen projects.
- Using appropriate independent evaluation before reporting final improvements.

The hybrid approach is a proposed research direction and has not yet been validated as superior to the existing models.

---

## 10. Documentation and Research Materials

The research introduction and literature review are stored in:

`Docs/Thesis_Intro+LiteratureReview.pdf`

Detailed phase documentation is available in:

- `promise_exploration_phase_1/README.md`
- `promise_preprocessing_phase_2/README.md`
- `promise_model_development_phase_3/README.md`
- `promise_advanced_models_phase_4/README.md`

Additional Phase 4 records include:

- `EXPERIMENT_LOG.md`: experiment descriptions and results.
- `FINDINGS_AND_LIMITATIONS.md`: experimental findings and research limitations.
- `PHASE4_FILE_INVENTORY.txt`: file inventory.
- `code_outputs_phase4.txt`: retained execution outputs.

The numbered Python scripts and generated metrics serve as the primary reproducibility records.

---

## 11. Research Status and Conclusion

Phases 1–4 have established a documented workflow for PROMISE+ requirements analysis, preprocessing, model development, and cross-project evaluation.

The experiments demonstrate that traditional TF-IDF-based classifiers remain competitive with pretrained transformer architectures on this dataset.

The principal unresolved challenges are severe class imbalance, limited minority-class representation, and differences between software projects.

The next stage will investigate improved lexical-semantic representations and more robust classification strategies.

---

## 12. License and Dataset Attribution

This repository is intended for academic research.

Dataset ownership and redistribution permissions remain with their respective rights holders. Any public redistribution must follow the original dataset's applicable licensing and attribution requirements.

Pretrained language models and third-party software retain their respective licenses and attribution requirements.

Research results should be interpreted within the evaluation protocols and limitations documented in this repository.
