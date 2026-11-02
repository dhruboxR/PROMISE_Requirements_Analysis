# PROMISE+ Requirements Classification | Phase 3: Machine Learning Model Development

## Overview

This directory contains **Phase 3: Machine Learning Model Development and Evaluation** of the PROMISE+ requirements classification research project.

The objective of this phase was to develop, optimize, and evaluate machine learning models for classifying software requirements into **12 categories** using natural language processing (NLP).

The experiments use the validated and preprocessed PROMISE+ dataset produced in Phases 1 and 2.

This phase includes:

- Classical machine learning baseline development
- TF-IDF feature extraction
- Hyperparameter optimization
- Per-class classification analysis
- Model comparison and error analysis
- Project-level generalization analysis
- Final independent test evaluation

**Phase 3 Status: Completed**

The final selected model was **TF-IDF + Tuned Logistic Regression**, achieving **68.00% test accuracy** and **37.55% macro-F1**.

---

## 1. Research Objectives

The main objectives were:

1. Develop reproducible machine learning baselines for 12-class requirements classification.
2. Compare different classifiers using consistent dataset partitions and evaluation metrics.
3. Investigate the influence of class imbalance on classification performance.
4. Optimize promising models using hyperparameter tuning.
5. Identify difficult requirement categories and common classification errors.
6. Examine classification performance across unseen software projects.
7. Evaluate the selected model using an independent test partition.

---

## 2. Dataset Information

**Dataset:** PROMISE+ Requirements Classification Dataset

| Property | Value |
|---|---|
| Total requirements | 3,677 |
| Total software projects | 88 |
| Classification categories | 12 |
| Training records | 2,574 |
| Validation records | 553 |
| Test records | 550 |
| Split ratio | Approximately 70:15:15 |
| Random seed | 42 |
| Selected split | Candidate 2 |

### Requirement Categories

The dataset contains the following labels:

`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`

The original requirement labels and project identifiers were preserved throughout the experiments.

### Dataset Splitting

The datasets were obtained from Phase 2:

```text
promise_preprocessing_phase_2/
└── outputs/
    └── splits/
        ├── final_train.csv
        ├── final_validation.csv
        └── final_test.csv
```

The split methodology prevents:

- Project overlap between training, validation, and test partitions
- Normalized duplicate-text overlap between partitions
- Duplicate record identifiers across partitions

Dataset integrity was verified using SHA-256 checksums.

All 3,677 original requirements were retained.

Four annotation-conflict groups identified during preprocessing remain unresolved and were not automatically relabeled.

---

## 3. Project Structure

```text
promise_model_development_phase_3/
│
├── scripts/
│   ├── common.py
│   ├── 01_baseline_logistic_regression.py
│   ├── 02_train_svm.py
│   ├── 03_train_naive_bayes.py
│   ├── 04_train_random_forest.py
│   ├── 05_compare_models.py
│   ├── 06_tune_linear_svm.py
│   ├── 07_tune_logistic_regression.py
│   ├── 08_analyze_tuned_models.py
│   ├── 09_review_classification_errors.py
│   ├── 10_project_error_analysis.py
│   └── 11_final_test_evaluation.py
│
├── outputs/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   ├── reports/
│   └── tables/
│
├── README.md
└── requirements.txt
```

### Script Descriptions

| Script | Description |
|---|---|
| `common.py` | Shared dataset loading, checksum verification, paths, and evaluation functions |
| `01_baseline_logistic_regression.py` | Majority-class and Logistic Regression baselines |
| `02_train_svm.py` | TF-IDF + Linear SVM training |
| `03_train_naive_bayes.py` | TF-IDF + Multinomial Naive Bayes |
| `04_train_random_forest.py` | TF-IDF + Random Forest |
| `05_compare_models.py` | Baseline comparison, figures, and per-class reports |
| `06_tune_linear_svm.py` | Linear SVM hyperparameter tuning |
| `07_tune_logistic_regression.py` | Logistic Regression hyperparameter tuning |
| `08_analyze_tuned_models.py` | Tuned-model disagreement and classification-error analysis |
| `09_review_classification_errors.py` | Investigation of difficult requirement classes |
| `10_project_error_analysis.py` | Project-level validation performance analysis |
| `11_final_test_evaluation.py` | Frozen-model independent test evaluation |

---

## 4. Experimental Environment

The experiments were conducted using Python 3.14.4 in a dedicated virtual environment.

### Main Libraries

| Library | Version |
|---|---|
| Python | 3.14.4 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |
| pandas | 2.3.3 |
| NumPy | 2.5.4 |
| Matplotlib | 3.11.2 |

### Environment Setup

Navigate to the thesis root directory:

```bash
cd ~/Unofficial\ Thesis
```

Activate the shared virtual environment:

```bash
source .venv/bin/activate
```

Navigate to Phase 3:

```bash
cd promise_model_development_phase_3
```

Install dependencies if required:

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python -m pip check
```

The scripts assume the Phase 2 output directory is available in the same thesis workspace.

---

## 5. Feature Extraction

All trainable baseline classifiers used **Term Frequency-Inverse Document Frequency (TF-IDF)**.

The initial TF-IDF configuration was:

| Parameter | Value |
|---|---|
| Lowercase | True |
| Strip accents | Unicode |
| N-gram range | (1, 2) |
| Minimum document frequency | 2 |
| Maximum document frequency | 0.95 |
| Sublinear TF | True |

TF-IDF vocabulary and feature weights were fitted exclusively on the training partition.

The trained vectorizer and classifier were saved together using scikit-learn pipelines.

---

## 6. Baseline Machine Learning Experiments

Five baseline configurations were evaluated:

1. Majority-class classifier
2. TF-IDF + Logistic Regression
3. TF-IDF + Linear Support Vector Machine
4. TF-IDF + Multinomial Naive Bayes
5. TF-IDF + Random Forest

### Baseline Validation Results

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Majority Class | 0.6148 | 0.0833 | 0.0635 | 0.4682 |
| Logistic Regression | 0.5642 | 0.4121 | 0.3435 | 0.5781 |
| **Linear SVM** | **0.6076** | **0.4169** | **0.3570** | **0.6078** |
| Multinomial Naive Bayes | 0.6166 | 0.0867 | 0.0699 | 0.4722 |
| Random Forest | 0.5208 | 0.3061 | 0.2640 | 0.5185 |

**Linear SVM achieved the highest baseline macro-F1 of 0.3570.**

Although Naive Bayes achieved slightly higher accuracy, it predicted the majority class `F` for 552 of 553 validation requirements, demonstrating poor minority-class recognition.

---

## 7. Hyperparameter Optimization

Hyperparameter tuning was performed on the two strongest initial classifiers:

- Linear SVM
- Logistic Regression

Each classifier was evaluated across **24 configurations**, resulting in 48 tuning experiments.

### Search Space

| Hyperparameter | Tested values |
|---|---|
| Regularization strength (`C`) | 0.1, 1.0, 10.0 |
| Class weight | None, balanced |
| N-gram range | (1, 1), (1, 2) |
| Minimum document frequency | 1, 2 |

The primary model-selection metric was **validation macro-F1**.

### Tuned Linear SVM

Best configuration:

```python
LinearSVC(
    C=1.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)
```

TF-IDF configuration:

```python
TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)
```

### Tuned Logistic Regression

Best configuration:

```python
LogisticRegression(
    C=10.0,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)
```

The TF-IDF configuration used word unigrams and bigrams with `min_df=1`.

### Tuned Model Validation Results

| Model | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Tuned Linear SVM | 0.6564 | 0.4129 | 0.3766 | 0.6423 |
| **Tuned Logistic Regression** | **0.6691** | **0.4131** | **0.3894** | **0.6505** |

Tuned Logistic Regression achieved the highest validation macro-F1 and was selected for final independent testing.

---

## 8. Error Analysis

Detailed analysis was performed using classification reports, confusion matrices, and prediction-level comparisons.

### Difficult Validation Classes

The tuned Logistic Regression classifier struggled with several minority classes.

| Class | Validation records | Recall |
|---|---:|---:|
| PO | 24 | 0.0000 |
| FT | 10 | 0.1000 |
| L | 9 | 0.2222 |
| SC | 16 | 0.1250 |
| MN | 9 | 0.1111 |
| LF | 12 | 0.2500 |

### PO Classification Problem

All 24 PO validation records originated from Project 52.

The model failed to correctly classify any of these records.

The incorrect predictions were distributed as follows:

| Predicted class | Count |
|---|---:|
| F | 12 |
| PE | 4 |
| SE | 3 |
| O | 2 |
| L | 2 |
| US | 1 |

This result suggests that project-specific terminology and differences in class representation may contribute to classification difficulty.

The training data contained 42 PO records, of which 34 originated from Project 59.

### Tuned Model Agreement

| Outcome | Records |
|---|---:|
| Both models correct | 353 |
| Only SVM correct | 10 |
| Only Logistic Regression correct | 17 |
| Both models incorrect | 173 |

The models disagreed on 43 of 553 validation requirements.

---

## 9. Project-Level Generalization

Since the dataset partitions are project-disjoint, evaluation measures classification performance on projects excluded from training.

Project-level validation performance varied considerably.

Selected examples:

| Project ID | Requirements | Accuracy |
|---|---:|---:|
| 52 | 26 | 0.0000 |
| 60 | 36 | 0.3889 |
| 6 | 74 | 0.5405 |
| 61 | 50 | 0.8000 |
| 77 | 95 | 0.9368 |

The results indicate substantial variation in model performance across unseen projects.

Small projects should be interpreted cautiously because one incorrect prediction can significantly affect project-level accuracy.

---

## 10. Final Independent Test Evaluation

After hyperparameter tuning and model selection, the selected Logistic Regression pipeline was evaluated on the frozen test partition.

The test dataset had not been used for model training or hyperparameter selection.

### Final Selected Model

**TF-IDF + Tuned Logistic Regression**

| Parameter | Value |
|---|---|
| `C` | 10.0 |
| `class_weight` | balanced |
| `ngram_range` | (1, 2) |
| `min_df` | 1 |
| Random seed | 42 |

### Final Test Results

| Metric | Score |
|---|---:|
| **Accuracy** | **0.6800** |
| Balanced Accuracy | 0.3465 |
| Macro Precision | 0.5291 |
| Macro Recall | 0.3465 |
| **Macro-F1** | **0.3755** |
| Weighted-F1 | 0.6378 |

The model correctly classified **374 of 550 test requirements**.

### Validation vs Test Performance

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 0.6691 | 0.6800 |
| Balanced Accuracy | 0.4131 | 0.3465 |
| Macro-F1 | 0.3894 | 0.3755 |
| Weighted-F1 | 0.6505 | 0.6378 |

Although test accuracy increased slightly, balanced accuracy decreased, indicating weaker average recall across categories.

### Per-Class Test Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| F | 0.7644 | 0.9104 | 0.8311 | 335 |
| A | 0.7273 | 0.7273 | 0.7273 | 11 |
| L | 0.0000 | 0.0000 | 0.0000 | 35 |
| LF | 0.6667 | 0.2000 | 0.3077 | 10 |
| MN | 0.4000 | 0.1818 | 0.2500 | 11 |
| O | 0.2800 | 0.3333 | 0.3043 | 21 |
| PE | 0.6111 | 0.4400 | 0.5116 | 25 |
| SC | 0.0000 | 0.0000 | 0.0000 | 9 |
| SE | 0.5000 | 0.6857 | 0.5783 | 35 |
| US | 0.4000 | 0.2791 | 0.3288 | 43 |
| FT | 1.0000 | 0.2000 | 0.3333 | 5 |
| PO | 1.0000 | 0.2000 | 0.3333 | 10 |

The strongest test performance was observed for `F` and `A`.

The model failed to correctly identify any test examples belonging to `L` or `SC`.

---

## 11. Evaluation Metrics

The experiments used the following metrics:

| Metric | Description |
|---|---|
| Accuracy | Proportion of correctly classified requirements |
| Balanced Accuracy | Average recall across all classes |
| Macro Precision | Average class precision with equal class weighting |
| Macro Recall | Average recall across all classes |
| Macro-F1 | Average class F1-score with equal class weighting |
| Weighted-F1 | Average F1-score weighted by class support |

**Macro-F1 was the primary model-selection metric** because the dataset contains substantial class imbalance.

All classification reports used zero-division handling for categories without positive predictions.

---

## 12. Reproducing the Experiments

### Activate the Environment

From the thesis root:

```bash
source .venv/bin/activate
cd promise_model_development_phase_3
```

### Run Baseline Models

```bash
python scripts/01_baseline_logistic_regression.py
python scripts/02_train_svm.py
python scripts/03_train_naive_bayes.py
python scripts/04_train_random_forest.py
python scripts/05_compare_models.py
```

### Run Hyperparameter Tuning

```bash
python scripts/06_tune_linear_svm.py
python scripts/07_tune_logistic_regression.py
```

### Run Error Analysis

```bash
python scripts/08_analyze_tuned_models.py
python scripts/09_review_classification_errors.py
python scripts/10_project_error_analysis.py
```

### Final Test Evaluation

The final test evaluation has already been completed and its results are preserved.

Script:

```bash
python scripts/11_final_test_evaluation.py
```

**Important:** The final test script refuses to overwrite an existing final evaluation report. It is documented here for reproducibility, not as an instruction to rerun the test or use test results for further model selection.

The final recorded result is available in:

```text
outputs/reports/11_final_test_evaluation.json
```

---

## 13. Important Output Files

### Trained Models

```text
outputs/models/
├── 01_tfidf_logistic_regression.joblib
├── 02_tfidf_linear_svm.joblib
├── 03_tfidf_naive_bayes.joblib
├── 04_tfidf_random_forest.joblib
├── 06_tuned_linear_svm.joblib
└── 07_tuned_logistic_regression.joblib
```

### Metrics

```text
outputs/metrics/
├── 01_baseline_metrics.csv
├── 02_svm_metrics.csv
├── 03_naive_bayes_metrics.csv
├── 04_random_forest_metrics.csv
├── 06_svm_tuning_results.csv
├── 07_logistic_regression_tuning_results.csv
├── 07_tuned_models_comparison.csv
└── 11_final_test_metrics.csv
```

### Figures

```text
outputs/figures/
├── 05_baseline_macro_f1.png
└── 05_svm_per_class_f1.png
```

### Final Evaluation Records

```text
outputs/reports/11_final_test_evaluation.json

outputs/tables/11_test_per_class_metrics.csv
outputs/tables/11_test_confusion_matrix.csv
outputs/tables/11_test_predictions.csv
outputs/tables/11_test_project_performance.csv
```

These files support reproducibility and further research analysis.

---

## 14. Research Limitations

Several limitations were identified during Phase 3.

### Class Imbalance

The dataset is dominated by functional requirements (`F`), while several non-functional categories contain relatively few examples.

This significantly affects minority-class recall and macro-F1.

### Cross-Project Generalization

The project-disjoint evaluation demonstrates considerable variation in classification accuracy across unseen projects.

Models may learn project-specific vocabulary that does not transfer effectively to other software domains.

### Minority-Class Recognition

The final model achieves zero recall for `L` and `SC`, despite improved overall accuracy after tuning.

### Dataset Size

The dataset contains only 3,677 records across 12 categories, limiting the diversity of examples available for training.

### Annotation Quality

Four conflicting annotation groups remain unresolved.

These conflicts were preserved for manual review instead of being automatically corrected.

### Model Selection

A total of 48 hyperparameter configurations were evaluated using the validation partition.

Repeated validation-based selection can introduce optimistic performance estimates.

The final test partition was evaluated after model selection.

### Single Evaluation Split

The reported results are based on one frozen project-disjoint split.

Additional group-aware validation and external datasets would provide stronger evidence about generalization.

---

## 15. Phase 3 Conclusion

Phase 3 established a reproducible classical machine learning baseline for multiclass software requirements classification.

Among the evaluated models, tuned Logistic Regression achieved the strongest validation macro-F1.

Its final independent test results were:

- **Accuracy: 68.00%**
- **Macro-F1: 37.55%**
- **Balanced Accuracy: 34.65%**
- **Weighted-F1: 63.78%**

The evaluation demonstrated that conventional TF-IDF-based models can learn useful patterns in requirements text but still struggle with minority categories and cross-project generalization.

These findings provide the baseline for subsequent advanced model development.

---

## 16. Next Phase

### Phase 4: Advanced Requirements Classification and Performance Improvement

Future experiments will investigate:

- Group-aware cross-validation