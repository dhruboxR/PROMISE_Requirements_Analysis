# Phase 5: Improved Requirements Classification Framework

**Hybrid TF-IDF and MiniLM-Based Software Requirements Classification**

This directory contains Phase 5 of the PROMISE+ Requirements Analysis research project. The research investigates methods for improving automated software requirements classification, particularly on previously unseen software projects.

Phase 5 builds on the results of earlier dataset exploration, preprocessing, traditional machine learning, and transformer-based experiments.

The proposed hybrid framework combines word and character TF-IDF features, pretrained MiniLM sentence embeddings, and Logistic Regression with controlled class weighting.

**Current status:** Experiments 01–07 completed. Consolidated model training, model serialization, and inference development are planned.

---

## 1. Research Objectives

The objectives of Phase 5 are to:

1. Verify and reproduce the evaluation baseline established in Phase 4.
2. Develop a hybrid classification approach combining lexical and semantic text representations.
3. Evaluate the contribution of pretrained MiniLM sentence embeddings.
4. Compare unweighted, tempered, and fully balanced classification strategies.
5. Improve macro-F1 while maintaining reasonable overall classification accuracy.
6. Investigate classification failures involving minority requirement categories.
7. Analyze how training-example availability changes across software projects and validation folds.
8. Identify a provisional model configuration for further development.
9. Prepare a reproducible inference pipeline and saved classifier.

The primary evaluation metric is **macro-F1**, because the dataset contains substantial class imbalance.

Accuracy, balanced accuracy, weighted-F1, and per-class metrics are also reported.

---

## 2. Dataset Information

Phase 5 uses the PROMISE+ dataset prepared in the earlier research phases.

| Property | Value |
|---|---|
| Dataset | PROMISE+ |
| Original requirements | 3,677 |
| Software projects | 88 |
| Requirement categories | 12 |
| Development requirements | 3,127 |
| Connected development groups | 66 |
| Cross-validation folds | 3 |
| Random seed | 42 |
| Original test partition | 550 requirements |

The 12 requirement categories are:

`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`

### 2.1 Development Dataset

The original Phase 2 training and validation partitions were combined to produce the 3,127-record development dataset.

The original 550-record test partition was not accessed during Phase 5 Experiments 01–07.

However, that partition was previously evaluated in Phase 3. Therefore, it should not be presented as a completely untouched independent evaluation set for future model selection.

### 2.2 Data Integrity and Leakage Prevention

Phase 5 reuses the validated dataset-loading and grouping procedures from Phase 4.

Connected groups link requirements that:

- Belong to the same software project.
- Have identical normalized requirement text.

All records belonging to a connected group remain together in either the training or held-out portion of a fold.

This approach prevents project overlap and matching normalized-text overlap between training and validation within each fold.

Experiment 01 verified that the Phase 5 folds exactly reproduce the Phase 4 fold assignments.

---

## 3. Hybrid Classification Framework

The proposed classification framework combines lexical and semantic text representations.

```text
                PROMISE+ Requirement Text
                           |
                    Normalized Text
                           |
              +------------+------------+
              |                         |
              v                         v
       Word + Character            Pretrained
           TF-IDF                    MiniLM
              |                         |
              v                         v
       Lexical Features         Semantic Embeddings
                                  (384 dimensions)
              |                         |
              +------------+------------+
                           |
                           v
                 Weighted Feature Fusion
                           |
                           v
                    Logistic Regression
                           |
                           v
                 Predicted Requirement
                         Category
```

### 3.1 Lexical Feature Extraction

The lexical branch uses word and character TF-IDF features.

Word features represent individual words and word sequences, while character features capture smaller text patterns and variations in terminology.

The TF-IDF vectorizers are fitted only on the training records of each fold.

### 3.2 Semantic Feature Extraction

The semantic branch uses the pretrained model:

`sentence-transformers/all-MiniLM-L6-v2`

The encoder generates 384-dimensional embeddings.

| Parameter | Configuration |
|---|---|
| Pretrained model | all-MiniLM-L6-v2 |
| Embedding dimensions | 384 |
| Maximum token length | 128 |
| Fine-tuning | None |
| Encoder status | Frozen |
| Computation device | CUDA GPU when available |

The MiniLM encoder is not fine-tuned using PROMISE+ classification labels during the current experiments.

### 3.3 Feature Fusion

The hybrid model combines the TF-IDF and MiniLM representations using feature concatenation.

The representation is defined as:

\[
X_{\text{hybrid}}=[X_{\text{TF-IDF}},\ \alpha X_{\text{MiniLM}}]
\]

Here, \(\alpha\) controls the contribution of the semantic embedding features.

Experiment 02 evaluates:

- TF-IDF only, with \(\alpha=0\).
- Hybrid with \(\alpha=0.5\).
- Hybrid with \(\alpha=1.0\).

The feature-weight experiments use a fixed classifier configuration to make the comparisons consistent.

### 3.4 Classification

Logistic Regression is used to predict one of the 12 requirement categories.

| Parameter | Value |
|---|---|
| Classifier | Logistic Regression |
| Regularization parameter C | 10.0 |
| Maximum iterations | 2,000 |
| Random seed | 42 |
| Class-weighting methods | None, tempered, balanced |

Tempered weighting is derived from the square root of inverse-frequency balanced weights and normalized using the class frequencies in each training fold.

All class weights are calculated using training-fold labels only.

---

## 4. Evaluation Methodology

Phase 5 uses three-fold group-aware cross-validation.

The evaluation strategy is based on `StratifiedGroupKFold`, using a fixed random seed of 42.

### 4.1 Fold Distribution

| Fold | Training records | Held-out records |
|---|---:|---:|
| Fold 1 | 2,159 | 968 |
| Fold 2 | 2,024 | 1,103 |
| Fold 3 | 2,071 | 1,056 |

All three folds preserve connected-group separation.

### 4.2 Evaluation Metrics

The following metrics are used:

- **Accuracy:** Proportion of correctly predicted requirements.
- **Macro-F1:** Unweighted average of the F1 scores across all 12 categories.
- **Balanced accuracy:** Average class recall.
- **Weighted-F1:** Class-frequency-weighted average F1 score.
- **Per-class precision, recall, and F1:** Used to examine individual categories.

### 4.3 Mean Cross-Validation Versus OOF Metrics

Two types of aggregate results are reported.

**Mean cross-validation metrics** are calculated by averaging the three fold-level scores.

**Pooled out-of-fold (OOF) metrics** are calculated by combining all held-out predictions and evaluating them together.

These results may differ because the folds have different sizes and class distributions.

Mean cross-validation macro-F1 is the primary model-selection metric used in the current development experiments.

### 4.4 Development Evaluation Limitations

The same folds have been examined repeatedly during Phases 4 and 5.

As a result, the reported cross-validation scores are useful for development comparisons but are not fully independent performance estimates.

A new, independent evaluation procedure is required before making final claims about model generalization.

---

## 5. Phase 5 Experiments

Phase 5 currently contains seven completed experiments.

| Experiment | Title | Objective |
|---|---|---|
| 01 | Baseline and Fold Verification | Reproduce Phase 4 baseline results and validate fold assignments |
| 02 | Hybrid TF-IDF + MiniLM | Compare lexical and hybrid representations |
| 03 | Hybrid Error Analysis | Analyze class-level performance and prediction changes |
| 04 | Difficult-Class Investigation | Examine O, SC, and PO classification failures |
| 05 | Minority Training Coverage | Analyze minority-class training coverage across folds |
| 06 | Hybrid Class-Weight Comparison | Compare unweighted, tempered, and balanced Logistic Regression |
| 07 | Class-Weight Error Analysis | Compare per-class performance and prediction differences |

### Experiment 01: Baseline and Fold Verification

The experiment confirmed:

- 3,127 development records.
- 12 requirement classes.
- 66 connected groups.
- Exact agreement with Phase 4 cross-validation folds.
- No original test-partition access.

The reproduced Phase 4 baseline was word and character TF-IDF with Logistic Regression.

| Metric | Mean CV | Standard deviation |
|---|---:|---:|
| Accuracy | 0.6254 | 0.0364 |
| Macro-F1 | 0.2816 | 0.0424 |
| Balanced accuracy | 0.2827 | 0.0228 |
| Weighted-F1 | 0.5878 | 0.0590 |

### Experiment 02: Hybrid Feature Classification

This experiment combined TF-IDF with pretrained MiniLM embeddings using two semantic-feature weights.

The classifier used fully balanced class weighting.

| Configuration | Mean accuracy | Mean macro-F1 | Mean balanced accuracy |
|---|---:|---:|---:|
| TF-IDF baseline | 0.6254 | 0.2816 | 0.2827 |
| Hybrid 0.5 | **0.6386** | 0.3070 | 0.3190 |
| Hybrid 1.0 | 0.6296 | **0.3141** | **0.3336** |

Hybrid 1.0 improved mean macro-F1 compared with the TF-IDF baseline.

The macro-F1 improvement was observed in all three folds.

### Experiment 03: Hybrid Error Analysis

This experiment evaluated pooled out-of-fold predictions.

| Configuration | OOF accuracy | OOF macro-F1 | OOF balanced accuracy |
|---|---:|---:|---:|
| TF-IDF baseline | 0.6239 | 0.2768 | 0.2637 |
| Hybrid 0.5 | **0.6370** | 0.3051 | 0.2985 |
| Hybrid 1.0 | 0.6278 | **0.3148** | **0.3127** |

When comparing Hybrid 1.0 with the TF-IDF baseline:

| Prediction outcome | Requirements |
|---|---:|
| Both correct | 1,792 |
| Corrected by hybrid | 171 |
| Damaged by hybrid | 159 |
| Both incorrect | 1,005 |

The hybrid produced a net gain of 12 correct predictions.

Per-class analysis showed improvements for 10 of the 12 classes.

The largest F1 gains were observed for A, PE, PO, and L.

The O and SC categories experienced slight F1 decreases.

### Experiment 04: Difficult-Class Investigation

This experiment investigated the O, SC, and PO categories.

| Class | Records | Projects | Baseline correct | Hybrid correct | Hybrid recall |
|---|---:|---:|---:|---:|---:|
| O | 136 | 23 | 29 | 30 | 22.06% |
| SC | 86 | 9 | 2 | 2 | 2.33% |
| PO | 66 | 7 | 0 | 3 | 4.55% |

Important findings:

- O requirements were frequently predicted as F.
- SC requirements were primarily confused with F and several other categories.
- Project 80 contributed 54 SC requirements, and the hybrid correctly classified none of them.
- PO requirements were concentrated in Projects 59 and 52.
- The hybrid improved PO recall slightly but did not resolve the classification problem.

The findings indicate that cross-project differences may contribute to minority-class failures.

### Experiment 05: Minority-Class Training Coverage

This experiment measured training-example availability for O, SC, and PO.

| Class | Fold 1 training | Fold 2 training | Fold 3 training |
|---|---:|---:|---:|
| O | 90 | 92 | 90 |
| SC | 74 | 74 | 24 |
| PO | 41 | 31 | 60 |

The project diversity of these classes also varied between folds.

**SC:** Fold 3 contained only 24 training examples, while 62 SC examples were held out.

**PO:** In Fold 1, Project 59 contributed 34 of 41 training examples. In Fold 2, Project 52 contributed 24 of 31 training examples.

This concentration suggests that class-level sample counts alone may not adequately represent how much project diversity is available for model training.

### Experiment 06: Controlled Class-Weight Comparison

This experiment evaluated Hybrid 1.0 using three weighting strategies.

| Configuration | Mean accuracy | Mean macro-F1 | Mean balanced accuracy | Mean weighted-F1 |
|---|---:|---:|---:|---:|
| Unweighted | **0.6648** | 0.2983 | 0.2870 | 0.6056 |
| Tempered | 0.6551 | **0.3200** | 0.3198 | **0.6116** |
| Balanced | 0.6296 | 0.3141 | **0.3336** | 0.6059 |

The tempered configuration achieved the highest mean CV macro-F1.

The balanced configuration achieved the highest mean balanced accuracy.

The unweighted configuration achieved the highest mean accuracy.

The experiment also verified that the balanced configuration reproduced Experiment 02 results.

### Experiment 07: Class-Weight Error Analysis

This experiment analyzed pooled OOF predictions for the three weighting configurations.

| Configuration | OOF accuracy | OOF macro-F1 | OOF balanced accuracy |
|---|---:|---:|---:|
| Unweighted | **0.6633** | 0.2874 | 0.2569 |
| Tempered | 0.6533 | 0.3114 | 0.2900 |
| Balanced | 0.6278 | **0.3148** | **0.3127** |

The tempered model achieved higher pooled F1 scores for F, MN, O, SC, SE, US, and FT.

The balanced model achieved higher pooled F1 scores for A, L, LF, PE, and PO.

The prediction-level comparison showed:

| Outcome | Requirements |
|---|---:|
| Both correct | 1,922 |
| Correct only by tempered | 121 |
| Correct only by balanced | 41 |
| Both incorrect | 1,043 |

Tempered weighting produced 80 more correct predictions overall.

However, balanced weighting produced slightly higher pooled macro-F1 because macro-F1 gives equal weight to every category.

---

## 6. Best Development Configuration

Based on mean cross-validation macro-F1, the provisional primary configuration is:

**TF-IDF + MiniLM + Tempered Logistic Regression**

| Component | Selected configuration |
|---|---|
| Lexical representation | Word and character TF-IDF |
| Semantic representation | Frozen MiniLM |
| Semantic embedding size | 384 |
| Semantic feature weight | 1.0 |
| Classification algorithm | Logistic Regression |
| Regularization C | 10.0 |
| Maximum iterations | 2,000 |
| Class weighting | Tempered |
| Random seed | 42 |
| Evaluation | Three-fold group-aware cross-validation |

### Comparison with Phase 4

| Metric | Phase 4 TF-IDF baseline | Phase 5 tempered hybrid |
|---|---:|---:|
| Mean accuracy | 62.54% | **65.51%** |
| Mean macro-F1 | 28.16% | **32.00%** |
| Mean balanced accuracy | 28.27% | **31.98%** |
| Mean weighted-F1 | 58.78% | **61.16%** |

The hybrid improved mean macro-F1 by **3.84 percentage points**, approximately 13.6% relative to the Phase 4 baseline.

This improvement was observed during development cross-validation and has not been independently confirmed.

---

## 7. Repository Structure

The current Phase 5 directory contains seven experiment scripts, saved evaluation results, and research documentation.

```text
promise_improved_model_phase_5/
├── code_outputs_phase5.txt
├── EXPERIMENT_LOG.md
├── README.md
│
├── scripts/
│   ├── 01_verify_phase5_baseline.py
│   ├── 02_hybrid_tfidf_minilm_cv.py
│   ├── 03_hybrid_error_analysis.py
│   ├── 04_difficult_class_analysis.py
│   ├── 05_minority_training_coverage.py
│   ├── 06_hybrid_class_weight_cv.py
│   └── 07_class_weight_error_analysis.py
│
└── outputs/
    ├── metrics/
    │   ├── 02_hybrid_cv_summary.csv
    │   ├── 02_hybrid_fold_metrics.csv
    │   ├── 06_class_weight_fold_metrics.csv
    │   └── 06_class_weight_summary.csv
    │
    ├── reports/
    │   ├── 01_baseline_manifest.json
    │   ├── 02_hybrid_0_5_class_report.json
    │   ├── 02_hybrid_1_0_class_report.json
    │   ├── 02_hybrid_experiment_metadata.json
    │   ├── 02_tfidf_baseline_class_report.json
    │   └── 06_class_weight_manifest.json
    │
    └── tables/
        ├── 01_fixed_fold_assignments.csv
        ├── 02_hybrid_oof_predictions.csv
        ├── 03_per_class_comparison.csv
        ├── 03_prediction_comparison.csv
        ├── 04_difficult_class_errors.csv
        ├── 04_difficult_class_project_analysis.csv
        ├── 04_hybrid_confusion_matrix.csv
        ├── 05_minority_training_coverage.csv
        ├── 06_class_weight_oof_predictions.csv
        └── 07_tempered_vs_balanced_predictions.csv
```

### Directory and File Descriptions

| Directory or file | Purpose |
|---|---|
| `scripts/` | Python implementations of Experiments 01–07 |
| `outputs/metrics/` | Per-fold model metrics and comparison summaries |
| `outputs/reports/` | Experiment metadata and class-level reports |
| `outputs/tables/` | Fold assignments, predictions, confusion matrices, and error analyses |
| `code_outputs_phase5.txt` | Recorded terminal outputs from the completed experiments |
| `EXPERIMENT_LOG.md` | Experiment documentation and research notes |
| `README.md` | Phase 5 overview, methodology, results, and reproduction instructions |

Saved final-model artifacts and inference scripts are not yet included in the directory.

---

## 8. Technologies and Dependencies

Phase 5 uses Python and the following libraries:

| Library | Purpose |
|---|---|
| pandas | Dataset and prediction-table processing |
| NumPy | Numerical operations |
| SciPy | Sparse feature matrices and concatenation |
| scikit-learn | TF-IDF, Logistic Regression, cross-validation, and evaluation |
| PyTorch | GPU-accelerated embedding generation |
| Hugging Face Transformers | Pretrained MiniLM tokenizer and encoder |

The pretrained semantic model is:

`sentence-transformers/all-MiniLM-L6-v2`

### Environment

The experiments were conducted using the existing `.venv-transformers` environment in the main research workspace.

From the thesis root:

```bash
source .venv-transformers/bin/activate
cd promise_improved_model_phase_5
```

The current directory does not contain a separate dependency file. The Phase 4 directory contains the recorded transformer environment dependencies.

For reproducibility, the active environment can also be recorded within Phase 5 using:

```bash
python -m pip freeze > requirements-transformers.lock.txt
```

This command creates an additional file. It should be added to the repository structure documentation if generated and tracked.

---

## 9. Reproducing the Experiments

The Phase 5 scripts should be executed from the Phase 5 directory, with the transformer environment activated.

```bash
python scripts/01_verify_phase5_baseline.py
python scripts/02_hybrid_tfidf_minilm_cv.py
python scripts/03_hybrid_error_analysis.py
python scripts/04_difficult_class_analysis.py
python scripts/05_minority_training_coverage.py
python scripts/06_hybrid_class_weight_cv.py
python scripts/07_class_weight_error_analysis.py
```

### Dependencies on Earlier Phases

Phase 5 reuses existing scripts and prepared dataset outputs from earlier research phases.

In particular, it uses Phase 4 implementations for:

- Validated development dataset loading.
- Connected-group construction.
- Word and character TF-IDF feature generation.
- MiniLM sentence embedding generation.
- Baseline comparison and fold verification.

The required Phase 2 and Phase 4 directories must remain accessible in the original repository structure.

### Key Saved Outputs

**Experiment 01**

- `outputs/tables/01_fixed_fold_assignments.csv`
- `outputs/reports/01_baseline_manifest.json`

**Experiment 02**

- `outputs/metrics/02_hybrid_cv_summary.csv`
- `outputs/metrics/02_hybrid_fold_metrics.csv`
- `outputs/tables/02_hybrid_oof_predictions.csv`

**Experiment 03**

- `outputs/tables/03_per_class_comparison.csv`
- `outputs/tables/03_prediction_comparison.csv`

**Experiment 04**

- `outputs/tables/04_difficult_class_errors.csv`
- `outputs/tables/04_difficult_class_project_analysis.csv`
- `outputs/tables/04_hybrid_confusion_matrix.csv`

**Experiment 05**

- `outputs/tables/05_minority_training_coverage.csv`

**Experiment 06**

- `outputs/metrics/06_class_weight_fold_metrics.csv`
- `outputs/metrics/06_class_weight_summary.csv`
- `outputs/tables/06_class_weight_oof_predictions.csv`
- `outputs/reports/06_class_weight_manifest.json`

**Experiment 07**

- `outputs/tables/07_class_weight_per_class_comparison.csv`
- `outputs/tables/07_tempered_vs_balanced_predictions.csv`

---

## 10. Main Research Findings

### 10.1 Hybrid Representations Improve Development Performance

Combining lexical TF-IDF features and frozen MiniLM embeddings improved mean cross-validation macro-F1 compared with the Phase 4 TF-IDF baseline.

The results suggest that lexical and semantic representations provide complementary information for certain requirement categories.

### 10.2 Moderate Class Weighting Provides a Useful Trade-Off

Tempered weighting achieved the highest mean cross-validation macro-F1 among the tested weighting configurations.

Balanced weighting produced stronger mean balanced accuracy, while unweighted classification produced higher overall accuracy.

### 10.3 Minority-Class Performance Remains Limited

Classes such as SC and PO remain difficult despite the improvements.

Some classes are concentrated in a small number of software projects, reducing the diversity of training examples available within each fold.

### 10.4 Cross-Project Generalization Remains a Challenge

The project-level analysis identified substantial differences in classification performance between held-out projects.

This indicates that a model trained on one set of projects may struggle when requirements use terminology or writing patterns that differ substantially from the training data.

### 10.5 Class-Level Evaluation Is Essential

The hybrid model improved the primary evaluation metric but did not improve every requirement category.

Per-class precision, recall, F1, and project-level results are therefore necessary to interpret classification performance.

---

## 11. Research Limitations

### Dataset Imbalance

The PROMISE+ dataset contains uneven class distributions, with Functional requirements representing approximately 61.41% of all records.

### Limited Minority-Class Diversity

Several minority categories appear in relatively few software projects.

This restricts the available training examples when entire projects are held out.

### Repeated Cross-Validation

The same three group-aware folds have been examined repeatedly.

This allows controlled comparisons but may bias model-selection decisions toward the observed development folds.

### Original Test Partition

The original Phase 3 test partition was previously evaluated.

Although it was excluded from Phase 5 model development, it is not an untouched independent test set.

### Remaining Classification Errors

The hybrid framework still performs poorly on SC, PO, and several other categories.

The current results do not establish whether the remaining errors are caused primarily by annotation quality, limited training coverage, or insufficient model representation.

### Final Model Not Yet Trained

The Phase 5 results currently come from cross-validation experiments.

A consolidated classifier trained on all available development records, saved model artifacts, and a standalone inference pipeline have not yet been completed.

---

## 12. Planned Development

The next phase of work will focus on consolidating the selected hybrid model.

### Experiment 08: Train and Save the Hybrid Model

The selected configuration will be trained on all 3,127 development requirements.

The work will include:

- Fitting the lexical TF-IDF representation.
- Generating frozen MiniLM embeddings.
- Constructing the hybrid feature matrix.
- Computing tempered class weights from the development labels.
- Training Logistic Regression.
- Saving the trained classifier and feature extractor.
- Saving configuration metadata and label information.

### Experiment 09: Build the Inference Pipeline

A standalone prediction workflow will be developed to:

- Load the saved model.
- Normalize input requirement text consistently.
- Generate lexical and semantic features.
- Predict one of the 12 requirement categories.
- Provide optional class probabilities.
- Verify predictions using reproducible test inputs.

### Experiment 10: Reproducibility and Robustness

The selected model and inference workflow will be reviewed for:

- Artifact integrity.
- Prediction consistency.
- Compatibility with the recorded Python environment.
- Robustness to input variations.
- Documented classification limitations.

Additional classification methods may be explored later under a separately defined experimental protocol.

---

## 13. Conclusion

Phase 5 has developed and evaluated a hybrid lexical-semantic classification approach for the PROMISE+ dataset.

The provisional best model combines word and character TF-IDF, frozen MiniLM sentence embeddings, and tempered Logistic Regression.

This configuration achieved:

- Mean cross-validation accuracy: **65.51%**
- Mean cross-validation macro-F1: **32.00%**
- Mean cross-validation balanced accuracy: **31.98%**
- Mean cross-validation weighted-F1: **61.16%**

Compared with the Phase 4 TF-IDF baseline, mean macro-F1 increased by **3.84 percentage points**.

The results support further investigation of hybrid representations, although SC and PO remain difficult and the improvement has not yet been independently validated.

The next work will focus on model consolidation, serialized model artifacts, reproducible inference, and stronger evaluation procedures.

---

**Research note:** All reported Phase 5 performance results are development-stage findings. Final generalization claims require an appropriately independent evaluation.