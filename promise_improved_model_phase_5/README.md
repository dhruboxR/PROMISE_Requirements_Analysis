# Phase 5: Improved Requirements Classification Framework

**Hybrid TF-IDF and MiniLM-Based Software Requirements Classification**

This phase develops and evaluates a hybrid machine learning framework for classifying PROMISE+ software requirements. The framework combines lexical TF-IDF features, pretrained MiniLM sentence embeddings, and Logistic Regression with controlled class weighting.

**Status: Completed (Experiments 01–10)**

## 1. Research Objectives

- Improve classification performance compared with the Phase 4 baseline.
- Combine lexical and semantic text representations.
- Investigate class-weighting strategies for imbalanced requirements.
- Analyze minority-class errors and cross-project generalization.
- Train and save a reproducible hybrid classifier.
- Develop and verify an inference pipeline for new requirements.

## 2. Dataset and Evaluation

| Property | Value |
|---|---|
| Dataset | PROMISE+ |
| Total requirements | 3,677 |
| Software projects | 88 |
| Requirement categories | 12 |
| Development requirements | 3,127 |
| Connected groups | 66 |
| Cross-validation | 3-fold StratifiedGroupKFold |
| Random seed | 42 |

The development dataset combines the original training and validation partitions.

Connected grouping prevents project and identical normalized-text overlap between training and validation folds.

The original 550-record test partition was excluded from all Phase 5 experiments.

## 3. Hybrid Model Architecture

```text
             Software Requirement
                      |
              Text Normalization
                      |
             +--------+--------+
             |                 |
             v                 v
      Word + Character       MiniLM
          TF-IDF          (384 dimensions)
             |                 |
             +--------+--------+
                      |
                Feature Fusion
                      |
                      v
            Logistic Regression
             (Tempered Weights)
                      |
                      v
              Predicted Class
```

### Selected Model Configuration

| Component | Configuration |
|---|---|
| Lexical features | Word + character TF-IDF |
| Semantic encoder | all-MiniLM-L6-v2 (frozen) |
| Semantic feature weight | 1.0 |
| Classifier | Logistic Regression |
| Regularization C | 10.0 |
| Class weighting | Tempered |
| Maximum iterations | 2,000 |
| Total fitted features | 48,993 |
| Requirement categories | 12 |

The final classifier was trained using all 3,127 development requirements.

## 4. Completed Experiments

| ID | Experiment | Outcome |
|---|---|---|
| 01 | Baseline and fold verification | Verified Phase 4 baseline and fold assignments |
| 02 | Hybrid TF-IDF + MiniLM | Improved macro-F1 over TF-IDF |
| 03 | Hybrid error analysis | Identified class-level improvements and errors |
| 04 | Difficult-class investigation | Analyzed O, SC, and PO failures |
| 05 | Minority training coverage | Identified limited training diversity |
| 06 | Hybrid class-weight comparison | Tempered weighting achieved the highest mean macro-F1 |
| 07 | Class-weight error analysis | Compared per-class and pooled OOF performance |
| 08 | Consolidated model training | Trained and saved the selected hybrid model |
| 09 | Inference pipeline | Implemented raw-text classification with Phase 2 normalization |
| 10 | Model verification | All technical verification checks passed |

Detailed experiment records are available in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## 5. Model Performance

The following results are means across three project-aware cross-validation folds.

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Phase 4 TF-IDF baseline | 62.54% | 28.16% | 28.27% |
| Hybrid 0.5, balanced | 63.86% | 30.70% | 31.90% |
| Hybrid 1.0, balanced | 62.96% | 31.41% | **33.36%** |
| Hybrid 1.0, unweighted | **66.48%** | 29.83% | 28.70% |
| **Hybrid 1.0, tempered** | 65.51% | **32.00%** | 31.98% |

The selected tempered hybrid improved mean macro-F1 by **3.84 percentage points** over the Phase 4 baseline.

These are exploratory cross-validation results. The consolidated classifier was not independently evaluated after training.

## 6. Final Model and Verification

Experiment 08 trained Logistic Regression on 48,993 combined features:

- 48,609 word and character TF-IDF features.
- 384 MiniLM semantic features.

The saved artifacts are:

```text
outputs/models/
├── 08_hybrid_tfidf_vectorizer.joblib
├── 08_hybrid_classifier.joblib
└── 08_hybrid_model_metadata.json
```

Experiment 09 implemented an inference script that accepts raw requirement text, applies the original Phase 2 normalization, and returns predicted categories and model probabilities.

Experiment 10 successfully verified:

- Model artifact checksums and class labels.
- Feature dimensions.
- Phase 2 and Phase 5 normalization consistency.
- Repeated prediction consistency.
- Equivalent raw-text input consistency.

Verification report: `outputs/reports/10_model_verification.json`

Large trained-model files may be excluded from GitHub through `.gitignore`.

## 7. Repository Structure

```text
promise_improved_model_phase_5/
├── scripts/
│   ├── 01_verify_phase5_baseline.py
│   ├── 02_hybrid_tfidf_minilm_cv.py
│   ├── 03_hybrid_error_analysis.py
│   ├── 04_difficult_class_analysis.py
│   ├── 05_minority_training_coverage.py
│   ├── 06_hybrid_class_weight_cv.py
│   ├── 07_class_weight_error_analysis.py
│   ├── 08_train_final_hybrid_model.py
│   ├── 09_predict_requirement.py
│   └── 10_verify_hybrid_model.py
├── outputs/
│   ├── metrics/
│   ├── reports/
│   ├── tables/
│   └── models/
├── README.md
├── EXPERIMENT_LOG.md
└── code_outputs_phase5.txt
```

## 8. Setup and Usage

Phase 5 uses Python, NumPy, pandas, SciPy, scikit-learn, PyTorch, and Hugging Face Transformers.

Activate the existing transformer environment from the thesis root:

```bash
source .venv-transformers/bin/activate
cd promise_improved_model_phase_5
```

To run a prediction using the saved model:

```bash
python scripts/09_predict_requirement.py \
  --text "The system shall allow users to reset their passwords." \
  --top-k 3
```

The pretrained MiniLM checkpoint and locally generated model artifacts must be available.

## 9. Research Limitations

- SC and PO remain difficult to classify because of limited examples and project diversity.
- Repeated development cross-validation may influence model-selection decisions.
- The original Phase 3 test partition was previously examined and is not an untouched independent test set.
- Model probabilities have not been independently calibrated.
- Technical verification confirms reproducible inference behavior, not independent classification accuracy.

## 10. Conclusion and Future Work

Phase 5 completed the development, training, and technical verification of a hybrid lexical-semantic requirements classifier.

The selected model achieved **32.00% mean cross-validation macro-F1**, compared with **28.16%** for the Phase 4 baseline.

Future research will focus on independent evaluation, improving difficult requirement classes, and investigating additional methods for cross-project generalization.

**Phase 5 Status: Completed.**