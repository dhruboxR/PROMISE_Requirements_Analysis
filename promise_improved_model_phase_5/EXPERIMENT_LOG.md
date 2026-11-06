# Phase 5: Experiment Log

**Research Project:** PROMISE+ Requirements Analysis  
**Phase:** Improved Requirements Classification Framework  
**Current Progress:** Experiments 01–07 completed  
**Primary Evaluation Metric:** Macro-F1  
**Development Dataset:** 3,127 requirements  
**Evaluation Strategy:** Three-fold group-aware cross-validation

---

## 1. Purpose

This experiment log documents the methodology, configuration, results, observations, and conclusions of each experiment performed during Phase 5.

The main research objective is to improve cross-project software requirements classification using a hybrid representation that combines lexical TF-IDF features with semantic MiniLM embeddings.

The experiments build on the Phase 4 findings, where word and character TF-IDF with Logistic Regression achieved a mean macro-F1 of 0.2816 under three-fold group-aware cross-validation.

Phase 5 investigates whether hybrid feature representations and controlled class weighting can improve this result.

All seven completed experiments use the development dataset or saved development predictions. The original 550-record test partition was not accessed.

---

## 2. Common Experimental Setup

### 2.1 Dataset

The experiments use the PROMISE+ requirements classification dataset.

| Property | Value |
|---|---|
| Original requirements | 3,677 |
| Total projects | 88 |
| Requirement classes | 12 |
| Development requirements | 3,127 |
| Connected development groups | 66 |
| Original test requirements | 550 |
| Cross-validation folds | 3 |
| Random seed | 42 |

The requirement categories are:

`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`

The development dataset combines the original training and validation partitions from Phase 2.

### 2.2 Evaluation Protocol

The experiments use `StratifiedGroupKFold` with three folds, shuffling enabled, and random seed 42.

Connected groups are constructed using software project membership and matching normalized requirement text.

This prevents the same project or identical normalized text from appearing in both the training and validation portions of a fold.

| Fold | Training records | Validation records |
|---|---:|---:|
| 1 | 2,159 | 968 |
| 2 | 2,024 | 1,103 |
| 3 | 2,071 | 1,056 |

### 2.3 Evaluation Metrics

The following metrics are used:

- Accuracy
- Macro-F1
- Balanced accuracy
- Weighted-F1
- Per-class precision
- Per-class recall
- Per-class F1

Mean cross-validation metrics are calculated from the three fold-level scores.

Pooled out-of-fold (OOF) metrics are calculated from all held-out predictions combined.

These metrics are recorded separately.

### 2.4 Shared Model Components

**Lexical representation**

Word and character TF-IDF features implemented using the Phase 4 feature-generation procedures.

**Semantic representation**

Pretrained model: `sentence-transformers/all-MiniLM-L6-v2`

Embedding dimension: 384

Maximum token length: 128

The semantic encoder remains frozen.

**Classifier**

Logistic Regression with:

- `C=10.0`
- `max_iter=2000`
- `random_state=42`

The experiments compare different semantic feature weights and class-weighting strategies.

---

# Experiment 01: Baseline and Fold Verification

## Objective

Verify the Phase 4 reference baseline and ensure that Phase 5 uses exactly the same development records, grouping strategy, and cross-validation folds.

## Script

`scripts/01_verify_phase5_baseline.py`

## Methodology

1. Load the validated Phase 4 development dataset.
2. Verify the expected requirement count and class labels.
3. Reconstruct connected groups.
4. Reconstruct the three group-aware folds.
5. Verify that the fold assignments match the previously saved Phase 4 predictions.
6. Load the saved Phase 4 Logistic Regression baseline metrics.
7. Save the fixed fold assignments and baseline metadata.

No model training was performed during this experiment.

## Results

Development requirements: 3,127

Connected groups: 66

Classes: 12

Fold alignment with Phase 4: Verified

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.6254 | 0.0364 |
| Macro-F1 | 0.2816 | 0.0424 |
| Balanced accuracy | 0.2827 | 0.0228 |
| Weighted-F1 | 0.5878 | 0.0590 |

## Findings

The fixed evaluation protocol is consistent with Phase 4.

The baseline was recovered from the original saved results without retraining the classifier.

The verified folds provide a common reference for subsequent Phase 5 experiments.

## Outputs

- `outputs/tables/01_fixed_fold_assignments.csv`
- `outputs/reports/01_baseline_manifest.json`

## Conclusion

The Phase 5 development environment and fold assignments were successfully verified.

**Status:** Completed.

---

# Experiment 02: Hybrid TF-IDF and MiniLM Classification

## Objective

Investigate whether combining lexical TF-IDF features with semantic MiniLM embeddings improves classification performance compared with the Phase 4 lexical baseline.

## Script

`scripts/02_hybrid_tfidf_minilm_cv.py`

## Methodology

The experiment combines two feature representations:

1. Word and character TF-IDF features.
2. Frozen MiniLM sentence embeddings.

The representations are concatenated to create a hybrid feature vector.

The semantic-feature weight controls the relative scale of the MiniLM embedding component.

Three configurations were evaluated:

| Configuration | Semantic weight |
|---|---:|
| TF-IDF baseline | 0.0 |
| Hybrid 0.5 | 0.5 |
| Hybrid 1.0 | 1.0 |

All three configurations used Logistic Regression with balanced class weighting, `C=10.0`, and the same evaluation folds.

TF-IDF features were fitted on the training portion of each fold.

The MiniLM encoder was frozen and did not use class labels for embedding generation.

## Fold-Level Results

### Fold 1

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| TF-IDF baseline | 0.6550 | 0.2890 | 0.2957 |
| Hybrid 0.5 | 0.6674 | 0.2929 | 0.3169 |
| Hybrid 1.0 | 0.6674 | 0.3041 | 0.3321 |

### Fold 2

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| TF-IDF baseline | 0.5848 | 0.2359 | 0.2563 |
| Hybrid 0.5 | 0.5884 | 0.2700 | 0.2851 |
| Hybrid 1.0 | 0.5748 | 0.2754 | 0.2883 |

### Fold 3

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| TF-IDF baseline | 0.6364 | 0.3198 | 0.2960 |
| Hybrid 0.5 | 0.6600 | 0.3581 | 0.3551 |
| Hybrid 1.0 | 0.6468 | 0.3629 | 0.3804 |

## Three-Fold Summary

| Configuration | Mean accuracy | Mean macro-F1 | Mean balanced accuracy | Mean weighted-F1 |
|---|---:|---:|---:|---:|
| TF-IDF baseline | 0.6254 | 0.2816 | 0.2827 | 0.5878 |
| Hybrid 0.5 | **0.6386** | 0.3070 | 0.3190 | 0.6035 |
| Hybrid 1.0 | 0.6296 | **0.3141** | **0.3336** | **0.6059** |

## Findings

Hybrid 1.0 improved macro-F1 relative to the TF-IDF baseline in all three folds.

The highest mean accuracy was obtained with Hybrid 0.5.

Hybrid 1.0 achieved the highest mean macro-F1 and balanced accuracy.

However, Hybrid 1.0 reduced Fold 2 accuracy compared with the lexical baseline.

## Outputs

- `outputs/metrics/02_hybrid_fold_metrics.csv`
- `outputs/metrics/02_hybrid_cv_summary.csv`
- `outputs/tables/02_hybrid_oof_predictions.csv`
- `outputs/reports/02_hybrid_experiment_metadata.json`
- `outputs/reports/02_tfidf_baseline_class_report.json`
- `outputs/reports/02_hybrid_0_5_class_report.json`
- `outputs/reports/02_hybrid_1_0_class_report.json`

## Conclusion

The hybrid representation improved development macro-F1 compared with the lexical baseline.

Hybrid 1.0 was selected as the reference representation for further class-weight analysis.

**Status:** Completed.

---

# Experiment 03: Hybrid Class-Level Error Analysis

## Objective

Identify which requirement categories benefit from hybrid representations and measure how individual predictions change relative to the TF-IDF baseline.

## Script

`scripts/03_hybrid_error_analysis.py`

## Methodology

The experiment analyzes saved out-of-fold predictions from Experiment 02.

The analysis includes:

1. Per-class precision, recall, and F1.
2. Comparison of the TF-IDF baseline and both hybrid configurations.
3. Record-level alignment using requirement identifiers.
4. Counting corrected and damaged predictions.
5. Fold-level accuracy comparisons.

No model training was performed.

## Pooled OOF Results

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| TF-IDF baseline | 0.6239 | 0.2768 | 0.2637 |
| Hybrid 0.5 | **0.6370** | 0.3051 | 0.2985 |
| Hybrid 1.0 | 0.6278 | **0.3148** | **0.3127** |

## Per-Class F1 Results

| Class | TF-IDF baseline | Hybrid 0.5 | Hybrid 1.0 |
|---|---:|---:|---:|
| F | 0.7835 | 0.7904 | 0.7892 |
| A | 0.5049 | 0.6207 | 0.6379 |
| L | 0.0617 | 0.0696 | 0.1250 |
| LF | 0.2178 | 0.2136 | 0.2435 |
| MN | 0.0522 | 0.0702 | 0.0650 |
| O | 0.2214 | 0.2357 | 0.2013 |
| PE | 0.4358 | 0.4692 | 0.5036 |
| SC | 0.0323 | 0.0163 | 0.0296 |
| SE | 0.5303 | 0.5877 | 0.5720 |
| US | 0.3742 | 0.4398 | 0.4217 |
| FT | 0.1081 | 0.1000 | 0.1224 |
| PO | 0.0000 | 0.0482 | 0.0667 |

## Prediction-Level Comparison

TF-IDF baseline versus Hybrid 1.0:

| Outcome | Requirements |
|---|---:|
| Both correct | 1,792 |
| Corrected by hybrid | 171 |
| Damaged by hybrid | 159 |
| Both incorrect | 1,005 |
| Net correct gain | 12 |

## Findings

Hybrid 1.0 improved F1 for 10 of the 12 requirement categories.

The largest gains occurred for A, PE, PO, and L.

The pooled F1 scores for O and SC decreased slightly.

The hybrid corrected 171 baseline errors but introduced 159 new errors.

Thus, the macro-F1 gain was larger than the gain in overall correct predictions.

## Outputs

- `outputs/tables/03_per_class_comparison.csv`
- `outputs/tables/03_prediction_comparison.csv`

## Conclusion

Hybrid features provided useful complementary information but did not improve every class.

O, SC, and PO were selected for additional diagnostic investigation.

**Status:** Completed.

---

# Experiment 04: Difficult-Class Error Investigation

## Objective

Investigate classification errors affecting O, SC, and PO, with particular attention to project-level performance and frequent misclassifications.

## Script

`scripts/04_difficult_class_analysis.py`

## Methodology

The experiment analyzes saved predictions from Experiment 03.

For each target class, the analysis calculates:

- Number of development requirements.
- Number of represented projects.
- Correct predictions under the baseline and hybrid.
- Corrections and new errors.
- Most frequent predicted classes.
- Project-level classification performance.

No model training was performed.

## Overall Results

| Class | Requirements | Projects | Baseline correct | Hybrid correct | Hybrid recall |
|---|---:|---:|---:|---:|---:|
| O | 136 | 23 | 29 | 30 | 0.2206 |
| SC | 86 | 9 | 2 | 2 | 0.0233 |
| PO | 66 | 7 | 0 | 3 | 0.0455 |

### Class O

The hybrid classified 30 of 136 O requirements correctly.

Frequent incorrect predictions included:

- F: 60 requirements.
- MN: 12 requirements.
- PO: 7 requirements.
- PE: 7 requirements.
- US: 7 requirements.

Project-level results varied considerably.

Project 6 achieved 11 correct predictions among 15 O requirements.

Project 74 achieved zero correct predictions among 16 O requirements.

### Class SC

The hybrid classified only 2 of 86 SC requirements correctly.

The most frequent predicted class was F, with 52 SC requirements assigned to it.

Project 80 contained 54 SC requirements, all of which were misclassified.

SC examples were represented across only nine development projects.

### Class PO

The hybrid correctly classified 3 of 66 PO requirements, while the TF-IDF baseline correctly classified none.

Frequent incorrect predictions included:

- F: 27 requirements.
- O: 19 requirements.
- PE: 11 requirements.

Project 59 contributed 34 PO requirements.

Project 52 contributed 24 PO requirements.

Together, these two projects accounted for 58 of the 66 development PO requirements.

## Findings

The hybrid performs inconsistently across software projects.

The SC and PO categories show substantial concentration within a small number of projects.

These observations are consistent with cross-project generalization difficulties, but the experiment does not establish that project concentration is the only cause.

## Outputs

- `outputs/tables/04_difficult_class_errors.csv`
- `outputs/tables/04_difficult_class_project_analysis.csv`
- `outputs/tables/04_hybrid_confusion_matrix.csv`

## Conclusion

The analysis identified severe SC and PO classification problems and substantial project-level performance differences.

The next experiment investigates training-class coverage across folds.

**Status:** Completed.

---

# Experiment 05: Minority-Class Training Coverage

## Objective

Determine how the availability of O, SC, and PO training examples varies between project-aware folds.

## Script

`scripts/05_minority_training_coverage.py`

## Methodology

The experiment uses the fixed fold assignments saved in Experiment 01.

For each fold and target class, it calculates:

1. Training-example count.
2. Number of training projects.
3. Validation-example count.
4. Number of validation projects.
5. Projects contributing the largest numbers of training examples.

No model training was performed.

## Training Coverage Results

| Class | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| O | 90 examples, 15 projects | 92 examples, 18 projects | 90 examples, 13 projects |
| SC | 74 examples, 8 projects | 74 examples, 5 projects | 24 examples, 5 projects |
| PO | 41 examples, 5 projects | 31 examples, 5 projects | 60 examples, 4 projects |

## Validation Coverage

| Class | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|
| O | 46 examples, 8 projects | 44 examples, 5 projects | 46 examples, 10 projects |
| SC | 12 examples, 1 project | 12 examples, 4 projects | 62 examples, 4 projects |
| PO | 25 examples, 2 projects | 35 examples, 2 projects | 6 examples, 3 projects |

## Detailed Findings

### SC

Fold 3 contained only 24 SC training examples, while 62 SC requirements appeared in the held-out portion.

Project 80 contributed 54 SC requirements and was held out in Fold 3.

This created a difficult evaluation scenario in which the model had limited SC examples from other projects.

### PO

In Fold 1, Project 59 contributed 34 of the 41 PO training examples.

In Fold 2, Project 52 contributed 24 of the 31 PO training examples.

Therefore, PO training data were heavily concentrated in one project for both folds.

### O

O had a broader distribution across projects than SC and PO.

However, classification performance remained inconsistent, indicating that example counts and project coverage alone do not fully explain the observed errors.

## Outputs

- `outputs/tables/05_minority_training_coverage.csv`

## Conclusion

Minority-class training coverage changes considerably across project-aware folds.

Limited project diversity is a plausible contributor to poor cross-project generalization.

The next experiment examines whether different class-weighting strategies can improve the hybrid classifier.

**Status:** Completed.

---

# Experiment 06: Controlled Hybrid Class-Weight Comparison

## Objective

Compare unweighted, tempered, and fully balanced Logistic Regression using the Hybrid 1.0 feature representation.

## Script

`scripts/06_hybrid_class_weight_cv.py`

## Methodology

All configurations use:

- Word and character TF-IDF.
- Frozen MiniLM embeddings.
- Semantic feature weight 1.0.
- Logistic Regression with C=10.0.
- Three-fold group-aware cross-validation.

The three class-weighting methods are:

**Unweighted:** Standard Logistic Regression without class weighting.

**Tempered:** Square-root inverse-frequency weighting, normalized using the training class distribution.

**Balanced:** Standard inverse-frequency balanced class weighting.

Class weights are computed independently from each training fold.

## Fold-Level Results

### Fold 1

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Unweighted | 0.6942 | 0.2772 | 0.2775 |
| Tempered | 0.6890 | 0.3000 | 0.3099 |
| Balanced | 0.6674 | 0.3041 | 0.3321 |

### Fold 2

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Unweighted | 0.6174 | 0.2711 | 0.2548 |
| Tempered | 0.6047 | 0.2766 | 0.2747 |
| Balanced | 0.5748 | 0.2754 | 0.2883 |

### Fold 3

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Unweighted | 0.6828 | 0.3464 | 0.3288 |
| Tempered | 0.6714 | 0.3834 | 0.3747 |
| Balanced | 0.6468 | 0.3629 | 0.3804 |

## Three-Fold Summary

| Configuration | Mean accuracy | Mean macro-F1 | Mean balanced accuracy | Mean weighted-F1 |
|---|---:|---:|---:|---:|
| Unweighted | **0.6648** | 0.2983 | 0.2870 | 0.6056 |
| Tempered | 0.6551 | **0.3200** | 0.3198 | **0.6116** |
| Balanced | 0.6296 | 0.3141 | **0.3336** | 0.6059 |

## Difficult-Class Recall

Mean recall across the three folds:

| Class | Unweighted | Tempered | Balanced |
|---|---:|---:|---:|
| O | 0.1993 | **0.2292** | 0.2220 |
| SC | 0.0278 | **0.0332** | **0.0332** |
| PO | 0.0133 | 0.0229 | **0.0324** |

## Findings

Tempered weighting achieved the highest mean CV macro-F1.

Unweighted classification achieved the highest mean accuracy.

Fully balanced weighting achieved the highest mean balanced accuracy.

The balanced configuration reproduced Experiment 02 Hybrid 1.0 results.

The improvement in the primary metric did not resolve the severe limitations affecting SC and PO.

## Outputs

- `outputs/metrics/06_class_weight_fold_metrics.csv`
- `outputs/metrics/06_class_weight_summary.csv`
- `outputs/tables/06_class_weight_oof_predictions.csv`
- `outputs/reports/06_class_weight_manifest.json`

## Conclusion

Tempered weighting was selected as the provisional primary configuration based on mean cross-validation macro-F1.

Fully balanced weighting remains an important comparison configuration.

**Status:** Completed.

---

# Experiment 07: Class-Weight Error Analysis

## Objective

Compare the per-class behavior and record-level prediction differences between unweighted, tempered, and fully balanced hybrid classifiers.

## Script

`scripts/07_class_weight_error_analysis.py`

## Methodology

The experiment uses saved out-of-fold predictions from Experiment 06.

It calculates:

- Pooled OOF accuracy.
- Pooled OOF macro-F1.
- Pooled OOF balanced accuracy.
- Per-class F1 and recall.
- Tempered versus balanced prediction agreement.
- Differences in correctly classified requirements.

No model training was performed.

## Pooled OOF Results

| Configuration | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Unweighted | **0.6633** | 0.2874 | 0.2569 |
| Tempered | 0.6533 | 0.3114 | 0.2900 |
| Balanced | 0.6278 | **0.3148** | **0.3127** |

## Per-Class F1 Comparison

| Class | Unweighted | Tempered | Balanced |
|---|---:|---:|---:|
| F | 0.8004 | 0.8010 | 0.7892 |
| A | 0.4545 | 0.5347 | 0.6379 |
| L | 0.0849 | 0.0917 | 0.1250 |
| LF | 0.1875 | 0.2243 | 0.2435 |
| MN | 0.0465 | 0.0755 | 0.0650 |
| O | 0.2379 | 0.2403 | 0.2013 |
| PE | 0.4758 | 0.4962 | 0.5036 |
| SC | 0.0177 | 0.0320 | 0.0296 |
| SE | 0.5959 | 0.5907 | 0.5720 |
| US | 0.4621 | 0.4482 | 0.4217 |
| FT | 0.0588 | 0.1538 | 0.1224 |
| PO | 0.0270 | 0.0488 | 0.0667 |

## Prediction-Level Agreement

Tempered versus balanced:

| Outcome | Requirements |
|---|---:|
| Both correct | 1,922 |
| Correct only by tempered | 121 |
| Correct only by balanced | 41 |
| Both incorrect | 1,043 |

Tempered weighting produced 80 additional correct predictions compared with full balanced weighting.

However, fully balanced weighting produced slightly higher pooled macro-F1.

## Findings

Tempered weighting achieved higher pooled F1 for F, MN, O, SC, SE, US, and FT.

Balanced weighting achieved higher pooled F1 for A, L, LF, PE, and PO.

The difference between the two methods depends on the evaluation metric and the requirement categories being emphasized.

Tempered weighting remains the provisional primary configuration because it achieved the highest mean cross-validation macro-F1, which was the predefined development metric.

## Outputs

- `outputs/tables/07_class_weight_per_class_comparison.csv`
- `outputs/tables/07_tempered_vs_balanced_predictions.csv`

## Conclusion

Experiment 07 confirmed that the choice of class-weighting strategy affects category-level performance and overall accuracy.

The tempered hybrid was retained as the provisional primary model.

**Status:** Completed.

---

## 3. Overall Phase 5 Results

### Mean Cross-Validation Comparison

| Model | Mean accuracy | Mean macro-F1 | Mean balanced accuracy |
|---|---:|---:|---:|
| Phase 4 TF-IDF baseline | 0.6254 | 0.2816 | 0.2827 |
| Hybrid 0.5, balanced | 0.6386 | 0.3070 | 0.3190 |
| Hybrid 1.0, balanced | 0.6296 | 0.3141 | 0.3336 |
| Hybrid 1.0, unweighted | **0.6648** | 0.2983 | 0.2870 |
| Hybrid 1.0, tempered | 0.6551 | **0.3200** | 0.3198 |

### Primary Research Result

The provisional best configuration, measured by mean cross-validation macro-F1, is:

**Word + Character TF-IDF + Frozen MiniLM + Tempered Logistic Regression**

Compared with the Phase 4 baseline:

- Mean accuracy improved from 0.6254 to 0.6551.
- Mean macro-F1 improved from 0.2816 to 0.3200.
- Mean balanced accuracy improved from 0.2827 to 0.3198.

The observed mean macro-F1 improvement is 0.0384, equivalent to 3.84 percentage points.

These are exploratory development results and do not establish independently verified performance.

---

## 4. Remaining Research Challenges

Despite the improvement, several issues remain unresolved:

1. SC and PO have extremely low classification recall.
2. The available minority-class examples are concentrated in relatively few projects.
3. Different project distributions produce substantially different performance across folds.
4. Class weighting produces trade-offs between overall accuracy and minority-class performance.
5. Repeated experimentation using the same folds limits the independence of model-selection results.
6. The original Phase 3 test set was previously evaluated and should not be treated as untouched for future model selection.

The results support further investigation of project-robust representations, minority-class handling, and independent evaluation.

---

## 5. Planned Experiments

### Experiment 08: Consolidated Hybrid Model Training

**Status:** Planned

Train the selected tempered hybrid on all 3,127 development requirements.

Save the TF-IDF feature extractor, Logistic Regression classifier, configuration metadata, and class-label information.

### Experiment 09: Standalone Inference Pipeline

**Status:** Planned

Build a prediction script that loads the saved hybrid model and classifies new requirement texts using the same preprocessing and feature-extraction procedures.

### Experiment 10: Reproducibility and Robustness Checks

**Status:** Planned

Verify model loading, prediction consistency, artifact integrity, and input-handling behavior.

Document remaining limitations and requirements for independent evaluation.

---

## 6. Final Research Notes

Experiments 01–07 establish a reproducible development workflow and a provisional hybrid classification model.

The experimental evidence suggests that combining lexical and semantic representations, together with controlled class weighting, can improve macro-F1 on the PROMISE+ development folds.

However, improvement is not uniform across requirement categories.

The current model should be regarded as a development-stage candidate rather than a fully validated classifier for unseen projects.

**Phase 5 status:** Experiments 01–07 completed. Model consolidation and inference development remain pending.