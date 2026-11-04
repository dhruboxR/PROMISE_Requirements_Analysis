# PROMISE+ Requirements Classification: Phase 4

## Advanced Models, Cross-Project Evaluation, and Error Analysis

### 1. Overview

Phase 4 investigates advanced machine learning and deep learning approaches for automated software requirements classification using the PROMISE+ dataset.

The goal is to evaluate whether pretrained transformer models, semantic sentence embeddings, and ensemble methods improve classification performance compared with traditional TF-IDF-based classifiers.

Special attention is given to minority-class performance and generalization to previously unseen software projects.

This phase contains Experiments 01–19, including model development, diagnostic experiments, group-aware cross-validation, and comparative analysis.

### 2. Dataset

The PROMISE+ dataset contains:

| Property | Value |
|---|---|
| Total requirements | 3,677 |
| Software projects | 88 |
| Requirement classes | 12 |
| Original training records | 2,574 |
| Original validation records | 553 |
| Original test records | 550 |
| Cross-validation development records | 3,127 |

The 12 classes are F, A, L, LF, MN, O, PE, SC, SE, US, FT, and PO.

Phase 4 uses the preprocessing outputs generated in Phase 2. The original test partition is not loaded during Phase 4 experiments.

### 3. Evaluation methodology

Two evaluation protocols were used.

**Original validation protocol:** Models were trained on 2,574 requirements and evaluated on 553 requirements from separate projects. This protocol supported initial feasibility studies and model development.

**Group-aware cross-validation protocol:** The original training and validation partitions were combined into 3,127 development requirements. Three-fold StratifiedGroupKFold was used with random seed 42.

Connected groups were constructed so that requirements from the same project or with matching normalized text remained in the same group. This reduced the risk of project and duplicate-text leakage.

The primary evaluation metrics were accuracy, macro-F1, balanced accuracy, and weighted-F1.

Macro-F1 was prioritized because the dataset contains severe class imbalance.

### 4. Models investigated

**Classical models**

- Logistic Regression with word TF-IDF
- Logistic Regression with word and character TF-IDF
- Linear Support Vector Machine

**Transformer models**

- DistilBERT
- Class-weighted DistilBERT
- Extended-training class-weighted DistilBERT
- Class-weighted RoBERTa

**Semantic embedding models**

- Frozen MiniLM sentence embeddings with balanced Logistic Regression
- Frozen MiniLM sentence embeddings with unweighted Logistic Regression

**Ensemble analysis**

- Prediction complementarity analysis
- Three-model majority voting
- Shared-error analysis

### 5. Original validation results

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Tuned Logistic Regression | 0.6691 | 0.3894 | 0.4131 |
| Standard DistilBERT | 0.5570 | 0.1887 | 0.2228 |
| Weighted DistilBERT, 3 epochs | 0.5986 | 0.2735 | 0.3211 |
| Weighted DistilBERT, 6 epochs | 0.5678 | 0.3174 | 0.3758 |
| Weighted RoBERTa | 0.5244 | 0.2966 | 0.3376 |
| Three-model majority voting | 0.6166 | 0.3449 | 0.3829 |

These results belong to the original 553-record validation protocol and should not be mixed with cross-validation results.

### 6. Group-aware cross-validation results

The following scores are means across three project-aware folds.

| Model | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Word + character TF-IDF Logistic Regression | 0.6254 | 0.2816 | 0.2827 |
| Weighted DistilBERT | 0.6172 | 0.2442 | 0.2793 |
| MiniLM + balanced Logistic Regression | 0.4699 | 0.2557 | 0.3217 |
| MiniLM + unweighted Logistic Regression | 0.6284 | 0.2621 | 0.2647 |

The strongest mean macro-F1 was obtained with word and character TF-IDF Logistic Regression.

Unweighted MiniLM achieved slightly higher mean accuracy, whereas balanced MiniLM achieved the highest mean balanced accuracy.

No advanced configuration consistently surpassed the classical baseline across all three metrics.

### 7. Main error-analysis findings

The experiments identified persistent difficulty with PO, SC, L, MN, FT, and LF.

In Experiment 16, Logistic Regression and DistilBERT both misclassified:

| Class | Shared errors | Total records |
|---|---:|---:|
| PO | 64 | 66 |
| SC | 83 | 86 |
| L | 166 | 174 |
| MN | 60 | 64 |
| FT | 28 | 30 |
| LF | 68 | 79 |

The models also displayed substantial differences in performance between projects.

Lexical similarity analysis showed that validation requirements from difficult classes often had low similarity to same-class training requirements.

This suggests a cross-project text-representation challenge, although category ambiguity and annotation consistency also require further investigation.

### 8. Ensemble findings

Majority voting among Logistic Regression, DistilBERT, and RoBERTa performed worse than Logistic Regression alone.

In the original validation experiment, majority voting changed 99 Logistic Regression predictions. Of those changes, 22 corrected errors, 51 introduced new errors, and 26 remained incorrect.

Group-aware comparison between Logistic Regression and DistilBERT showed 260 requirements correctly classified only by Logistic Regression and 237 correctly classified only by DistilBERT.

These complementary predictions suggest potential value in further fusion research, but no improvement from an ensemble was established in Phase 4.

### 9. Research limitations

The dataset contains substantial class imbalance and limited examples for rare classes.

Some categories may have overlapping definitions or inconsistent annotation patterns.

Project-aware cross-validation uses a limited number of connected groups, and performance varies across projects.

The original validation partition and development folds have influenced several research decisions. Therefore, subsequent improvements evaluated on these same data should be treated as exploratory rather than independently confirmed.

The original Phase 3 test partition was previously evaluated and its results examined. It was excluded from Phase 4, but it should not be reused as an untouched final test for selecting Phase 5 models.

### 10. Conclusion

Phase 4 established that stronger pretrained architectures do not automatically improve requirements classification on PROMISE+.

Class weighting improved minority-class recognition in DistilBERT, but it did not consistently surpass the classical baseline.

The principal remaining challenges are cross-project generalization, category imbalance, and weak representation of difficult classes.

These findings motivate Phase 5, which will investigate hybrid lexical-semantic representations and carefully controlled approaches to minority-class classification.

### 11. Reproducibility

All executable experiments are stored in `scripts/`. Generated metrics, prediction tables, reports, and model artifacts are stored in `outputs/`.

The exact Phase 4 Python environment is recorded in `requirements-transformers.lock.txt`.

See `EXPERIMENT_LOG.md` for individual experiment descriptions and `FINDINGS_AND_LIMITATIONS.md` for interpretation and methodological notes.
