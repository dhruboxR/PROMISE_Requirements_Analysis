# Phase 8: PROMISE+ Classification Improvement Experiments

**Project:** PROMISE+ Requirements Classification Thesis  
**Repository folder:** `promise_classification_phase_8`  
**Status:** **Closed after Experiment 24B**  
**Purpose:** Investigate methods to improve 12-class classification, especially minority-class recognition and cross-project generalization.

## Research context

PROMISE+ contains **3,677 requirements from 88 projects**, labeled across 12 categories:

`F`, `A`, `L`, `LF`, `MN`, `O`, `PE`, `SC`, `SE`, `US`, `FT`, `PO`.

Existing data partitions include 2,574 training, 553 validation, and 550 original test records. For exploratory project-aware development, Phase 8 used **3,127 development records** (training + validation) and the pre-existing fixed group-aware assignments. The original test set was **not accessed during Phase 8**. Training/validation splits were checked for project and connected-group overlap.

### Reference benchmark

The previously established **Phase 5 hybrid** consisted of word and character TF-IDF, frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings, and normalized square-root-balanced Logistic Regression (`C=10`). Across its **three existing outer development folds**, it reported:

| Metric | Mean |
|---|---:|
| Accuracy | **65.51%** |
| Macro-F1 | **32.00%** |
| Balanced accuracy | **31.98%** |

These are **development results, not independent final test estimates**. They cannot be directly ranked against experiments evaluated on different folds.

## Experiments and outcomes

| Experiment | Focus | Main finding |
|---|---|---|
| 01 | Hard/soft hierarchical classification | Hard: accuracy 56.75%, macro-F1 28.06%; soft: 63.28%, 30.02%; neither beat the hybrid reference |
| 02 | Out-of-fold minority-class diagnostics | Baseline struggled with O, SC, and PO; particularly low SC and PO recall |
| 03–04 | Binary minority specialists and threshold exploration | Some pairwise discrimination was promising, but threshold-based correction was risky |
| 05–06 | Nested class-aware specialist correction and audit | Accuracy 64.61%, macro-F1 32.02%; 63 changed predictions yielded 2 corrections and 31 newly wrong predictions, net −29 |
| 07 | Nested semantic-feature weight selection | Mean accuracy 63.30%, macro-F1 30.04%; below hybrid reference |
| 08–11 | MiniLM fine-tuning and error analysis | Fine-tuning did not consistently exceed the hybrid; fold-one full-training fine-tuning produced accuracy 63.95%, macro-F1 26.44% |
| 12 | Class-weight sensitivity | Exponent 0.25 selected in exploratory inner tuning; fold-one accuracy 66.74%, macro-F1 25.97%, below hybrid fold-one macro-F1 |
| 13–16 | Minority-class feasibility, project errors, source-text review | Severe class-to-project concentration and shifts in requirement wording across projects |
| 17–18 | Annotation review templates and assisted assessment | 103 review records prepared; 24 priority records assessed provisionally, without changing ground truth |
| 19 | Cross-project supervised contrastive loss | Matched class-aware CE: 47.34% accuracy, 31.00% macro-F1; CE + SupCon: 44.05%, 28.00%, on one inner split |
| 20 | Sampling ablation, one inner split | Shuffled CE: 57.47%/25.02% accuracy/macro-F1; class-aware CE: 47.34%/31.00% |
| 21 | Nine matched inner-fold sampling comparisons | Shuffled CE 61.12% accuracy, 24.94% macro-F1; fully class-aware CE 38.05%, 25.00% |
| 22 | Mixed 75/25 sampling | 60.54% accuracy, **29.41% macro-F1** across nine inner evaluations; macro-F1 exceeded shuffled CE in 8/9 comparisons |
| 23A–23B | Hybrid implementation audit and matched hybrid comparison | Hybrid 64.94% accuracy, 28.41% macro-F1; mixed 60.54%, 29.41%; mixed macro-F1 higher in 5/9, hybrid accuracy higher in 7/9 with one tie |
| 24A | Reproduced-model fixed 75/25 probability fusion | On one 790-record inner validation set: hybrid 64.05%/34.18%; mixed 61.65%/27.86%; fusion 64.18%/32.53% |
| 24B | Fusion error analysis | 24 predictions changed: 7 hybrid errors corrected, 6 new errors, 11 still wrong; no convincing macro-F1 benefit |

**Metric convention:** Percentages above are rounded. When two metrics are shown as `accuracy/macro-F1`, they refer to the same stated evaluation protocol. One-fold and nine-inner-fold results are not directly interchangeable with the three-fold reference benchmark.

## Annotation and project-generalization findings

The Phase 8 analysis identified highly concentrated labels: for example, **153 of 174 development-class L records** came from project 82, while **54 of 86 SC records** came from project 80. Class PO also appeared in relatively few projects. Such concentration makes reliable generalization to unseen projects difficult.

An assisted review of 24 intentionally selected requirements produced:

| Provisional review status | Count |
|---|---:|
| Potential mismatch | 12 |
| Potentially ambiguous | 7 |
| Consistent | 3 |
| Insufficient context | 2 |

These assessments are **not verified annotation errors**, and the selection is **not representative** of the whole dataset. Original labels were never changed. Official category definitions and source context should be checked before drawing conclusions about label quality.

## Key technical configurations

### Hybrid reference

- Word TF-IDF: 1–2 grams, up to 25,000 features.
- Character TF-IDF: `char_wb`, 3–5 grams, up to 40,000 features.
- Frozen MiniLM: mean pooling, L2-normalized 384-dimensional embeddings.
- Semantic block weight: 1.0.
- Logistic Regression: `C=10`, `max_iter=2000`.
- Class weighting: square root of balanced weights, normalized to weighted mean 1.
- TF-IDF vectorizers fit only on each training partition.

### Trainable MiniLM and sampling experiments

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`.
- Four epochs, learning rate `2e-5`, max token length 128, batch size 16.
- Class-weight exponent: 0.25 for sampling comparisons.
- Fully class-aware batches: eight class pairs (16 records), preferring cross-project pairs.
- Mixed batches: 12 ordinary samples plus 4 focused samples from two non-F classes, with cross-project pairing preferred.
- In Experiment 22, mean sampled F share was **46.13%**; distinct sampled positions / total draws averaged approximately **78%**.

## Detailed multi-split comparison

Experiments 21–23B used the same **nine matched inner project-aware splits** drawn from three outer-training partitions. Their data overlap; they are **not nine independent external tests**.

| Model | Mean accuracy | Mean macro-F1 | Mean F recall | Mean PO recall |
|---|---:|---:|---:|---:|
| Shuffled CE | 61.12% | 24.94% | 80.97% | 3.20% |
| Fully class-aware CE | 38.05% | 25.00% | 36.36% | 26.05% |
| Mixed 75/25 CE | 60.54% | **29.41%** | 78.08% | **27.70%** |
| Hybrid tempered | **64.94%** | 28.41% | **89.61%** | 0.44% |

The mixed model modestly exceeded hybrid **mean** macro-F1 by about one percentage point, but did so in only five of nine paired comparisons and sacrificed 4.40 percentage points of accuracy. This does **not** establish reliable superiority over the hybrid.

## Fusion experiment decision

In Experiment 24A, both standalone models exactly reproduced their previously saved inner-fold metrics. Fixed probability fusion used:

```text
P_fusion = 0.75 * P_hybrid + 0.25 * P_mixed
```

On the 790-record validation partition, fusion corrected seven hybrid errors, introduced six, and left eleven changed predictions incorrect. It increased overall accuracy by one correct prediction but reduced macro-F1 from **34.18%** (hybrid) to **32.53%**. Experiment 24B confirmed weak class-level gains. **Simple probability fusion was discontinued.** No probability calibration or independent weight selection was performed.

## Known output artifacts

Within `promise_classification_phase_8/`, key artifacts include:

```text
scripts/
  19_cross_project_contrastive_feasibility.py
  20_sampling_ablation.py
  21_multisplit_sampling_evaluation.py
  22_mixed_class_aware_sampling.py
  23_matched_hybrid_evaluation.py
  24_probability_fusion_feasibility.py
  24b_fusion_error_analysis.py
outputs/
  tables/
    17_annotation_review_template.csv
    18_priority_annotation_review.csv
    18_priority_annotation_review_assisted.csv
    21_sampling_summary.csv
    21_sampling_paired_differences.csv
    22_three_strategy_summary.csv
    22_mixed_paired_differences.csv
    22_mixed_sampling_counts.csv
    23_hybrid_vs_mixed_summary.csv
    23_hybrid_vs_mixed_paired.csv
    24_matched_probabilities_fold1.csv
    24_fusion_fold1_summary.csv
    24b_fusion_class_error_summary.csv
    24b_fusion_changed_predictions.csv
  reports/
    24_fusion_fold1_audit.json
```

This is a **selected inventory**, not a full listing. Earlier numbered scripts and additional tables/logs should be inspected in the local directory.

## Limitations and research integrity

1. **Repeated inspection of development splits:** Many methods were designed after seeing earlier fold results. These data are exploratory; they cannot provide an unbiased final estimate of the selected method.
2. **Project concentration:** Several rare classes originate largely from one or two projects, sometimes leaving very small training or validation supports.
3. **Overlapping inner partitions:** The nine comparisons reuse development examples and should not be reported as independent samples in significance tests.
4. **Model selection:** Some ratios, architectures, and loss settings were chosen after previous results; no independent confirmation has yet been demonstrated.
5. **Annotation review:** Potential mismatches were AI-assisted observations, not independently adjudicated relabeling decisions.
6. **Test integrity:** Phase 8 did **not** access the original test dataset or change original requirement labels.

## Final Phase 8 conclusion

The most promising new finding was that **moderate 75/25 mixed sampling** substantially improved macro-F1 relative to ordinary shuffled MiniLM fine-tuning on the same inner splits without a large accuracy penalty. However, it did **not** clearly outperform the established TF-IDF + frozen MiniLM hybrid. The hybrid retained higher accuracy; the mixed model modestly improved mean macro-F1 and some minority-class recall. Cross-project supervised contrastive loss, aggressive class-aware sampling, specialist correction, and simple probability fusion did not yield convincing overall gains under the evaluated setups.

**Phase 8 closed after Experiment 24B.** Preserve all outputs as historical development evidence. The next research effort is a **fresh restart** with a separately documented objective and evaluation protocol, rather than an unreported continuation of Phase 8.
