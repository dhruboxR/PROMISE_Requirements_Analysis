# Phase 7 Experiment 10C

## AI-Assisted Fuzzy-Match Assessment

### Method

Experiment 08B generated TF-IDF character n-gram similarity candidates for PURE passages not matched by Experiment 08A.

Experiment 08C selected 120 unique PURE candidates using reproducible stratified random sampling across five similarity-score ranges.

The highest-scoring PROMISE+ match for each selected candidate was assessed through ChatGPT-assisted textual comparison.

Decisions were recorded as Same requirement, Partial overlap, Different requirement, or Uncertain.

### Overall assessment

| Decision | Count | Percent of sample |
| --- | --- | --- |
| Same requirement | 46 | 38.33 |
| Partial overlap | 36 | 30.0 |
| Different requirement | 38 | 31.67 |
| Uncertain | 0 | 0.0 |

### Assessment by similarity range

| SimilarityGroup | PopulationCandidates | ReviewedPairs | SameRequirement | PartialOverlap | DifferentRequirement | Uncertain | SameRatePercent | AnyOverlapRatePercent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.90-1.00 | 106 | 30 | 23 | 7 | 0 | 0 | 76.67 | 100.0 |
| 0.80-0.90 | 151 | 30 | 19 | 8 | 3 | 0 | 63.33 | 90.0 |
| 0.60-0.80 | 286 | 20 | 4 | 14 | 2 | 0 | 20.0 | 90.0 |
| 0.40-0.60 | 789 | 20 | 0 | 7 | 13 | 0 | 0.0 | 35.0 |
| 0.00-0.40 | 5932 | 20 | 0 | 0 | 20 | 0 | 0.0 | 0.0 |

### Reviewer confidence

| Decision | Total | High | Medium | Low |
| --- | --- | --- | --- | --- |
| Same requirement | 46 | 36 | 10 | 0 |
| Partial overlap | 36 | 31 | 5 | 0 |
| Different requirement | 38 | 37 | 1 | 0 |
| Uncertain | 0 | 0 | 0 | 0 |

### Findings

- High-similarity pairs frequently express the same core requirement.
- Partial matches frequently arise from different requirement boundaries, omitted sentences, or additional conditions.
- Lower-similarity pairs more often express different requirements.
- PDF table extraction, inserted section identifiers, and line-merging artifacts affect textual matching.
- Similarity scores should prioritize source verification, not automatically establish requirement equivalence.

### Limitations

- All decisions are AI-assisted assessments, not independent expert-validated ground truth.
- The sample is stratified by similarity score; raw overall proportions are not population prevalence estimates.
- Only the highest-scoring candidate match per sampled PURE passage was assessed.
- Uncertain cases remain unresolved.
- No validated precision or recall is reported.
- Source-document independence and annotation quality are not established by this assessment.
- No PURE records are approved for training.

### Outputs

- 10c_similarity_group_results.csv
- 10c_confidence_distribution.csv
- 10c_flagged_matches.csv
- 10c_fuzzy_review_summary.json
