# PROMISE+ Dataset: Phase 1 Exploratory Analysis


Generated: 2026-10-31 19:40 UTC


## 1. Dataset Description


The validated PROMISE+ dataset contains **3,677 requirements**
associated with **88 unique project identifiers**.

The dataset contains three attributes:

- **ProjectID:** Identifier of the source project.
- **RequirementText:** Natural-language requirement.
- **_class_:** Requirement classification label.

There are **12 observed classification labels**.
All analyses use the validated working copy; the raw dataset is preserved.



## 2. Dataset Validation


The source ARFF file required formatting corrections before parsing.

The working copy was successfully loaded with 3,677 records.

Number of entries in the comma-repair audit:
**1010**.

Legacy punctuation escape sequences were also corrected during
validation. Those inferred substitutions require a complete
line-level repair audit before publication.

Original SHA-256:
`9772c11732b93a3b03d96f07bfacaaa552992ea492cb020c98284c930dbf510a`

Validated SHA-256:
`9c836825376dae7913a61ec0844502d8a706ca66a11aebcdd13e6e8cca77a173`



## 3. Data Quality

| Metric | Value |
| --- | --- |
| Total records | 3677 |
| Unique requirement texts | 3580 |
| Missing ProjectID | 0 |
| Missing RequirementText | 0 |
| Missing class labels | 0 |
| Blank requirement texts | 0 |
| Exact duplicate excess records | 21 |
| Records involved in exact duplicates | 38 |
| Normalized duplicate excess records | 113 |
| Records involved in normalized duplicates | 222 |


## 4. Class Distribution

| Class | Count | Percentage |
| --- | --- | --- |
| F | 2258 | 61.41 |
| SE | 237 | 6.45 |
| US | 212 | 5.77 |
| L | 209 | 5.68 |
| PE | 163 | 4.43 |
| O | 157 | 4.27 |
| SC | 95 | 2.58 |
| LF | 89 | 2.42 |
| PO | 76 | 2.07 |
| MN | 75 | 2.04 |
| A | 71 | 1.93 |
| FT | 35 | 0.95 |


The majority class is **F** and the minority
class is **FT**.

The majority-to-minority frequency ratio is approximately
**64.51:1**.

This imbalance makes macro-F1, balanced accuracy, and
per-class performance important evaluation metrics.



## 5. Requirement Length Statistics

| Statistic | Character count | Word count |
| --- | --- | --- |
| count | 3677.0 | 3677.0 |
| mean | 142.85 | 22.69 |
| std | 101.72 | 16.18 |
| min | 26.0 | 3.0 |
| 25% | 78.0 | 13.0 |
| 50% | 112.0 | 18.0 |
| 75% | 169.0 | 27.0 |
| max | 1136.0 | 208.0 |


The mean requirement length is **22.69 words**,
while the median is **18 words**.

The longest requirement contains **208 words**.

Word counts use whitespace tokenization, so values may differ
from those produced by linguistic tokenizers.



## 6. Duplicate and Label Analysis

| Metric | Count |
| --- | --- |
| Exact duplicate excess records | 21 |
| Normalized duplicate excess records | 113 |
| Repeated texts across projects | 89 |
| Rows involved across projects | 178 |
| Texts with multiple labels | 4 |
| Rows involved in label conflicts | 8 |


### Potential annotation disagreements


| Labels | Projects | Example |
| --- | --- | --- |
| MN, PO | 58, 59 | Software Accessibility – all systems and application source code shall be available to or on the systems that execute it.  |
| O, PE | 46, 67 | The product shall be based on web and has to be run from a web server. |
| SE, US | 46, 67 | The system shall confirm all transactions with the customer’s web browser. |
| LF, US | 46, 67 | The system shall provide a uniform look and feel between all the web pages. |


Identical requirements assigned different class labels may
indicate annotation disagreements or category ambiguity.
They do not directly establish contradictions between
software requirements.



## 7. Methodological Limitations


- Some ARFF formatting corrections were necessary to load the dataset.
- Legacy punctuation substitutions were inferred and should be audited.
- Duplicate records may affect model evaluation if they cross data splits.
- Identical requirements with different labels require manual review.
- Class imbalance may reduce minority-class recognition.
- Requirement length varies considerably across records.
- The dataset does not directly provide labels for semantic contradiction detection.
- Project-wise and text-group-wise split strategies should be evaluated before training.



## 8. Recommendations for Phase 2


1. Complete the line-level audit for all dataset repairs.
2. Manually review the identified conflicting-label records.
3. Define a consistent duplicate-handling policy.
4. Establish leakage-controlled training and testing splits.
5. Preserve all original requirement texts.
6. Perform controlled preprocessing experiments.
7. Evaluate models using macro-F1 and class-wise metrics.
8. Keep requirement contradiction detection separate from label disagreement analysis.
