# Phase 7: PURE Source Provenance and Leakage Audit

**Project:** PROMISE+ Requirements Classification Thesis  
**Repository folder:** `promise_pure_analysis_phase_7`  
**Status:** Closed; source verification and eligibility assessment completed  
**Scope:** Data provenance and source-document auditing, **not model training**

## Objective

Investigate whether original requirements documents from the PURE corpus can be traced to PROMISE+ requirements, and determine whether using those documents as additional training data would create source overlap or evaluation leakage. This phase examined document extraction, requirement matching, original-source context, project attribution, and eligibility under existing project-aware data splits.

## Data and safeguards

- **PROMISE+ classification task:** 3,677 requirements, 88 projects, 12 labels.
- **Fixed development partition:** 3,127 requirements, with held-out development folds containing 968, 1,103, and 1,056 records.
- **PURE source inventory:** 79 original documents, comprising 62 PDF, 13 DOC, 3 HTML/HTM, and 1 RTF; 18 XML files were also inventoried.
- Original PROMISE+ labels and source files were not altered. PURE text was investigated for **provenance and context**, not approved for augmenting training data.

## Experiment log and findings

| Experiment | Task | Recorded outcome |
|---|---|---|
| 01 | Source-file inventory | 79 original documents and 18 XML files identified |
| 02 | Priority requirement-to-document matching | 11 priority records matched to source documents for projects 52, 80, and 86 |
| 03 | Verify source context | Context found for all 11 priority records |
| 05 | Extract original documents | All 79 documents extracted successfully |
| 06 | Candidate requirement extraction | 8,462 **unverified** candidates found |
| 07 | Exact / normalized matching | 190 exact and 23 normalized matches reported; 8,249 candidates unmatched; 276 PROMISE+ records involved |
| 08A | Expanded candidate linkage | 1,728 candidate pairs, covering 1,198 PURE candidates and 1,574 PROMISE+ records |
| 08B | Fuzzy matching | 7,264 fuzzy candidates; 21,792 pairs; 257 pairs at similarity ≥0.80 and 106 at ≥0.90; mean similarity 0.3361 |
| 08E | Document mapping / leakage review | Five ambiguous PURE document mappings found, all classified under the Train partition in the reviewed audit |
| 08F | Fold and source-overlap audit | 79 documents: 39 unresolved provenance, 36 single-fold candidates, 4 original-test-overlap candidates, 0 confirmed multi-fold candidates |
| 08G | Final document eligibility policy | **Zero documents approved** for use as additional training sources |

**Documentation limitation:** Exact results for intermediate steps not listed above have not been independently verified for this README. Refer to the scripts and output files in the local Phase 7 directory for complete experiment-specific detail. Matching counts reflect each experiment's matching procedure and are not interchangeable measures of verified provenance.

### Priority sources investigated

- **Project 52:** `1995 - gemini.pdf`
- **Project 80:** `2009 - library.pdf` / `.doc`
- **Project 86:** `2010 - fishing.pdf`

The final eligibility review also treated sources associated with Gemini, E-store, Library, and Fishing as blocked from unreviewed training reuse because of held-out fold relationships.

### Ambiguous mappings

Five documents required special attention because they mapped to multiple project IDs:

| PURE document | Mapped project IDs |
|---|---|
| `2002 - evla back.pdf` | 43, 58, 59 |
| `2002 - evla corr.pdf` | 43, 58, 59 |
| `2007 - e-store.doc` | 34, 46, 67 |
| `2009 - peazip.pdf` | 81, 90 |
| `2010 - split merge.pdf` | 81, 90 |

These mappings require document-level verification; an apparent match alone does not establish safe training eligibility.

## Final source-eligibility decision

Experiment 08G recorded **237 document-by-fold review rows**:

| Decision status | Rows |
|---|---:|
| `BLOCKED_HELDOUT_SOURCE` | 36 |
| `BLOCKED_TEST_SOURCE` | 12 |
| `BLOCKED_UNRESOLVED` | 117 |
| `REVIEW_REQUIRED` | 72 |
| **Approved for model training** | **0 documents** |

The 237 values represent **document-fold decisions**, not 237 distinct source documents. No confirmed multi-fold source link was found in the 08F audit, but this does **not** establish that all documents are free of leakage risk, nor does it prove leakage in previously trained models.

## Practical conclusion

PURE documents can be useful to understand source context, investigate annotation conventions, and study provenance. **Do not automatically add PURE-derived candidates to training, validation, or test data.** The Phase 7 audit did not approve any document for training augmentation. A future reuse plan requires independently verified project provenance, source isolation, annotation rights/quality review, and a new leakage-safe protocol.

## Reproducibility and organization

Use the existing Phase 7 folder as the source of truth for numbered scripts and generated outputs. Known audit artifacts include:

```text
promise_pure_analysis_phase_7/
└── outputs/
    └── tables/
        └── 08e_document_leakage_audit.csv
```

Other scripts and artifact names should be checked against the actual local directory rather than inferred from experiment numbers. Keep logs, matching tables, provenance decisions, and manual-review notes with their corresponding experiment.

## Phase status

**Phase 7 is archived as a provenance investigation.** It did not retrain models, relabel PROMISE+ requirements, or authorize additional PURE training documents. Future research should refer to these findings as exploratory source-quality evidence.
