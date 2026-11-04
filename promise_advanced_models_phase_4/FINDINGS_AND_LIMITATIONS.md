# Phase 4: Findings and Limitations

## Main findings

### Finding 1: Classical models remain competitive

Word and character TF-IDF Logistic Regression achieved the highest mean macro-F1 among the evaluated group-aware configurations.

This demonstrates that pretrained transformer architectures do not necessarily provide better performance on a small, highly imbalanced requirements dataset.

### Finding 2: Class weighting changes model behavior

Applying class weighting improved DistilBERT's original validation macro-F1 from 0.1887 to 0.2735.

Extending weighted training to six epochs increased macro-F1 to 0.3174, with a corresponding reduction in overall accuracy relative to the three-epoch weighted model.

For MiniLM embeddings, removing balanced class weights greatly improved accuracy but reduced balanced accuracy.

### Finding 3: Minority classes remain difficult

PO, SC, L, MN, FT, and LF have particularly high shared-error rates under group-aware evaluation.

These categories require better analysis of class definitions, training representation, and generalization between projects.

### Finding 4: Cross-project lexical differences are substantial

TF-IDF similarity analysis found low same-class similarity for several difficult validation requirements.

This supports further investigation of semantic representations, but lexical similarity is not sufficient to establish semantic differences or annotation errors.

### Finding 5: Simple model ensembles were unsuccessful

Three-model majority voting performed worse than Logistic Regression alone.

Model complementarity exists, but selecting or combining predictions without a validated decision rule did not establish better performance.

## Methodological limitations

The dataset contains severe class imbalance and very few training requirements for certain classes.

The group-aware folds differ in project composition, class distribution, and difficulty.

Only three cross-validation folds were evaluated.

Several modeling choices were influenced by the original validation partition and subsequent development experiments. Reusing those data to choose models limits the independence of later performance estimates.

The Phase 3 test set was examined previously and is no longer fully untouched for future model-selection decisions.

The original annotation definitions and contradictory annotation groups require documentation and careful review. Labels should not be changed based solely on model predictions.

## Recommendations for Phase 5

Develop a hybrid lexical-semantic feature representation.

Investigate class-weighting strategies without sacrificing majority-class performance.

Evaluate minority-class behavior and project-specific performance explicitly.

Use a fixed, documented development protocol for comparisons.

Seek a new independent external test dataset, or design an appropriately separated evaluation procedure, before making final generalization claims.

## Phase 4 conclusion

Phase 4 provides a reproducible baseline and an evidence-based diagnosis of the current classification weaknesses.

The results justify further research into models designed specifically for cross-project generalization and uneven requirement-class distributions.
