# PROMISE+ Phase 3 Baseline Comparison

## Experimental Setup

- Task: 12-class requirements classification
- Training records: 2574
- Validation records: 553
- Test records: reserved for final evaluation
- Primary selection metric: macro-F1

## Model Comparison

 Rank                            Model  Partition  accuracy  balanced_accuracy  macro_precision  macro_recall  macro_f1  weighted_f1
    1              TF-IDF + Linear SVM validation  0.607595           0.416909         0.345464      0.416909  0.356970     0.607805
    2     TF-IDF + Logistic Regression validation  0.564195           0.412078         0.342333      0.412078  0.343544     0.578149
    3           TF-IDF + Random Forest validation  0.520796           0.306055         0.270503      0.306055  0.263966     0.518475
    4 TF-IDF + Multinomial Naive Bayes validation  0.616637           0.086667         0.134662      0.086667  0.069938     0.472181
    5                   Majority Class validation  0.614828           0.083333         0.051236      0.083333  0.063457     0.468178

## Best Baseline

- Model: TF-IDF + Linear SVM
- Macro-F1: 0.3570
- Accuracy: 0.6076

## Linear SVM Per-Class F1

Model  Linear SVM  Logistic Regression
Class                                 
F          0.7634               0.7199
A          0.5625               0.5926
L          0.1600               0.1250
LF         0.3333               0.2143
MN         0.2000               0.1818
O          0.5070               0.4810
PE         0.5200               0.6000
SC         0.1905               0.1905
SE         0.5243               0.5000
US         0.3797               0.3636
FT         0.1429               0.1538
PO         0.0000               0.0000

## Classes Requiring Review

       Precision  Recall      F1  Support
Class                                    
PO        0.0000  0.0000  0.0000     24.0
FT        0.2500  0.1000  0.1429     10.0
L         0.1250  0.2222  0.1600      9.0
SC        0.4000  0.1250  0.1905     16.0
MN        0.1818  0.2222  0.2000      9.0

## Evaluation Protocol

All comparisons use the frozen validation partition.
The test partition was not accessed.
Results are preliminary until final model selection.