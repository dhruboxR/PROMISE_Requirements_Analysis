from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT.parent / "promise_improved_model_phase_5"

hybrid_path = (
    PHASE5 / "outputs/tables/06_class_weight_oof_predictions.csv"
)

deberta_path = (
    ROOT / "outputs/tables/03_deberta_balanced_fold1_predictions.csv"
)

hybrid = pd.read_csv(
    hybrid_path,
    dtype={"RecordID": str, "ProjectID": str}
)

hybrid = hybrid[
    (hybrid["Configuration"] == "tempered")
    & (hybrid["Fold"] == 1)
].copy()

deberta = pd.read_csv(
    deberta_path,
    dtype={"RecordID": str, "ProjectID": str}
)

assert len(hybrid) == 968
assert len(deberta) == 968
assert hybrid["RecordID"].is_unique
assert deberta["RecordID"].is_unique

comparison = hybrid[
    ["RecordID", "ProjectID", "_class_", "PredictedLabel"]
].merge(
    deberta[
        ["RecordID", "ProjectID", "_class_", "PredictedLabel"]
    ],
    on="RecordID",
    validate="one_to_one",
    suffixes=("_Hybrid", "_DeBERTa")
)

assert len(comparison) == 968
assert (
    comparison["ProjectID_Hybrid"]
    == comparison["ProjectID_DeBERTa"]
).all()
assert (
    comparison["_class__Hybrid"]
    == comparison["_class__DeBERTa"]
).all()

actual = comparison["_class__Hybrid"]

hybrid_correct = comparison["PredictedLabel_Hybrid"] == actual
deberta_correct = comparison["PredictedLabel_DeBERTa"] == actual

results = {
    "Both correct": int((hybrid_correct & deberta_correct).sum()),
    "Only hybrid correct": int((hybrid_correct & ~deberta_correct).sum()),
    "Only DeBERTa correct": int((~hybrid_correct & deberta_correct).sum()),
    "Both incorrect": int((~hybrid_correct & ~deberta_correct).sum()),
}

print("\nPHASE 6 EXPERIMENT 04")
print("HYBRID VS BALANCED DEBERTA, FOLD 1")
print("=" * 55)

for name, count in results.items():
    print(f"{name}: {count}")

print("\nCorrect hybrid predictions:", int(hybrid_correct.sum()))
print("Correct DeBERTa predictions:", int(deberta_correct.sum()))

comparison["HybridCorrect"] = hybrid_correct
comparison["DeBERTaCorrect"] = deberta_correct

output = ROOT / "outputs/tables/04_hybrid_deberta_comparison.csv"
output.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(output, index=False)

print("\nSaved:", output)
