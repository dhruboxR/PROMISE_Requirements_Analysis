
"""Diagnose DeBERTa-v3 numerical precision.

Compares FP32 and BF16 forward passes.
No optimizer steps or dataset evaluation.
"""

import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_NAME = "microsoft/deberta-v3-base"

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda")

texts = [
    "The system shall allow users to reset passwords.",
    "The application must respond within two seconds.",
    "The interface shall support multiple languages.",
    "The system must encrypt stored customer information.",
]

labels = torch.tensor(
    [0, 1, 2, 3],
    dtype=torch.long,
    device=device,
)

print("\nDEBERTA PRECISION DIAGNOSTIC")
print("=" * 55)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=12,
    dtype=torch.float32,
).to(device)

model.eval()

print("GPU:", torch.cuda.get_device_name(0))
print("Model parameter dtype:", next(model.parameters()).dtype)
print("BF16 supported:", torch.cuda.is_bf16_supported())

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=96,
    return_tensors="pt",
).to(device)


def evaluate_precision(name, use_bf16):
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            outputs = model(**inputs)

        logits = outputs.logits.float()

        # Calculate cross-entropy explicitly in FP32.
        loss = F.cross_entropy(logits, labels)

    print(f"\n{name}")
    print("Logits dtype:", outputs.logits.dtype)
    print("Logits finite:", torch.isfinite(logits).all().item())
    print("Loss:", float(loss))
    print("Loss finite:", torch.isfinite(loss).item())
    print("Logit minimum:", float(logits.min()))
    print("Logit maximum:", float(logits.max()))

    return logits


fp32 = evaluate_precision("FP32", False)
bf16 = evaluate_precision("BF16 autocast", True)

difference = (fp32 - bf16).abs()

print("\nCOMPARISON")
print("Maximum absolute logit difference:", float(difference.max()))
print("Mean absolute logit difference:", float(difference.mean()))
print("\nDiagnostic completed.")
