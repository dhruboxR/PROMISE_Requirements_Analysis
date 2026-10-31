
"""Validate, repair, and audit the PROMISE+ ARFF dataset."""

from pathlib import Path
from collections import Counter
import csv
import re
import arff

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT.parent / "dataset" / "Promise+.arff"

VALIDATED_DIR = ROOT / "outputs" / "validated"
REPORT_DIR = ROOT / "outputs" / "reports"

VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = VALIDATED_DIR / "Promise+_repaired.arff"
AUDIT = REPORT_DIR / "arff_repair_audit.csv"

# --------------------------------------------------
# 2. ARFF class labels
# --------------------------------------------------

LABELS = [
    "F", "A", "L", "LF", "MN", "O",
    "PE", "SC", "SE", "US", "FT", "PO"
]

label_pattern = "|".join(
    re.escape(label)
    for label in sorted(LABELS, key=len, reverse=True)
)

# Detect a missing comma before the class label.
missing_comma_pattern = re.compile(
    rf"^(\s*[^,\n]+,\s*'.*')\s*({label_pattern})\s*$"
)

# --------------------------------------------------
# 3. Legacy punctuation replacement map
# --------------------------------------------------

# Inferred Windows-1252 punctuation representations.
# These replacements should be documented as
# inferred corrections, not verified source characters.

PUNCTUATION_MAP = {
    r"\91": "‘",
    r"\92": "’",
    r"\93": "“",
    r"\94": "”",
    r"\95": "•",
    r"\96": "–",
    r"\97": "—",
}

# --------------------------------------------------
# 4. Read the original dataset
# --------------------------------------------------

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Original dataset not found: {SOURCE}"
    )

lines = SOURCE.read_text(
    encoding="utf-8"
).splitlines(keepends=True)

repaired_lines = []
audit_rows = []

repair_counts = Counter()
punctuation_counts = Counter()

in_data = False
candidate_records = 0
changed_records = set()

# --------------------------------------------------
# 5. Process and audit every record
# --------------------------------------------------

for line_number, line in enumerate(lines, start=1):

    stripped = line.strip()

    # Detect ARFF data section.
    if stripped.lower() == "@data":
        in_data = True
        repaired_lines.append(line)
        continue

    # Preserve headers, comments, and empty lines.
    if (
        not in_data
        or not stripped
        or stripped.startswith("%")
    ):
        repaired_lines.append(line)
        continue

    candidate_records += 1

    original = line.rstrip("\r\n")
    newline = line[len(original):]

    corrected = original

    # ----------------------------------------------
    # A. Repair missing commas
    # ----------------------------------------------

    match = missing_comma_pattern.fullmatch(
        corrected.strip()
    )

    if match:
        previous = corrected

        corrected = (
            f"{match.group(1)},{match.group(2)}"
        )

        audit_rows.append({
            "line_number": line_number,
            "repair_type": "missing_comma",
            "original_value": previous,
            "corrected_value": corrected,
        })

        repair_counts["missing_comma"] += 1
        changed_records.add(line_number)

    # ----------------------------------------------
    # B. Repair punctuation sequences
    # ----------------------------------------------

    for sequence, replacement in PUNCTUATION_MAP.items():

        count = corrected.count(sequence)

        if count == 0:
            continue

        previous = corrected

        corrected = corrected.replace(
            sequence, replacement
        )

        audit_rows.append({
            "line_number": line_number,
            "repair_type": "punctuation_replacement",
            "original_value": previous,
            "corrected_value": corrected,
        })

        punctuation_counts[sequence] += count
        repair_counts["punctuation_replacement"] += count

        changed_records.add(line_number)

    repaired_lines.append(corrected + newline)

# --------------------------------------------------
# 6. Check for unresolved legacy sequences
# --------------------------------------------------

repaired_text = "".join(repaired_lines)

remaining_escapes = re.findall(
    r"\\[0-9]{2,3}",
    repaired_text
)

if remaining_escapes:
    print(
        "Warning: Remaining numeric escape sequences:",
        Counter(remaining_escapes)
    )

# --------------------------------------------------
# 7. Validate before saving
# --------------------------------------------------

try:
    dataset = arff.loads(repaired_text)

except Exception as error:
    print("\nARFF validation failed.")
    print(f"Error: {type(error).__name__}: {error}")
    raise SystemExit(1)

parsed_records = len(dataset["data"])

if parsed_records != candidate_records:
    raise ValueError(
        f"Record count mismatch: "
        f"{candidate_records} candidates, "
        f"{parsed_records} parsed."
    )

# --------------------------------------------------
# 8. Save repaired dataset and audit CSV
# --------------------------------------------------

TARGET.write_text(
    repaired_text,
    encoding="utf-8"
)

with AUDIT.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "line_number",
            "repair_type",
            "original_value",
            "corrected_value",
        ]
    )

    writer.writeheader()
    writer.writerows(audit_rows)

# --------------------------------------------------
# 9. Display validation report
# --------------------------------------------------

print("\nPROMISE+ ARFF VALIDATION")
print("=" * 50)

print(f"Source: {SOURCE}")
print(f"Candidate records: {candidate_records}")
print(f"Parsed records: {parsed_records}")

print("\nRepair summary")
print("-" * 50)

print(
    "Missing-comma repairs:",
    repair_counts["missing_comma"]
)

print(
    "Punctuation replacements:",
    repair_counts["punctuation_replacement"]
)

print(
    "Unique records modified:",
    len(changed_records)
)

print("\nPunctuation details:")

for sequence in PUNCTUATION_MAP:
    print(
        f"{sequence}: {punctuation_counts[sequence]}"
    )

print("\nOutput files:")
print(f"Validated dataset: {TARGET}")
print(f"Audit report: {AUDIT}")

print("\nARFF VALIDATION SUCCESSFUL")
print(f"Attributes: {len(dataset['attributes'])}")
print(f"Total records: {parsed_records}")
