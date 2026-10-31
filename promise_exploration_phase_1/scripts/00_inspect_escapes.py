
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "validated" / "Promise+_repaired.arff"

text = SOURCE.read_text(encoding="utf-8")

# Identify backslashes followed by an invalid octal digit.
pattern = re.compile(r"\\[89][0-9]?|\\[0-7]*[89][0-9]*")

matches = []

for line_number, line in enumerate(text.splitlines(), start=1):
    for match in pattern.finditer(line):
        matches.append((line_number, match.group(), line))

print("Potential invalid escape sequences:", len(matches))

for line_number, sequence, line in matches[:20]:
    print(f"\nLine: {line_number}")
    print(f"Sequence: {sequence}")
    print(f"Content: {line[:300]}")
