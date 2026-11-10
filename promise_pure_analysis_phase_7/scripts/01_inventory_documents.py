
"""Phase 7 Experiment 01: PURE document inventory."""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import csv

ROOT = Path(__file__).resolve().parents[1]
PURE = ROOT.parent / "dataset" / "Pure"

DOCUMENTS = PURE / "req"
XML = PURE / "XMLZIPFile"

OUTPUT = ROOT / "outputs"
TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

EXTENSIONS = {".pdf", ".doc", ".htm", ".html", ".rtf"}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not DOCUMENTS.is_dir() or not XML.is_dir():
        raise FileNotFoundError("PURE directories not found")

    documents = sorted(
        p for p in DOCUMENTS.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    xml_files = sorted(XML.rglob("*.xml"))

    rows = []

    for file in documents:
        rows.append({
            "DocumentName": file.name,
            "Format": file.suffix.lower().lstrip("."),
            "FileSizeBytes": file.stat().st_size,
            "RelativePath": str(file.relative_to(PURE)),
            "SHA256": sha256_file(file),
        })

    inventory_file = TABLES / "01_pure_document_inventory.csv"

    with inventory_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "DocumentName",
                "Format",
                "FileSizeBytes",
                "RelativePath",
                "SHA256",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["Format"] for row in rows)

    summary = {
        "experiment": 1,
        "dataset": "PURE",
        "original_documents": len(documents),
        "document_formats": dict(sorted(counts.items())),
        "xml_files": len(xml_files),
        "inventory_created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "original_files_modified": False,
    }

    report_file = REPORTS / "01_pure_inventory_summary.json"

    report_file.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPHASE 7 EXPERIMENT 01")
    print("PURE DOCUMENT INVENTORY")
    print("=" * 55)

    print("Original documents:", len(documents))

    for extension, count in sorted(counts.items()):
        print(f"{extension.upper()}: {count}")

    print("XML documents:", len(xml_files))

    if len(documents) != 79:
        print("WARNING: Expected 79 original documents")

    print("\nSaved:", inventory_file)
    print("Saved:", report_file)

    print("\nEXPERIMENT 01 COMPLETED")


if __name__ == "__main__":
    main()
