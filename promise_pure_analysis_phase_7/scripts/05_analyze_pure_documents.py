
"""
Phase 7 - Experiment 05
PURE Dataset-Wide Document Text Analysis

Processes 79 original PURE documents:
PDF, DOC, HTML, HTM, and RTF.

Outputs:
- outputs/tables/05_pure_document_statistics.csv
- outputs/reports/05_pure_analysis_summary.json
- outputs/extracted_text/*.txt

No PROMISE+ records or original documents are modified.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import csv
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "dataset" / "Pure" / "req"

OUTPUT = ROOT / "outputs"
TEXT_DIR = OUTPUT / "extracted_text"
TABLE_DIR = OUTPUT / "tables"
REPORT_DIR = OUTPUT / "reports"

SUPPORTED = {".pdf", ".doc", ".html", ".htm", ".rtf"}

FIELDS = [
    "DocumentName",
    "Format",
    "FileSizeBytes",
    "PDFPages",
    "CharacterCount",
    "WordCount",
    "LineCount",
    "ExtractionStatus",
    "ExtractionError",
    "ExtractedTextFile",
]


class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignore = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignore += 1
        elif tag in {
            "p", "div", "br", "li", "h1", "h2",
            "h3", "tr", "section", "article"
        }:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignore = max(0, self.ignore - 1)
        elif tag in {
            "p", "div", "li", "h1", "h2",
            "h3", "tr", "section", "article"
        }:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.ignore == 0:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


def execute(command):
    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode(
        "utf-8", errors="replace"
    )


def extract_text(path):
    extension = path.suffix.lower()

    if extension == ".pdf":
        return execute([
            "pdftotext", "-layout", str(path), "-"
        ])

    if extension == ".doc":
        return execute(["antiword", str(path)])

    if extension in {".html", ".htm"}:
        raw = path.read_text(
            encoding="utf-8", errors="replace"
        )
        parser = HTMLTextExtractor()
        parser.feed(raw)
        return parser.get_text()

    if extension == ".rtf":
        return execute([
            "unrtf", "--text", "--nopict", str(path)
        ])

    raise ValueError(f"Unsupported extension: {extension}")


def get_pdf_pages(path):
    if path.suffix.lower() != ".pdf":
        return None

    output = execute(["pdfinfo", str(path)])
    match = re.search(
        r"(?m)^Pages:\s*(\d+)",
        output,
    )

    return int(match.group(1)) if match else None


def clean_text(text):
    # Preserve paragraph boundaries, but remove excessive
    # whitespace within individual lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    return "\n".join(lines).strip()


def analyze_document(path):
    row = {
        "DocumentName": path.name,
        "Format": path.suffix.lower().lstrip("."),
        "FileSizeBytes": path.stat().st_size,
        "PDFPages": "",
        "CharacterCount": 0,
        "WordCount": 0,
        "LineCount": 0,
        "ExtractionStatus": "Failed",
        "ExtractionError": "",
        "ExtractedTextFile": "",
    }

    try:
        text = clean_text(extract_text(path))

        if not text:
            row["ExtractionStatus"] = "Empty"
            return row

        output_name = (
            f"{path.stem}__"
            f"{path.suffix.lower().lstrip('.')}.txt"
        )

        output_path = TEXT_DIR / output_name
        output_path.write_text(
            text, encoding="utf-8"
        )

        row["CharacterCount"] = len(text)
        row["WordCount"] = len(
            re.findall(r"\b[\w'-]+\b", text)
        )
        row["LineCount"] = len(text.splitlines())
        row["ExtractedTextFile"] = str(
            output_path.relative_to(ROOT)
        )
        row["ExtractionStatus"] = "Success"

        if path.suffix.lower() == ".pdf":
            row["PDFPages"] = get_pdf_pages(path)

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        UnicodeError,
        ValueError,
        OSError,
    ) as error:
        row["ExtractionError"] = str(error)[:300]

    return row


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)

    for directory in (
        TEXT_DIR, TABLE_DIR, REPORT_DIR
    ):
        directory.mkdir(
            parents=True, exist_ok=True
        )

    files = sorted(
        file for file in SOURCE.rglob("*")
        if file.is_file()
        and file.suffix.lower() in SUPPORTED
    )

    print("PHASE 7 EXPERIMENT 05")
    print("PURE DOCUMENT TEXT ANALYSIS")
    print("=" * 60)

    results = []

    for index, file in enumerate(files, start=1):
        row = analyze_document(file)
        results.append(row)

        print(
            f"[{index:02}/{len(files):02}] "
            f"{row['ExtractionStatus']:<8} "
            f"{row['WordCount']:>8} words | "
            f"{file.name}"
        )

    csv_path = (
        TABLE_DIR / "05_pure_document_statistics.csv"
    )

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=FIELDS
        )
        writer.writeheader()
        writer.writerows(results)

    successful = [
        row for row in results
        if row["ExtractionStatus"] == "Success"
    ]

    failed = [
        row for row in results
        if row["ExtractionStatus"] != "Success"
    ]

    total_words = sum(
        row["WordCount"] for row in successful
    )

    total_characters = sum(
        row["CharacterCount"] for row in successful
    )

    summary = {
        "experiment": "05",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "PURE",
        "total_documents": len(files),
        "format_counts": dict(
            sorted(
                Counter(
                    row["Format"] for row in results
                ).items()
            )
        ),
        "successful_extractions": len(successful),
        "failed_or_empty_extractions": len(failed),
        "total_extracted_words": total_words,
        "total_extracted_characters": total_characters,
        "mean_words_per_successful_document": (
            round(total_words / len(successful), 2)
            if successful else 0
        ),
        "longest_documents": [
            {
                "name": row["DocumentName"],
                "words": row["WordCount"]
            }
            for row in sorted(
                successful,
                key=lambda r: r["WordCount"],
                reverse=True
            )[:10]
        ],
        "extraction_issues": [
            {
                "name": row["DocumentName"],
                "status": row["ExtractionStatus"],
                "error": row["ExtractionError"],
            }
            for row in failed
        ],
        "requirement_count_validated": False,
        "classification_labels_assigned": False,
        "original_dataset_modified": False,
    }

    json_path = (
        REPORT_DIR / "05_pure_analysis_summary.json"
    )

    json_path.write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 05 SUMMARY")
    print("=" * 60)
    print("Documents processed:", len(files))
    print("Successful extractions:", len(successful))
    print("Failed/empty extractions:", len(failed))
    print("Total extracted words:", total_words)
    print("Total extracted characters:", total_characters)
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("Original dataset modified: No")


if __name__ == "__main__":
    main()

