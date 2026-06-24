"""Extract text from research PDFs in docs/papers/ -> docs/paper_texts/."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import fitz  # PyMuPDF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "docs" / "papers"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "paper_texts"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(glob.glob(str(PDF_DIR / "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}")
        return

    for pdf_path in pdf_paths:
        fname = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'=' * 80}")
        print(f"Extracting: {fname}")
        print(f"{'=' * 80}")

        doc = fitz.open(pdf_path)
        full_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")

        combined = "\n".join(full_text)
        out_path = OUTPUT_DIR / f"{fname}.txt"
        out_path.write_text(combined, encoding="utf-8")

        print(f"  Pages: {len(doc)}, Characters: {len(combined)}")
        print(f"  Saved to: {out_path}")
        doc.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
