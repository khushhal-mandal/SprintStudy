import re

import pdfplumber

from config import MIN_CHARS_PER_PAGE


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers.

    Pages with no extractable text are dropped, but if almost the whole
    document is empty we raise instead of silently indexing nothing.
    """
    pages: list[tuple[int, str]] = []
    total = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            text = _clean(page.extract_text() or "")
            if len(text) >= MIN_CHARS_PER_PAGE:
                pages.append((i, text))

    if not pages:
        raise ValueError(
            f"No extractable text found in {pdf_path}. "
            "This looks like a scanned PDF - only text-based PDFs are supported."
        )

    if len(pages) < total / 2:
        print(
            f"  warning: only {len(pages)} of {total} pages had usable text - "
            "the PDF may be partly scanned"
        )

    return pages


def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")          # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank-line runs
    return text.strip()
