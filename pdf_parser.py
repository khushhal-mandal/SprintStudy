import re

import pdfplumber

from config import (
    CID_MAP,
    FIRST_CONTENT_PAGE,
    MIN_CHARS_PER_PAGE,
    NEGATION_PAIR,
)

CID_RE = re.compile(r"\(cid:(\d+)\)")


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] with 1-based page numbers.

    Pages with no extractable text are dropped, but if almost the whole
    document is empty we raise instead of silently indexing nothing.
    """
    pages: list[tuple[int, str]] = []
    total = 0
    unmapped: set[int] = set()

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # Front matter carries no teachable content and the contents pages
            # are mostly dot leaders. Page numbers are unchanged by skipping,
            # because i counts PDF pages rather than kept ones.
            if i < FIRST_CONTENT_PAGE:
                continue
            total += 1
            text = _clean(page.extract_text() or "", unmapped)
            if len(text) >= MIN_CHARS_PER_PAGE:
                pages.append((i, text))

    if unmapped:
        print(f"  warning: dropped unmapped glyph codes {sorted(unmapped)}")

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


def _clean(text: str, unmapped: set[int] | None = None) -> str:
    text = _repair_glyphs(text, unmapped)
    text = text.replace("\u00ad", "")          # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)        # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse blank-line runs
    return text.strip()


def _repair_glyphs(text: str, unmapped: set[int] | None = None) -> str:
    """Turn pdfminer's unresolved "(cid:N)" markers back into characters.

    The book's math fonts use built-in Type 1 encodings pdfminer cannot map, so
    every square root, floor bracket and summation arrives as a "(cid:N)"
    marker. Left alone they corrupt the chunk, the embedding and anything
    quoting it.
    """
    # The negation slash is drawn over the "=" that follows it, so the pair has
    # to be replaced together or "x != 0" becomes "x = 0" - an inverted meaning.
    text = text.replace(*NEGATION_PAIR)

    def replace(match: re.Match) -> str:
        code = int(match.group(1))
        if code not in CID_MAP and unmapped is not None:
            unmapped.add(code)
        return CID_MAP.get(code, "")

    return CID_RE.sub(replace, text)
