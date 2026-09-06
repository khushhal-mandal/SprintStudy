"""Print the book's section list with the PDF page numbers used by the index.

The table of contents lists *printed* page numbers, which are offset from the
PDF page numbers stored in chunks.json. eval.json's expected_pages must use the
PDF numbers, so this resolves the offset and prints both.
"""

import json
import re
import sys
from collections import Counter

import pdfplumber

import pdf_parser
import store
from config import FIRST_CONTENT_PAGE

DEFAULT_PDF = "book.pdf"
MAX_OFFSET = 30

PART_RE = re.compile(r"^([IVX]+)\s+(.+?)\s+(\d+)$")
CHAPTER_RE = re.compile(r"^(\d+)\s+(.+?)\s+(\d+)$")
SECTION_RE = re.compile(r"^(\d+\.\d+)\s+(.+?)\s*(?:\.\s*){3,}(\d+)$")


def load_pages(pdf_path: str) -> dict[int, str]:
    """Front matter from the PDF, body pages from the index.

    The contents pages are no longer ingested - they are dot leaders rather
    than content - so the table of contents has to be read from the PDF
    directly. Body text still comes from chunks.json, because that is what the
    index actually holds and what the page numbers must agree with.
    """
    pages: dict[int, str] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages[: FIRST_CONTENT_PAGE - 1], start=1):
            pages[i] = pdf_parser._clean(page.extract_text() or "")

    by_page: dict[int, list[str]] = {}
    for c in json.loads(store.CHUNKS_PATH.read_text(encoding="utf-8")):
        by_page.setdefault(c["page"], []).append(c["text"])
    for page_no, texts in by_page.items():
        pages[page_no] = _merge(texts)

    return pages


def _merge(texts: list[str]) -> str:
    """Join chunks, collapsing the overlap each one carries from the previous."""
    out = texts[0]
    for nxt in texts[1:]:
        for n in range(min(len(out), len(nxt)), 0, -1):
            if out.endswith(nxt[:n]):
                out += nxt[n:]
                break
        else:
            out += nxt
    return out


def parse_toc(pages: dict[int, str]) -> list[dict]:
    """Pull (kind, number, title, printed_page) out of the contents listing."""
    entries: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for p in range(1, FIRST_CONTENT_PAGE):
        if p not in pages:
            continue
        for line in pages[p].splitlines():
            line = line.strip()
            for kind, rx in (("section", SECTION_RE), ("chapter", CHAPTER_RE), ("part", PART_RE)):
                m = rx.match(line)
                if not m:
                    continue
                number, title, printed = m.group(1), m.group(2).strip(), int(m.group(3))
                if (number, title) in seen:
                    break
                seen.add((number, title))
                entries.append(
                    {"kind": kind, "number": number, "title": title, "printed": printed}
                )
                break

    return entries


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def resolve_offset(entries: list[dict], pages: dict[int, str]) -> tuple[int, int, int]:
    """Find the constant printed->PDF page shift by locating titles in the body.

    Returns (offset, matched, checkable).
    """
    checkable = [e for e in entries if e["kind"] in ("chapter", "section")]
    votes: Counter[int] = Counter()

    for e in checkable:
        needle = _norm(e["title"])
        for off in range(MAX_OFFSET + 1):
            page = pages.get(e["printed"] + off)
            if page and needle in _norm(page):
                votes[off] += 1

    if not votes:
        return 0, 0, len(checkable)

    offset, matched = votes.most_common(1)[0]
    return offset, matched, len(checkable)


def main(pdf_path: str) -> None:
    pages = load_pages(pdf_path)
    entries = parse_toc(pages)
    if not entries:
        raise SystemExit("No table of contents found in the front matter.")

    offset, matched, checkable = resolve_offset(entries, pages)
    for e in entries:
        e["pdf"] = e["printed"] + offset

    # Chunks per entry, spanning to the next entry of the same kind: a chapter
    # row covers the whole chapter, a section row just that section.
    counts = Counter(c["page"] for c in json.loads(store.CHUNKS_PATH.read_text(encoding="utf-8")))
    end = max(pages) + 1
    for e in entries:
        if e["kind"] == "part":
            continue
        nxt = next(
            (o["pdf"] for o in entries if o["kind"] == e["kind"] and o["pdf"] > e["pdf"]), end
        )
        e["chunks"] = sum(n for p, n in counts.items() if e["pdf"] <= p < nxt)

    total_chunks = sum(counts.values())
    print(f"\n{total_chunks} chunks over {len(counts)} pages")
    print(f"printed page + {offset} = PDF page  (verified on {matched}/{checkable} headings)")
    print("\nUse the PDF column for expected_pages in eval.json.\n")
    print(f"  {'PDF':>5}  {'print':>5}  {'chunks':>6}  section")
    print(f"  {'-' * 5}  {'-' * 5}  {'-' * 6}  {'-' * 46}")

    for e in entries:
        if e["kind"] == "part":
            print(f"\n  {'':>5}  {'':>5}  {'':>6}  {e['number']}. {e['title'].upper()}")
            continue
        indent = "  " if e["kind"] == "section" else ""
        print(
            f"  {e['pdf']:>5}  {e['printed']:>5}  {e['chunks']:>6}  "
            f"{indent}{e['number']:<5} {e['title']}"
        )
    print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF)
