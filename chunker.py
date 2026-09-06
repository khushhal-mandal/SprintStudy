import re

from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(
    pages: list[tuple[int, str]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split pages into chunks of roughly `size` characters.

    Chunks never span a page boundary, so every chunk has one exact page
    number to cite. Splits happen at paragraph breaks where possible,
    then sentence breaks, then raw characters.
    """
    chunks: list[dict] = []

    for page_no, text in pages:
        for body in _pack(_split_units(text, size), size, overlap):
            chunks.append(
                {
                    "id": len(chunks),
                    "page": page_no,
                    "text": body,
                }
            )

    return chunks


def _split_units(text: str, size: int) -> list[str]:
    """Break text into pieces that each fit within `size`."""
    units: list[str] = []

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
            continue

        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= size:
                units.append(sent)
            else:
                units.extend(sent[i : i + size] for i in range(0, len(sent), size))

    return units


def _pack(units: list[str], size: int, overlap: int) -> list[str]:
    """Greedily fill chunks with units, prefixing each with the tail of the last."""
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        body = " ".join(buf)
        if out and overlap:
            body = _tail(out[-1], overlap) + " " + body
        out.append(body)
        buf, buf_len = [], 0

    for unit in units:
        if buf and buf_len + len(unit) + 1 > size:
            flush()
        buf.append(unit)
        buf_len += len(unit) + 1

    flush()
    return out


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, cut at a word boundary."""
    if len(text) <= n:
        return text
    piece = text[-n:]
    space = piece.find(" ")
    return piece[space + 1 :] if space != -1 else piece
