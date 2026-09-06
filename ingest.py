import sys

import store
from chunker import chunk_pages
from embedder import embed_chunks
from pdf_parser import extract_pages


def main(pdf_path: str) -> None:
    pages = extract_pages(pdf_path)
    print(f"parsed {len(pages)} pages")

    chunks = chunk_pages(pages)
    lengths = [len(c["text"]) for c in chunks]
    print(
        f"built {len(chunks)} chunks "
        f"(min {min(lengths)}, avg {sum(lengths) // len(lengths)}, max {max(lengths)} chars)"
    )

    vectors = embed_chunks([c["text"] for c in chunks])
    print(f"embedded to {vectors.shape}")

    store.save(chunks, vectors)
    print(f"saved to {store.VECTORS_PATH.parent}/")

    print("\nfirst chunk preview:")
    print(f"  page {chunks[0]['page']}: {chunks[0]['text'][:200]}...")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python ingest.py <file.pdf>")
    main(sys.argv[1])
