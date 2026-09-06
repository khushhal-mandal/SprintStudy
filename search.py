import sys

import store
from embedder import embed_query


def main(query: str, k: int = 3) -> None:
    chunks, vectors = store.load()
    hits = store.search(embed_query(query), chunks, vectors, k)

    print(f'\nquery: "{query}"  ({len(chunks)} chunks searched)\n')
    for rank, (score, chunk) in enumerate(hits, start=1):
        print(f"[{rank}] score {score:.3f}  page {chunk['page']}  chunk #{chunk['id']}")
        print(f"    {chunk['text'][:300]}...\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('usage: python search.py "your question" [k]')
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
