"""Okapi BM25 over the chunk texts.

Written out rather than pulled in as a dependency - the keyword half of hybrid
retrieval is exactly the part of this project worth being able to explain.

    idf(t)   = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
    score(D) = sum over query terms of
                 idf(t) * f(t,D) * (k1 + 1)
                 -------------------------------------------
                 f(t,D) + k1 * (1 - b + b * |D| / avgdl)

Two deliberate simplifications, both visible rather than hidden:

- No stemming, so "trees" does not match "tree". A stemmer would mean another
  dependency, and the cost belongs in the results write-up, not in a silent
  workaround.
- Query terms are deduplicated. The canonical formula sums over every query
  term occurrence, which lets a word repeated in a question double a
  document's score. For natural-language questions that is noise.
"""

import math
import re
from collections import Counter

import numpy as np

from config import BM25_B, BM25_K1

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, texts: list[str], k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b

        docs = [tokenize(t) for t in texts]
        self.n_docs = len(docs)
        self.lengths = np.array([len(d) for d in docs], dtype="float64")
        self.avgdl = float(self.lengths.mean()) if self.n_docs else 0.0

        # term -> (document indices, term frequencies), as parallel arrays so
        # scoring is one vectorised update per query term.
        raw: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, freq in Counter(doc).items():
                raw.setdefault(term, []).append((i, freq))

        self.postings = {
            term: (
                np.array([i for i, _ in p]),
                np.array([f for _, f in p], dtype="float64"),
            )
            for term, p in raw.items()
        }
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in raw.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document against the query. Shape (n_docs,)."""
        out = np.zeros(self.n_docs, dtype="float64")
        if not self.avgdl:
            return out

        # Length normalisation is per-document and query-independent.
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avgdl)

        for term in dict.fromkeys(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idx, freq = posting
            out[idx] += self.idf[term] * freq * (self.k1 + 1) / (freq + norm[idx])

        return out
