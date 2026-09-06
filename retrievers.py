"""Retrieval strategies behind one signature.

Every retriever takes a question and returns [(score, chunk), ...] exactly like
store.search, so eval.py can swap between them without knowing which is which.

The score means different things per retriever - cosine for dense and HyDE, a
reciprocal-rank-fusion score for hybrid - so it ranks within a retriever but
does not compare across them. eval.py records top1_dense separately for that.
"""

import json

import numpy as np

import bm25 as bm25_module
import llm
from config import (
    GROQ_MODEL,
    HYBRID_WEIGHT,
    HYDE_CACHE_PATH,
    HYDE_PROMPT,
    HYDE_RUNTIME_CACHE_PATH,
    RRF_K,
)
from embedder import embed_passage, embed_query

_bm25: dict[str, bm25_module.BM25] = {}
_caches: dict[str, dict] | None = None


def dense(question: str, store, k: int):
    """The milestone 1 path, unchanged."""
    return store.search(embed_query(question), k)


def hybrid(question: str, store, k: int):
    """Dense and BM25 rankings fused by RRF.

    Ranks are fused rather than scores because cosine sits around 0.6-0.9 while
    BM25 is unbounded; normalising scores per query would make every query's top
    hit 1.0 and destroy comparability across questions.
    """
    dense_ranks = _rrf(store.dense_scores(embed_query(question)))
    keyword_ranks = _rrf(_get_bm25(store).scores(question))

    fused = (1 - HYBRID_WEIGHT) * dense_ranks + HYBRID_WEIGHT * keyword_ranks
    top = np.argsort(fused)[-k:][::-1]
    return [(float(fused[i]), store.chunks[i]) for i in top]


def hyde(question: str, store, k: int):
    """Search with a hypothetical answer instead of the question.

    The generated passage is embedded as a document, with no query prefix: it
    stands in for the passage being looked for, so there is no question-shaped
    mismatch to correct.
    """
    return store.search(embed_passage(hypothetical(question)), k)


def _rrf(scores: np.ndarray) -> np.ndarray:
    """1 / (RRF_K + rank) for every document, rank 1 being the best score."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype="float64")
    ranks[order] = np.arange(1, len(scores) + 1)
    return 1.0 / (RRF_K + ranks)


def _get_bm25(store) -> bm25_module.BM25:
    """Built once per store and reused.

    Keyed by store rather than cached in a single global: with more than one
    document indexed, a global would score every query against whichever
    document happened to be seen first.
    """
    if store.key not in _bm25:
        _bm25[store.key] = bm25_module.BM25([c["text"] for c in store.chunks])
    return _bm25[store.key]


def _load_caches() -> tuple[dict, dict]:
    """(benchmark, runtime), each invalidated when the model or prompt changes.

    Without a cache the benchmark is not reproducible: generation varies between
    runs even at temperature 0, so the same retriever would score differently
    each time. The split keeps that guarantee while stopping live questions from
    accumulating in the committed file.
    """
    global _caches
    if _caches is None:
        _caches = {
            "benchmark": _read(HYDE_CACHE_PATH),
            "runtime": _read(HYDE_RUNTIME_CACHE_PATH),
        }
    return _caches["benchmark"], _caches["runtime"]


def _read(path) -> dict:
    empty = {"model": GROQ_MODEL, "prompt": HYDE_PROMPT, "passages": {}}
    if not path.exists():
        return empty
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("model") == GROQ_MODEL and saved.get("prompt") == HYDE_PROMPT:
        return saved
    print(f"{path.name}: model or prompt changed, regenerating")
    return empty


def hypothetical(question: str) -> str:
    benchmark, runtime = _load_caches()

    # Benchmark first: an eval question must always resolve to the committed
    # passage, whatever has since been asked through the API.
    if question in benchmark["passages"]:
        return benchmark["passages"][question]
    if question in runtime["passages"]:
        return runtime["passages"][question]

    passage = llm.generate(HYDE_PROMPT.format(question=question))
    runtime["passages"][question] = passage
    HYDE_RUNTIME_CACHE_PATH.write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return passage


RETRIEVERS = {"dense": dense, "hybrid": hybrid, "hyde": hyde}
