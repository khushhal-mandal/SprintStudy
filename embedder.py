"""Embedding, via fastembed's ONNX runtime rather than sentence-transformers.

The switch was made for deployment, not for quality: sentence-transformers pulls
torch, which put the API image at 1.58GB and over any small instance's memory
budget. fastembed runs the same BAAI/bge-small-en-v1.5 weights through
onnxruntime with no torch at all.

The two are not bit-identical - ONNX and PyTorch differ by up to 8.8e-04 per
dimension, about 7,400x float32 eps, at a cosine of 0.999998. That is small but
it is not noise, and it is enough to reorder near-ties. The index is therefore
embedded by whichever library serves queries, never a mix of the two, and the
eval numbers were re-measured after the swap rather than assumed to carry over.

One trap worth naming: fastembed exposes query_embed(), and for this model it
does NOT apply the bge query instruction - it is identical to embed(). Relying
on it would have dropped QUERY_PREFIX silently and degraded every query.
"""

import numpy as np
from fastembed import TextEmbedding

from config import EMBED_BATCH, EMBED_CACHE_DIR, EMBED_MODEL, QUERY_PREFIX

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Loaded once and reused - the first call downloads the ONNX weights."""
    global _model
    if _model is None:
        print(f"loading {EMBED_MODEL} ...")
        _model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=EMBED_CACHE_DIR)
    return _model


def embed_chunks(texts: list[str]) -> np.ndarray:
    """(n, 384) float32, L2-normalised so dot product == cosine similarity.

    fastembed normalises inside the graph, so there is nothing to add here -
    and nothing to add in the search path either.
    """
    vectors = list(_get_model().embed(texts, batch_size=EMBED_BATCH))
    return np.array(vectors, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    """(384,) float32. Uses the bge query instruction - chunks must not have it."""
    return _embed_one(QUERY_PREFIX + text)


def embed_passage(text: str) -> np.ndarray:
    """(384,) float32, no query prefix.

    For text that plays the role of a document rather than a question - a HyDE
    hypothetical answer. QUERY_PREFIX corrects a question-shaped vs
    passage-shaped mismatch, and a passage has no mismatch to correct.
    """
    return _embed_one(text)


def _embed_one(text: str) -> np.ndarray:
    return np.array(next(iter(_get_model().embed([text]))), dtype="float32")
