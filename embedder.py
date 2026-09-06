import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL, QUERY_PREFIX

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Loaded once and reused - the first call downloads ~130MB."""
    global _model
    if _model is None:
        print(f"loading {EMBED_MODEL} ...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed_chunks(texts: list[str]) -> np.ndarray:
    """(n, 384) float32, L2-normalised so dot product == cosine similarity."""
    return _get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def embed_query(text: str) -> np.ndarray:
    """(384,) float32. Uses the bge query instruction - chunks must not have it."""
    return _get_model().encode(
        QUERY_PREFIX + text,
        normalize_embeddings=True,
    ).astype("float32")


def embed_passage(text: str) -> np.ndarray:
    """(384,) float32, no query prefix.

    For text that plays the role of a document rather than a question - a HyDE
    hypothetical answer. QUERY_PREFIX corrects a question-shaped vs
    passage-shaped mismatch, and a passage has no mismatch to correct.
    """
    return _get_model().encode(text, normalize_embeddings=True).astype("float32")
