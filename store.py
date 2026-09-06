import json

import numpy as np

from config import DATA_DIR

VECTORS_PATH = DATA_DIR / "vectors.npy"
CHUNKS_PATH = DATA_DIR / "chunks.json"


def save(chunks: list[dict], vectors: np.ndarray) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    np.save(VECTORS_PATH, vectors)
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")


def load() -> tuple[list[dict], np.ndarray]:
    if not VECTORS_PATH.exists():
        raise FileNotFoundError("No index found. Run: python ingest.py <file.pdf>")
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return chunks, np.load(VECTORS_PATH)


def search(query_vec: np.ndarray, chunks: list[dict], vectors: np.ndarray, k: int = 5):
    """Brute-force exact search. Both sides are normalised, so this is cosine."""
    scores = vectors @ query_vec
    top = np.argsort(scores)[-k:][::-1]
    return [(float(scores[i]), chunks[i]) for i in top]


class NumpyStore:
    """The reference implementation: the .npy index built by ingest.py.

    Exposes the three operations the retrievers need. PostgresStore mirrors it,
    and is only correct insofar as it returns the same thing.
    """

    name = "numpy"

    def __init__(self, chunks: list[dict] | None = None, vectors: np.ndarray | None = None):
        if chunks is None or vectors is None:
            chunks, vectors = load()
        self.chunks = chunks
        self.vectors = vectors

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[float, dict]]:
        return search(query_vec, self.chunks, self.vectors, k)

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity against every chunk, aligned with self.chunks."""
        return self.vectors @ query_vec

    def dense_score_at(self, chunk_id: int, query_vec: np.ndarray) -> float:
        """One chunk's cosine similarity.

        Deliberately a single-row dot rather than indexing into dense_scores:
        a full matrix-vector product accumulates in a different order and lands
        ~1e-8 away, which is enough to make the Postgres port look inexact when
        it is not.
        """
        return float(self.vectors[chunk_id] @ query_vec)
