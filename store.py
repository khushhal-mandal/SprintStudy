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
