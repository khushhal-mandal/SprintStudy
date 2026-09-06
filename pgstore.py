"""The same index, served from Postgres with pgvector.

Correctness here means one thing: returning exactly what NumpyStore returns.
Three details carry that, and each has a plausible alternative that would break
it quietly.

- `<#>` is negative inner product. Embeddings are L2-normalised at creation, so
  the dot product already is cosine similarity - `-(embedding <#> q)` reproduces
  `vectors @ query_vec` operation for operation. `<=>` would recompute norms and
  reintroduce float drift the NumPy path never had.
- pgvector's vector is float4 and embedder emits float32, so storage is lossless
  rather than merely close.
- There is no ANN index on embedding, so this is an exact scan. See schema.sql.

Chunk dicts carry "id" as the chunk_index rather than the row's primary key, so
everything downstream - retrievers, eval, quiz - indexes them exactly as it does
the NumPy chunks. The primary key travels alongside as "db_id" for foreign keys.
"""

import numpy as np

import db

SEARCH_SQL = """
    SELECT id, chunk_index, page, text, -(embedding <#> %(q)s) AS score
      FROM chunks
     WHERE document_id = %(doc)s
     ORDER BY embedding <#> %(q)s, chunk_index
     LIMIT %(k)s
"""

SCORES_SQL = """
    SELECT -(embedding <#> %(q)s) AS score
      FROM chunks
     WHERE document_id = %(doc)s
     ORDER BY chunk_index
"""

SCORE_AT_SQL = """
    SELECT -(embedding <#> %(q)s) AS score
      FROM chunks
     WHERE document_id = %(doc)s AND chunk_index = %(i)s
"""

CHUNKS_SQL = """
    SELECT id, chunk_index, page, text
      FROM chunks
     WHERE document_id = %(doc)s
     ORDER BY chunk_index
"""


def latest_document_id() -> int:
    with db.pool().connection() as conn:
        row = conn.execute(
            "SELECT id FROM documents WHERE status = 'ready' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise SystemExit(
            "No ingested document is ready.\n"
            "  Upload one:  curl -F file=@book.pdf localhost:8000/upload"
        )
    return row[0]


def _row(record) -> dict:
    db_id, chunk_index, page, text = record[:4]
    return {"id": chunk_index, "db_id": db_id, "page": page, "text": text}


class PostgresStore:
    name = "postgres"

    def __init__(self, document_id: int | None = None):
        self.document_id = document_id if document_id is not None else latest_document_id()
        self._chunks: list[dict] | None = None
        # Distinguishes documents for anything cached per store, such as the
        # BM25 index in retrievers.py.
        self.key = f"postgres:{self.document_id}"

    @property
    def chunks(self) -> list[dict]:
        """All chunks in chunk_index order, so position == chunk["id"]."""
        if self._chunks is None:
            with db.pool().connection() as conn:
                rows = conn.execute(CHUNKS_SQL, {"doc": self.document_id}).fetchall()
            self._chunks = [_row(r) for r in rows]
        return self._chunks

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[float, dict]]:
        params = {"q": np.asarray(query_vec, dtype="float32"), "doc": self.document_id, "k": k}
        with db.pool().connection() as conn:
            rows = conn.execute(SEARCH_SQL, params).fetchall()
        return [(float(r[4]), _row(r)) for r in rows]

    def dense_score_at(self, chunk_id: int, query_vec: np.ndarray) -> float:
        params = {
            "q": np.asarray(query_vec, dtype="float32"),
            "doc": self.document_id,
            "i": chunk_id,
        }
        with db.pool().connection() as conn:
            row = conn.execute(SCORE_AT_SQL, params).fetchone()
        if row is None:
            raise KeyError(f"no chunk_index {chunk_id} in document {self.document_id}")
        return float(row[0])

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        params = {"q": np.asarray(query_vec, dtype="float32"), "doc": self.document_id}
        with db.pool().connection() as conn:
            rows = conn.execute(SCORES_SQL, params).fetchall()
        return np.array([r[0] for r in rows], dtype="float32")
