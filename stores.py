"""Pick a vector store backend.

NumPy is the reference; Postgres is a port of it. Kept in its own module so
importing the NumPy path never pulls in psycopg, and the CLI scripts and the API
resolve the backend the same way.
"""

from config import STORE


def open_store(backend: str | None = None, document_id: int | None = None):
    """Open a store. document_id selects which document the Postgres store reads.

    Only meaningful for Postgres, and only once more than one document is
    indexed: without it the benchmark silently follows whichever upload happened
    most recently.
    """
    name = backend or STORE
    if name == "numpy":
        from store import NumpyStore

        return NumpyStore()
    if name == "postgres":
        from pgstore import PostgresStore

        return PostgresStore(document_id)
    raise SystemExit(f"Unknown store {name!r}. Use 'numpy' or 'postgres'.")
