"""Pick a vector store backend.

NumPy is the reference; Postgres is a port of it. Kept in its own module so
importing the NumPy path never pulls in psycopg, and the CLI scripts and the API
resolve the backend the same way.
"""

from config import STORE


def open_store(backend: str | None = None):
    name = backend or STORE
    if name == "numpy":
        from store import NumpyStore

        return NumpyStore()
    if name == "postgres":
        from pgstore import PostgresStore

        return PostgresStore()
    raise SystemExit(f"Unknown store {name!r}. Use 'numpy' or 'postgres'.")
