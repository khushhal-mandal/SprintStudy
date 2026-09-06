"""Postgres connection pool and schema bootstrap.

A pool rather than a connection per request: FastAPI serves requests
concurrently and reconnecting per request is the standard way to make a small
service slow. Opened lazily so the CLI scripts and the API share one path.
"""

import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from config import DATABASE_URL

SCHEMA_PATH = "schema.sql"

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """Teach the connection about the vector type so numpy arrays round-trip."""
    register_vector(conn)


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        # Probe once directly. The pool retries in a background thread and
        # reports PoolTimeout, which says the pool did not fill but not why -
        # a single connect surfaces the actual cause for the message below.
        try:
            psycopg.connect(DATABASE_URL, connect_timeout=5).close()
        except psycopg.Error as exc:
            raise SystemExit(_unreachable(exc)) from None

        # The pool logs every failed attempt at ERROR. We report failures
        # ourselves, so its retry chatter is noise on the way to a clear message.
        quiet = logging.getLogger("psycopg.pool")
        previous = quiet.level
        quiet.setLevel(logging.CRITICAL)
        try:
            _pool = ConnectionPool(
                DATABASE_URL,
                min_size=1,
                max_size=8,
                open=False,
                configure=_configure,
                kwargs={"connect_timeout": 5},
            )
            _pool.open(wait=True, timeout=10)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
            _pool = None
            raise SystemExit(_unreachable(exc)) from None
        finally:
            quiet.setLevel(previous)
    return _pool


def _unreachable(exc: Exception) -> str:
    """A cause and a fix, rather than a psycopg traceback."""
    safe = DATABASE_URL
    if "@" in safe:
        safe = safe.split("://", 1)[0] + "://" + safe.split("@", 1)[1]
    detail = next(
        (line.strip() for line in str(exc).splitlines() if line.strip()), str(exc)
    )
    return (
        f"Cannot reach Postgres at {safe}\n"
        f"  {type(exc).__name__}: {detail}\n\n"
        "Is the container running?\n"
        "  docker compose up -d\n"
        "  docker compose ps\n\n"
        "Override the target with DATABASE_URL if it lives elsewhere."
    )


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def apply_schema() -> None:
    """Idempotent - every statement in schema.sql is IF NOT EXISTS."""
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        ddl = fh.read()
    with pool().connection() as conn:
        conn.execute(ddl)


def healthy() -> bool:
    try:
        with pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except SystemExit:
        return False
