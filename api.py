"""HTTP API over the study index.

A port of the scripts, not a rewrite: /upload runs ingest.py's pipeline, /ask
calls qa.answer, /quiz/generate calls quiz.build. Retrieval and grounding are
untouched. /quiz/submit is the one piece of new logic - no script scores an
attempt.

    uvicorn api:app --reload
"""

import hashlib
import tempfile
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

import db
import quiz as quiz_module
from chunker import chunk_pages
from config import ANSWER_K, QUIZ_DEFAULT_N
from embedder import embed_chunks
from pdf_parser import extract_pages


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.apply_schema()
    yield
    db.close()


app = FastAPI(title="SprintStudy", lifespan=lifespan)


@contextmanager
def upstream():
    """Turn a script-shaped failure into an HTTP one.

    llm.generate and db.pool() raise SystemExit, which is right for a CLI and
    wrong here: SystemExit is a BaseException, so it bypasses FastAPI's handlers
    and the caller gets a bare 500 with no body explaining a rate limit or a
    stopped container. 503 with the message is something a client can act on.
    """
    try:
        yield
    except SystemExit as exc:
        raise HTTPException(503, str(exc)) from None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: int | None = None


class QuizRequest(BaseModel):
    start_page: int
    end_page: int
    n: int = QUIZ_DEFAULT_N
    document_id: int | None = None


class SubmitRequest(BaseModel):
    quiz_id: uuid.UUID
    responses: list[int]


# --------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------


@app.post("/upload", status_code=202)
async def upload(file: UploadFile, background: BackgroundTasks) -> dict:
    """Accept a PDF and ingest it in the background.

    Ingestion of a 296-page PDF takes minutes, so holding the request open
    would die at any proxy in front of this. The row is created as 'pending'
    and the caller polls GET /documents/{id}.
    """
    payload = await file.read()
    if not payload:
        raise HTTPException(400, "empty upload")

    digest = hashlib.sha256(payload).hexdigest()

    with upstream(), db.pool().connection() as conn:
        existing = conn.execute(
            "SELECT id, status FROM documents WHERE sha256 = %s", (digest,)
        ).fetchone()
        if existing:
            # Same bytes, same index. Re-ingesting would duplicate 655 rows.
            return {"document_id": existing[0], "status": existing[1], "reused": True}

        row = conn.execute(
            "INSERT INTO documents (filename, sha256) VALUES (%s, %s) RETURNING id",
            (file.filename or "upload.pdf", digest),
        ).fetchone()
        document_id = row[0]

    scratch = Path(tempfile.gettempdir()) / f"upload-{document_id}.pdf"
    scratch.write_bytes(payload)
    background.add_task(_ingest, document_id, scratch)

    return {"document_id": document_id, "status": "pending", "reused": False}


def _ingest(document_id: int, path: Path) -> None:
    """The ingest.py pipeline, writing to Postgres instead of .npy."""
    try:
        with db.pool().connection() as conn:
            conn.execute(
                "UPDATE documents SET status = 'processing' WHERE id = %s", (document_id,)
            )

        pages = extract_pages(str(path))
        chunks = chunk_pages(pages)
        vectors = embed_chunks([c["text"] for c in chunks])

        with db.pool().connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, page, text, embedding)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    [
                        (document_id, c["id"], c["page"], c["text"], vectors[i])
                        for i, c in enumerate(chunks)
                    ],
                )
            conn.execute(
                "UPDATE documents SET status = 'ready', pages = %s, chunk_count = %s"
                " WHERE id = %s",
                (len(pages), len(chunks), document_id),
            )
    except Exception as exc:  # noqa: BLE001 - recorded on the row for the client
        with db.pool().connection() as conn:
            conn.execute(
                "UPDATE documents SET status = 'failed', error = %s WHERE id = %s",
                (f"{type(exc).__name__}: {exc}"[:2000], document_id),
            )
    finally:
        path.unlink(missing_ok=True)


@app.get("/documents")
def documents() -> list[dict]:
    """Every indexed document, newest first.

    Exists so the frontend can offer what is already indexed instead of
    demanding an upload before anything works. A deployed instance with a
    document in it should be usable by someone who arrived with no PDF.
    """
    with upstream(), db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, filename, status, pages, chunk_count, created_at"
            "  FROM documents WHERE status = 'ready' ORDER BY id DESC"
        ).fetchall()
    keys = ("document_id", "filename", "status", "pages", "chunk_count", "created_at")
    return [dict(zip(keys, row)) for row in rows]


@app.get("/documents/{document_id}")
def document(document_id: int) -> dict:
    """Ingestion progress. Without this, background work is unobservable."""
    with upstream(), db.pool().connection() as conn:
        row = conn.execute(
            "SELECT id, filename, status, pages, chunk_count, error, created_at"
            "  FROM documents WHERE id = %s",
            (document_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(404, f"no document {document_id}")
    keys = ("document_id", "filename", "status", "pages", "chunk_count", "error", "created_at")
    return dict(zip(keys, row))


# --------------------------------------------------------------------------
# ask
# --------------------------------------------------------------------------


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    import qa

    with upstream():
        store = _ready_store(request.document_id)
        # Passed, never assigned onto qa: sync handlers run in a threadpool, so a
        # module-level store is shared across concurrent requests.
        return qa.answer(request.question, ANSWER_K, store=store)


# --------------------------------------------------------------------------
# quiz
# --------------------------------------------------------------------------


@app.post("/quiz/generate")
def generate_quiz(request: QuizRequest) -> dict:
    with upstream():
        store = _ready_store(request.document_id)
        questions, stats = quiz_module.build(
            store.chunks, request.start_page, request.end_page, request.n
        )
    if not questions:
        raise HTTPException(
            422, f"no usable chunks on pages {request.start_page}-{request.end_page}"
        )

    quiz_id = uuid.uuid4()
    by_index = {c["id"]: c for c in store.chunks}

    with db.pool().connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO quiz_questions (quiz_id, document_id, source_chunk_id,"
                " position, start_page, end_page, question, options, answer_index,"
                " evidence, explanation, page)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        quiz_id,
                        store.document_id,
                        # The row's primary key, not the chunk_index - this is a
                        # real foreign key, so the traceability contract is
                        # enforced by the database rather than by convention.
                        by_index[q["source_chunk_id"]]["db_id"],
                        position,
                        request.start_page,
                        request.end_page,
                        q["question"],
                        q["options"],
                        q["answer_index"],
                        q["evidence"],
                        q["explanation"],
                        q["page"],
                    )
                    for position, q in enumerate(questions)
                ],
            )

    return {
        "quiz_id": str(quiz_id),
        "pages": [request.start_page, request.end_page],
        "stats": dict(stats),
        # answer_index withheld: the client is about to be graded on it.
        "questions": [
            {
                "position": position,
                "question": q["question"],
                "options": q["options"],
                "page": q["page"],
            }
            for position, q in enumerate(questions)
        ],
    }


@app.post("/quiz/submit")
def submit_quiz(request: SubmitRequest) -> dict:
    """Score an attempt. The only logic here that no script already had."""
    with upstream(), db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT position, answer_index, page, source_chunk_id, explanation"
            "  FROM quiz_questions WHERE quiz_id = %s ORDER BY position",
            (request.quiz_id,),
        ).fetchall()

        if not rows:
            raise HTTPException(404, f"no quiz {request.quiz_id}")
        if len(request.responses) != len(rows):
            raise HTTPException(
                400, f"quiz has {len(rows)} questions, got {len(request.responses)} responses"
            )

        marks = [
            {
                "position": position,
                "submitted": submitted,
                "correct_index": correct,
                "correct": submitted == correct,
                "page": page,
                "source_chunk_id": chunk_id,
                "explanation": explanation,
            }
            for (position, correct, page, chunk_id, explanation), submitted in zip(
                rows, request.responses
            )
        ]
        score = sum(m["correct"] for m in marks)

        conn.execute(
            "INSERT INTO quiz_attempts (quiz_id, responses, score, total)"
            " VALUES (%s, %s, %s, %s)",
            (request.quiz_id, request.responses, score, len(rows)),
        )

    return {"quiz_id": str(request.quiz_id), "score": score, "total": len(rows), "marks": marks}


# --------------------------------------------------------------------------


def _ready_store(document_id: int | None):
    """The Postgres store for a document, refusing anything not yet ingested."""
    from pgstore import PostgresStore, latest_document_id

    resolved = document_id if document_id is not None else latest_document_id()
    with db.pool().connection() as conn:
        row = conn.execute("SELECT status FROM documents WHERE id = %s", (resolved,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no document {resolved}")
    if row[0] != "ready":
        raise HTTPException(409, f"document {resolved} is {row[0]}, not ready")
    return PostgresStore(resolved)
