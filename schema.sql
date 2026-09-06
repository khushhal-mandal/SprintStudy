CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    filename    text        NOT NULL,
    sha256      text        NOT NULL UNIQUE,
    pages       integer,
    chunk_count integer,
    status      text        NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    error       text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial   PRIMARY KEY,
    document_id bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index integer     NOT NULL,
    page        integer     NOT NULL,
    text        text        NOT NULL,
    embedding   vector(384) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

-- Page-range scan for quiz generation.
CREATE INDEX IF NOT EXISTS chunks_document_page_idx ON chunks (document_id, page);

-- DELIBERATELY NO INDEX ON embedding.
--
-- ivfflat and hnsw are approximate: they trade recall for speed and would
-- return different neighbours from the NumPy store this was ported from, with
-- nothing in a diff to explain the change. The whole point of the port is that
-- eval.py produces identical numbers against either backend.
--
-- A sequential scan over a few hundred rows is instant. Do not add an index
-- here without re-running eval.py against both stores and accepting whatever
-- the recall figures do.

CREATE TABLE IF NOT EXISTS quiz_questions (
    id              bigserial   PRIMARY KEY,
    quiz_id         uuid        NOT NULL,
    document_id     bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    -- The traceability contract as a constraint rather than a convention: a
    -- question cannot reference a chunk that does not exist.
    source_chunk_id bigint      NOT NULL REFERENCES chunks(id),
    position        integer     NOT NULL,
    start_page      integer     NOT NULL,
    end_page        integer     NOT NULL,
    question        text        NOT NULL,
    options         text[]      NOT NULL,
    answer_index    integer     NOT NULL,
    evidence        text        NOT NULL,
    explanation     text        NOT NULL,
    page            integer     NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (quiz_id, position),
    CHECK (answer_index >= 0 AND answer_index < array_length(options, 1))
);

CREATE INDEX IF NOT EXISTS quiz_questions_quiz_idx ON quiz_questions (quiz_id);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id           bigserial   PRIMARY KEY,
    quiz_id      uuid        NOT NULL,
    responses    integer[]   NOT NULL,
    score        integer     NOT NULL,
    total        integer     NOT NULL,
    submitted_at timestamptz NOT NULL DEFAULT now()
);
