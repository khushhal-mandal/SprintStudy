# Smart Study Assistant

A RAG-based study tool. Upload a PDF, ask questions and get answers grounded only in
that document with a page citation, and auto-generate an MCQ quiz from it.

The retrieval layer is written from scratch — no LangChain, no LlamaIndex, no separate
vector database. Chunking, embedding, storage and search are all in this repo.

> **Status:** in progress. Milestone 1 (ingestion) and milestone 2 (eval harness) are
> done. Eval numbers go in this README once the eval set is scored.

## Test document

Everything is developed and measured against:

**Competitive Programmer's Handbook** — Antti Laaksonen, draft of 3 July 2018, 296 pages.

**The PDF is not committed.** It's in `.gitignore` as `book.pdf`, along with the
generated index in `data/`. Nothing here redistributes the book — supply your own copy
to reproduce the numbers, or point the pipeline at any text-based PDF.

Rough shape of the indexed corpus, for context on the eval numbers: 277 of 296 pages
carried extractable text, producing 680 chunks averaging 685 characters (max 407 tokens
against the embedding model's 512 limit).

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```sh
python ingest.py book.pdf          # parse, chunk, embed, write data/
python search.py "your question"   # top-k chunks with page numbers
python sections.py                 # section list with PDF page numbers
python eval.py                     # score retrieval against eval.json
```

`sections.py` is the reference for `expected_pages` in `eval.json`. The book's printed
page numbers are offset from the PDF page numbers the index stores, so the two are not
interchangeable.

## Stack

Python 3.11, pdfplumber, sentence-transformers (`bge-small-en-v1.5`), numpy.
Later: FastAPI, PostgreSQL + pgvector, React, Groq.
