# Smart Study Assistant

A RAG-based study tool. Upload a PDF, ask questions and get answers grounded only in
that document with a page citation, and auto-generate an MCQ quiz from it.

The retrieval layer is written from scratch — no LangChain, no LlamaIndex, no separate
vector database. Chunking, embedding, storage and search are all in this repo.

> **Status:** in progress. Milestones 1 (ingestion), 2 (eval harness), 3 (grounded Q&A)
> and 4 (retrieval variants) are done. Quiz generation is next.

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
export GROQ_API_KEY="..."   # https://console.groq.com/keys
```

The key is only needed for answering; ingestion and retrieval run offline.

## Usage

```sh
python ingest.py book.pdf          # parse, chunk, embed, write data/
python search.py "your question"   # top-k chunks with page numbers
python sections.py                 # section list with PDF page numbers
python eval.py                     # score retrieval against eval.json
python qa.py "your question"       # grounded answer with page citations
python qa.py --all                 # every eval question through the pipeline
```

`sections.py` is the reference for `expected_pages` in `eval.json`. The book's printed
page numbers are offset from the PDF page numbers the index stores, so the two are not
interchangeable.

## Retrieval results

Measured over 20 hand-written questions — 15 answerable with expected page numbers,
5 unanswerable — with `python eval.py --retriever <name>`. All three run against the
same questions and the same index.

| metric | dense | hybrid | **HyDE** |
|---|---|---|---|
| recall@5 | 0.800 | 0.800 | **0.933** |
| recall@1 | 0.600 | 0.467 | **0.867** |
| MRR | 0.667 | 0.633 | **0.900** |
| paraphrase recall@1 | 0.286 | 0.000 | **0.714** |

HyDE wins because of what the eval set is built to expose. Every dense failure is a
question that describes a concept without naming it — p04 asks about walking every
connection in a network exactly once and never says *Eulerian*. Generating a
hypothetical answer first puts the missing word into the search vector: the cached
passage for p04 opens *"A walk that traverses every edge of a graph exactly once (an
Eulerian trail)…"*, and the item moves from a miss to rank 1.

End to end, answering with HyDE gives 15/15 answerable questions answered and 5/5
unanswerable refused.

### Recorded negative result: hybrid retrieval

Dense and BM25 rankings fused with reciprocal rank fusion, `w` being keyword's share.
Swept rather than tuned, and **not adopted**:

| w | recall@5 | recall@1 | MRR | paraphrase recall@1 | |
|---|---|---|---|---|---|
| 0.0 | 0.800 | 0.600 | 0.667 | 0.286 | pure dense |
| 0.1 | 0.800 | 0.600 | 0.663 | 0.286 | |
| 0.2 | 0.800 | 0.467 | 0.597 | 0.143 | |
| 0.3 | 0.800 | 0.533 | 0.639 | 0.143 | |
| 0.4 | **0.867** | 0.467 | 0.636 | 0.000 | best recall@5 |
| 0.5 | 0.800 | 0.467 | 0.633 | 0.000 | |
| 0.6 | 0.800 | 0.467 | 0.600 | 0.000 | |
| 0.7 | 0.800 | 0.533 | 0.616 | 0.143 | |
| 0.8 | 0.667 | 0.533 | 0.580 | 0.143 | |
| 0.9 | 0.600 | 0.533 | 0.567 | 0.143 | |
| 1.0 | 0.600 | 0.533 | 0.567 | 0.143 | pure BM25 |

**No weight beats dense.** recall@1 never exceeds its 0.600 baseline anywhere in the
range, and paraphrase recall@1 never exceeds 0.286. The one gain — recall@5 0.867 at
w=0.4 — costs 0.133 of recall@1 and takes paraphrase recall@1 to zero. At w=0.1 hybrid
is indistinguishable from dense, so it does not degrade gracefully toward being useful;
it converges on being dense with extra machinery.

The trade is structural rather than a tuning artefact. The items BM25 helps (p01, p04)
and the ones it harms (p03, p05, p06) are disjoint, and the harmed ones already sit at
rank 1, where keyword influence can only push them down.

**p07 is never retrieved at any weight, pure BM25 included.** Queried directly with the
term *"amortized analysis"*, BM25 returns page 87 — exactly the right page. But p07 asks
*"why does an operation that occasionally does a lot of work still count as cheap when
you average it out?"* and contains no matchable term at all. Keyword scoring cannot
reach a passage with which the question shares no vocabulary, however it is weighted.
That is the gap HyDE closes and BM25 structurally cannot.

BM25 is implemented in `bm25.py` rather than pulled in as a dependency. It has no
stemmer, so "trees" does not match "tree".

## Stack

Python 3.11, pdfplumber, sentence-transformers (`bge-small-en-v1.5`), numpy.
Later: FastAPI, PostgreSQL + pgvector, React, Groq.
