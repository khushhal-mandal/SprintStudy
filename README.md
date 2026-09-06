# Smart Study Assistant

Upload a PDF, ask questions about it, and get answers grounded **only** in that document
with the page they came from. Generate a multiple-choice quiz from any page range, where
every question is traceable to the chunk it was written from.

The retrieval layer is written from scratch — no LangChain, no LlamaIndex, no separate
vector database. Chunking, embedding, BM25, rank fusion, storage and search are all in this
repo, and every retrieval decision in it was measured against a hand-written eval set
before it was kept.

> **Status:** milestones 1–8 complete. Known gaps are in [Limitations](#limitations), not
> hidden.

## What it does

- **Grounded answers.** Retrieval returns the top 5 chunks; the model answers from those
  and nothing else. Questions the document does not cover are refused rather than guessed
  at — 5 of 5 on the unanswerable half of the eval set.
- **Citations that cannot be fabricated.** The model emits passage indices, never page
  numbers. Pages are looked up from chunk metadata afterwards, so a wrong page is
  unrepresentable rather than merely unlikely.
- **Quiz generation with traceability.** Each question carries a verbatim `evidence` span
  checked against its source chunk, and `source_chunk_id` is a foreign key — the database
  rejects a question citing a chunk that does not exist.

## Architecture

```
INGESTION                                              one row per page-scoped chunk
                                                                     |
  PDF ──> pdf_parser ──> chunker ──> embedder ──────────────> chunks(embedding vector(384))
           │              │           │                              │
           │              │           └─ bge-small-en-v1.5,          │
           │              │              L2-normalised at creation   │
           │              └─ never spans a page boundary,            │
           │                 so every chunk cites one page           │
           └─ repairs 36 pdfminer (cid:N) glyph codes

QUERY
                    ┌─ HyDE: write a hypothetical answer, embed that ─┐
  question ─────────┤                                                 ├──> pgvector
                    └─ dense: embed the question directly ────────────┘    exact scan
                                                                            │
                                                                     top 5 chunks
                                                                            │
                          grounding prompt: answer from these passages only,
                          or reply INSUFFICIENT_CONTEXT
                                                                            │
                             answer with [n] markers ──> pages resolved from
                                                          chunk metadata
```

Two things are load-bearing and easy to miss. **The query prefix goes on questions only** —
`bge` models are trained that way, and a HyDE hypothetical is a passage, so it is embedded
without one. **There is no ANN index on the embedding column**: `ivfflat` and `hnsw` are
approximate and would return different neighbours from the NumPy reference implementation.

## Quickstart

```sh
cp .env.example .env        # fill in POSTGRES_*, DATABASE_URL, GROQ_API_KEY
docker compose up -d        # Postgres + pgvector, and the API on :8000

curl -F file=@book.pdf localhost:8000/upload      # 202, ingests in the background
curl localhost:8000/documents/1                   # poll until status is "ready"

curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question":"How does a binary indexed tree compute a prefix sum?"}'

curl -X POST localhost:8000/quiz/generate -H 'Content-Type: application/json' \
     -d '{"start_page":93,"end_page":103,"n":5}'
curl -X POST localhost:8000/quiz/submit -H 'Content-Type: application/json' \
     -d '{"quiz_id":"...","responses":[0,2,1,3,0]}'
```

The API waits on Postgres's healthcheck rather than crash-looping against a database that
is still initialising. `GROQ_API_KEY` is only needed for answering and quiz generation —
ingestion and retrieval are entirely offline, and a missing key surfaces as a 503 naming
the cause.

**The embedding model is not baked into the image.** It is fetched on first container start
into the `study-models` volume, so the first `docker compose up` takes about a minute longer
while 128MB of weights download — `docker compose logs -f api` shows it — and every start
after that is a cache hit. Baking it cost a 136MB layer (80MB compressed) holding a copy of
a third-party artefact versioned independently of this repo; without it the image is 1.58GB
rather than 1.72GB. The trade is deliberate: the image no longer works on a machine with no
network and a cold volume.

Frontend (React, Vite, plain `fetch`, no state library):

```sh
cd frontend && npm install && npm run dev     # :5173, proxies /api to the backend
```

Scripts, for working against the NumPy index without the service:

```sh
python ingest.py book.pdf                     # parse, chunk, embed, write data/
python search.py "your question"              # top-k chunks with pages
python sections.py                            # section list with PDF page numbers
python eval.py --retriever hyde               # score retrieval against eval.json
python eval.py --compare                      # every saved run side by side
python sweep.py                               # hybrid blend weight curve
python qa.py --all                            # every eval question through the pipeline
python quiz.py 93 103 5                       # 5 MCQs sampled across a page range
```

---

## Results

Measured over 20 hand-written questions — 15 answerable with expected page numbers, 5
unanswerable — against a 673-chunk index of a 296-page textbook. Every figure below comes
from a committed run in `eval_runs/`.

### Retrieval

| metric | dense | hybrid | **HyDE** |
|---|---|---|---|
| recall@5 | 0.800 | 0.800 | **0.933** |
| recall@1 | 0.600 | 0.467 | **0.867** |
| MRR | 0.667 | 0.611 | **0.900** |
| paraphrase recall@1 | 0.286 | 0.000 | **0.714** |
| multi_page strict | 1.000 | 1.000 | **1.000** |

**HyDE wins for the reason the eval set was built to expose.** Every dense failure is a
question that describes a concept without naming it — p04 asks about walking every
connection in a network exactly once and never says *Eulerian*. Writing a hypothetical
answer first puts the missing word into the search vector: the cached passage for p04 opens
*"A walk that traverses every edge of a graph exactly once (an Eulerian trail)…"*, and the
item moves from a miss to rank 1.

`multi_page strict` is the stricter reading of the multi-page questions: a hit requires a
chunk from *every* cluster the answer needs, not just one. It is reported because the
lenient number flatters items with wide page lists — and it came back 1.000, so the perfect
lenient score turned out to be real.

### Recorded negative result: hybrid retrieval

Dense and BM25 rankings fused with reciprocal rank fusion, `w` being keyword's share.
Swept, not tuned, and **not adopted** (`eval_runs/hybrid-sweep.json`):

| w | recall@5 | recall@1 | MRR | paraphrase r@1 | |
|---|---|---|---|---|---|
| 0.0 | 0.800 | 0.600 | 0.667 | 0.286 | pure dense |
| 0.1 | 0.800 | 0.600 | 0.663 | 0.286 | |
| 0.2 | 0.733 | 0.533 | 0.617 | 0.286 | |
| 0.3 | 0.800 | 0.600 | 0.667 | 0.286 | |
| 0.4 | 0.800 | 0.467 | 0.594 | 0.000 | |
| 0.5 | 0.800 | 0.467 | 0.611 | 0.000 | committed default |
| 0.6 | 0.800 | 0.467 | 0.589 | 0.000 | |
| 0.7 | 0.800 | 0.467 | 0.582 | 0.000 | |
| 0.8 | 0.667 | 0.467 | 0.547 | 0.000 | |
| 0.9 | 0.600 | 0.467 | 0.533 | 0.000 | |
| 1.0 | 0.600 | 0.467 | 0.533 | 0.000 | pure BM25 |

**No weight beats dense on any metric.** recall@5 never exceeds its 0.800 baseline anywhere
in the range, recall@1 never exceeds 0.600, and paraphrase recall@1 never exceeds 0.286. At
w=0.1 hybrid is indistinguishable from dense, so it does not degrade gracefully toward
being useful — it converges on being dense with extra machinery.

The trade is structural rather than a tuning artefact. The items BM25 helps (p01, p04) and
the ones it harms (p03, p05, p06) are disjoint, and the harmed ones already sit at rank 1,
where keyword influence can only push them down.

**p07 is never retrieved at any weight, pure BM25 included.** Queried directly with the
term *"amortized analysis"*, BM25 returns page 87 — exactly right. But p07 asks *"why does
an operation that occasionally does a lot of work still count as cheap when you average it
out?"* and shares no vocabulary with its target at all. Keyword scoring cannot reach a
passage it has no term in common with, however it is weighted. That is the gap HyDE closes
and BM25 structurally cannot.

### Refusal is a prompt, not a threshold

The obvious design — refuse when the top similarity score is low — does not work here, and
the eval set shows why. Top-1 similarity between the question and the winning chunk:

| | min | max |
|---|---|---|
| answerable | 0.568 | 0.888 |
| unanswerable | 0.574 | 0.675 |

**The lowest answerable question scores below every unanswerable one.** The distributions
are not merely overlapping but interleaved, so no cut separates them — any threshold placed
in that band refuses real questions to no purpose. This is a consequence of the eval set
being adversarial: the unanswerable questions are in-domain algorithms the book does not
cover (Fibonacci heaps, red-black trees, simplex, Karatsuba, treaps), not off-topic
questions that would have separated trivially.

So refusal moved into the grounding prompt. Retrieval always returns the top 5 regardless
of score, and the model decides whether they contain the answer, replying with a sentinel
when they do not. `MIN_CONTEXT_SCORE = 0.40` remains, reframed as a crash guard against
empty or nonsense input — not a relevance judgement, and it has never fired on the eval set.

End to end, the last complete pipeline run scored 5/5 unanswerable refused, 15/15 answerable
answered, 0 page-number leaks, and 0 answers without a citation (`qa_run.json`). *That run
predates the front-matter change and was measured on a 655-chunk index; the re-run against
the current 673 chunks reached 18 of 20 questions before the daily token cap stopped it,
with all 15 answerable answered and 3 of 3 unanswerable reached refused.*

### The pgvector port is exact

The Postgres store had to return exactly what the NumPy one returns, verified by re-running
the whole eval against it rather than spot-checking:

| | numpy | postgres |
|---|---|---|
| dense recall@5 | 0.800 | 0.800 |
| hybrid recall@1 | 0.467 | 0.467 |
| HyDE MRR | 0.900 | 0.900 |

Identical `rank`, `RR` and `top_pages` on all 20 questions for all three retrievers, with
**zero top-5 ordering mismatches** and a maximum score delta of **2 units in the last place
of float32** (2.384e-07 against an eps of 1.192e-07). Chunk text, pages and embeddings
round-trip bit-identically.

Three decisions carry that, each with a plausible alternative that would have broken it
silently: no ANN index; `<#>` (negative inner product) rather than `<=>`, since the vectors
are already L2-normalised and `<=>` would recompute norms; and `vector` being float4, which
matches the float32 the embedder emits.

---

## What building this actually surfaced

### Porting scripts to a service exposed three latent bugs

Milestones 1–5 were scripts. Milestone 6 put the same logic behind FastAPI and Postgres
with no changes to retrieval or grounding — and that move alone exposed three bugs. All
three were state or error handling. None were in the core logic, which ported unchanged and
reproduces its numbers exactly on either backend.

- **`SystemExit` bypassing FastAPI's error handlers.** `llm.generate` and `db.pool()` exit
  the process on failure, which is right for a CLI. `SystemExit` is a `BaseException`, so it
  skipped FastAPI's handling entirely and an exhausted token quota reached the client as a
  bare 500 with no body. Now translated to a 503 carrying the cause.

- **A module-level store assigned per request.** `/ask` resolved the caller's document then
  assigned it onto a global before answering. FastAPI runs sync handlers in a threadpool, so
  two overlapping requests shared it: the second overwrote the global before the first read
  it, and a question about one document was answered — and cited — from another. The SQL had
  been filtered by `document_id` the whole time; the plumbing choosing *which* document was
  not. Reproduced with two threads, fixed by passing the store as an argument, verified at
  zero leaks across four concurrent requests.

- **A globally cached BM25 index.** Built once from whichever document was indexed first, so
  with more than one document every keyword query would have scored against the wrong
  vocabulary. Latent rather than live, and waiting for a config change to become real.

Single-document, single-threaded and fail-by-exiting are assumptions a script holds silently
and a service cannot. The retrieval maths needed nothing.

### Two ingestion bugs, both invisible until something else exposed them

- **36 unmapped glyph codes.** The book is typeset with Fourier and AMS math fonts whose
  built-in Type 1 encodings pdfminer cannot resolve, so every square root, floor bracket and
  summation extracted as a `(cid:N)` marker — corrupting 110 of 680 chunks across 62 pages.
  All 36 codes were enumerated from the PDF and identified from their font and surrounding
  text rather than guessed. One mattered more than the rest: `(cid:54)` is a negation slash
  drawn *over* the following `=`, so `x(cid:54)=0` means `x ≠ 0`. Substituting the code
  alone would have produced `x = 0` — an inverted meaning, silently, in the text a model
  answers from.

- **A front-matter skip applied to every document.** Ingestion skipped a fixed prefix of
  pages as front matter, with the count derived from this book's layout. Applied to a
  second, unrelated upload it discarded the first 10 pages of a 16-page document whose page
  1 was already body text — roughly two thirds of it, missing from the index with no error.

**Both were measured, and both had zero retrieval impact on the eval set**: repairing 110
corrupted chunks and re-adding 18 front-matter chunks moved no metric on any retriever, to
three decimal places. The glyph repair did fix something downstream — uncited answers went
from 2 to 0, because the model would not cite passages it could not parse.

That is the useful lesson, and it cuts both ways: a corpus can be visibly broken while every
headline number looks fine, which is an argument for looking at the data and not only the
metrics.

---

## Limitations

- **One retrieval failure remains: p01.** It asks how to precompute repeated range sums. The
  answer is the prefix sum array on page 94; HyDE returns page 96, where the binary indexed
  tree begins. Both pages were read against the book and the expected pages left unchanged —
  the BIT is the *dynamic* variant, for when values change between queries, which p01 never
  asks about. The cause is inherent to the method: a hypothetical answer about precomputing
  range sums naturally describes both structures, so it carries the wrong one's vocabulary
  into the search vector. HyDE's strength is writing the words the question omitted; the cost
  is that it also writes words the question did not want.

- **Front matter is indexed, not detected.** Every page goes in, contents pages included.
  Detection was considered and rejected: a heuristic tuned on two documents and applied to
  everything users upload is exactly how the skip bug above happened, and reliable front
  matter detection across arbitrary PDF layouts is not something two test documents can
  validate. Indexed front matter is noise rather than corruption — it does not outscore real
  content, and the dot-leader pre-filter already keeps contents pages out of quiz generation.

- **BM25 has no stemmer**, so "trees" does not match "tree". Adding one meant another
  dependency; the limitation is left visible rather than papered over.

- **`/quiz/generate` has never been exercised against a live model.** Its logic is tested
  against stubs and its contract is enforced by validation, but the endpoint's happy path is
  unverified. The frontend quiz view runs against a fixture (`USE_FIXTURE` in
  `frontend/src/api.js`) holding real model output captured from a CLI run.

- **The frontend has never been visually verified.** It builds, every module transforms, the
  dev proxy reaches the API and the citation rendering is exercised directly — but no one has
  confirmed how it looks or behaves in a browser.

- **The free tier shaped how often the benchmark could run.** Groq's API reported a
  200,000 token/day cap during development, which repeatedly gated end-to-end runs. HyDE
  hypotheticals are cached in a committed file precisely so the benchmark stays reproducible
  and cheap; runtime questions go to a separate gitignored cache so live usage never
  contaminates it. *(This figure comes from Groq's own rate-limit responses, not from
  anything in this repo.)*

- **The eval set is 20 questions written by one person.** It is enough to catch the failures
  documented here and to compare retrievers against each other. It is not enough to claim a
  number generalises.

---

## Stack

Python 3.11, pdfplumber, sentence-transformers (`bge-small-en-v1.5`), numpy, FastAPI,
PostgreSQL 17 + pgvector 0.8.6, psycopg 3, React 19 + Vite. Groq (`openai/gpt-oss-120b`)
for generation.

Deliberately excluded: Kubernetes, Kafka, Redis, a separate vector database, an ORM, and a
frontend state library. This project's scale does not justify them, and claiming otherwise
is worse than not having them.

### Test document

**Competitive Programmer's Handbook** — Antti Laaksonen, draft of 3 July 2018, 296 pages.
**The PDF is not committed** — it is in `.gitignore`, along with the generated index. Supply
your own copy to reproduce the numbers, or point the pipeline at any text-based PDF.

277 of 296 pages carry extractable text, producing 673 chunks averaging 685 characters
(max 407 tokens against the embedding model's 512 limit).

### Layout

| | |
|---|---|
| `pdf_parser.py` `chunker.py` `embedder.py` | ingestion |
| `store.py` `pgstore.py` `stores.py` | NumPy and Postgres backends behind one interface |
| `bm25.py` `retrievers.py` | keyword scoring and the three retrieval strategies |
| `llm.py` `qa.py` `quiz.py` | generation, grounded answering, quiz building |
| `eval.py` `sweep.py` `sections.py` | measurement |
| `api.py` `db.py` `schema.sql` | the service |
| `eval.json` `eval_runs/` | the eval set and every recorded run |
