# SprintStudy — how it works

A walkthrough of the whole system, module by module, with the reasoning behind each
decision and the alternative that was rejected. The README is the summary; this is the
long version.

Read it top to bottom once and the codebase should hold no surprises.

---

## Contents

1. [The one-sentence version](#1-the-one-sentence-version)
2. [Invariants — what breaks silently](#2-invariants--what-breaks-silently)
3. [Ingestion: PDF to vectors](#3-ingestion-pdf-to-vectors)
4. [Storage: two backends, one interface](#4-storage-two-backends-one-interface)
5. [Retrieval: three strategies](#5-retrieval-three-strategies)
6. [Answering: grounding and citation](#6-answering-grounding-and-citation)
7. [Quiz generation](#7-quiz-generation)
8. [The service layer](#8-the-service-layer)
9. [Measurement: the eval harness](#9-measurement-the-eval-harness)
10. [Deployment](#10-deployment)
11. [Bugs worth knowing about](#11-bugs-worth-knowing-about)
12. [If someone asks you to defend it](#12-if-someone-asks-you-to-defend-it)

---

## 1. The one-sentence version

Upload a PDF; it is split into page-scoped chunks, each embedded into a 384-dimension
vector; a question is embedded the same way, the nearest chunks are retrieved by cosine
similarity, and a language model is asked to answer **from those chunks only** — citing
them by index, never by page, so the page number can be looked up from metadata rather
than trusted from the model.

Everything else in this document is detail about why each of those steps is done the
particular way it is.

### The whole flow, one diagram

```
INGEST
  book.pdf
    │
    ├─ pdf_parser.extract_pages()      → [(page_no, text), ...]
    │    repairs 36 (cid:N) glyph codes, drops pages under 100 chars
    │
    ├─ chunker.chunk_pages()           → [{id, page, text}, ...]
    │    800 chars, 150 overlap, never crosses a page boundary
    │
    ├─ embedder.embed_chunks()         → ndarray (n, 384) float32, L2-normalised
    │    fastembed / ONNX, bge-small-en-v1.5, NO query prefix
    │
    └─ store                           → data/*.npy  or  chunks table

QUERY
  question
    │
    ├─ retriever ─┬─ dense:  embed_query(QUERY_PREFIX + q)
    │             ├─ hybrid: RRF(dense ranks, BM25 ranks)
    │             └─ hyde:   generate a fake answer, embed_passage(that)
    │
    ├─ store.search(vec, k=5)          → [(score, chunk), ...]
    │
    ├─ qa.build_prompt()               → numbered passages + rules
    ├─ llm.generate()                  → text with [1][3] markers
    └─ qa.parse_citations()            → markers resolved to pages from metadata
```

---

## 2. Invariants — what breaks silently

These are the rules that, if broken, produce a system that still runs, still returns
answers, and is quietly worse or wrong. They are the most important part of this document.

| # | Invariant | What breaks if violated |
|---|---|---|
| 1 | `QUERY_PREFIX` goes on **questions only**, never on chunks | bge models are trained asymmetrically. Prefixing both sides costs retrieval accuracy with no error anywhere. |
| 2 | Chunks never span a page boundary | Every chunk would need a page *range*, and a citation could no longer name one page truthfully. |
| 3 | Embeddings are L2-normalised **at creation** | `vectors @ query_vec` is only cosine similarity if both sides are unit length. Adding a normalisation step in the search path would paper over a broken index. |
| 4 | Chunk size stays under the model's token limit | `bge-small-en-v1.5` truncates at 512 tokens. Over that, text is dropped **silently** — no warning, no error, just a worse embedding. Current max is 407 tokens. |
| 5 | The LLM never generates a page number | It emits passage indices; pages come from chunk metadata. This makes a fabricated citation *unrepresentable* rather than merely unlikely. |
| 6 | The index is embedded by whichever library serves queries | Mixing sentence-transformers vectors with fastembed queries is a silent accuracy tax. |
| 7 | No ANN index on `chunks.embedding` | `ivfflat`/`hnsw` are approximate and would return different neighbours from the NumPy reference, with nothing in a diff to explain it. |

---

## 3. Ingestion: PDF to vectors

### 3.1 `pdf_parser.py` — text extraction and glyph repair

```python
extract_pages(pdf_path) -> [(page_number, text), ...]
```

Uses `pdfplumber`. Returns 1-based page numbers. Pages yielding fewer than
`MIN_CHARS_PER_PAGE` (100) characters are dropped — that threshold distinguishes a text
PDF from a scanned one. If *no* page has usable text, it raises rather than silently
indexing nothing.

**The interesting part is glyph repair.** The test book is typeset with Fourier and AMS
math fonts whose built-in Type 1 encodings pdfminer cannot resolve to Unicode. Every
square root, floor bracket and summation arrived as a literal `(cid:N)` marker —
**corrupting 110 of 680 chunks across 62 pages.**

All 36 distinct codes were enumerated from the PDF and identified from their font and
surrounding text, then mapped in `config.CID_MAP`:

```python
88:  "∑",   # sum
112: "√",   # radical
98:  "⌊",   # floor left
78:  "ℕ",   # blackboard N
```

**One case matters more than the rest.** `cid:54` is a negation slash drawn *over* the
following `=`, so the raw stream reads `x(cid:54)=0` and means `x ≠ 0`. Substituting the
code alone would produce `x = 0` — **an inverted meaning, silently, in text a model answers
from.** It is therefore handled as a pair:

```python
NEGATION_PAIR = ("(cid:54)=", "≠")
```

Unmapped codes are collected and warned about rather than passed through, so a new document
with different fonts reports what it lost.

### 3.2 `chunker.py` — splitting

```python
chunk_pages(pages, size=800, overlap=150) -> [{id, page, text}, ...]
```

Three-level split, most semantic first:

1. **Paragraph breaks** (`\n\s*\n`) — preferred
2. **Sentence breaks** (`(?<=[.!?])\s+`) — if a paragraph exceeds `size`
3. **Raw character slices** — last resort for a single enormous sentence

Units are then packed greedily into chunks of ~800 characters, each prefixed with the last
~150 characters of the previous chunk, cut at a word boundary.

**Why 800/150.** 800 characters is ~200 tokens, comfortably under the 512-token limit
(invariant 4), while large enough to contain a complete explanation. The 150-character
overlap means a fact split across a boundary still appears whole in one chunk.

**Why chunks never cross pages** — the loop is per-page, so this is structural rather than
enforced by a check. That is what makes invariant 2 free.

The overlap has a downstream consequence that shows up in quiz generation: consecutive
chunks *share text*, so sampling the first N chunks would produce near-duplicate questions.
See [§7.1](#71-sampling).

### 3.3 `embedder.py` — vectors

```python
embed_chunks(texts) -> (n, 384) float32     # no prefix
embed_query(text)  -> (384,)  float32       # QUERY_PREFIX applied
embed_passage(text)-> (384,)  float32       # no prefix — for HyDE
```

Model: **`BAAI/bge-small-en-v1.5`**, 384 dimensions, run through **fastembed / onnxruntime**.

Three functions rather than one because the prefix rule is subtle:

- `embed_query` prepends `"Represent this sentence for searching relevant passages: "`.
  bge models are trained with this instruction on the query side **only**.
- `embed_passage` does *not*, and exists specifically for HyDE. A hypothetical answer plays
  the role of a *document*, not a question — there is no question-shaped mismatch to correct.
- `embed_chunks` does not, obviously.

**A trap worth knowing.** fastembed exposes a `query_embed()` method. For this model it does
**not** apply the bge instruction — it is byte-identical to `embed()`. Trusting the name
would have silently dropped the prefix from every query and broken invariant 1 with no
error anywhere.

**On the migration from sentence-transformers.** The swap was made for deployment, not
quality: sentence-transformers pulls torch, putting the image at 1.58 GB, over a free
instance's budget. The two libraries are *not* bit-identical — ONNX and PyTorch differ by
up to **8.837e-04 per dimension** (~7,400× float32 eps) at a cosine of 0.999998. Small, but
enough to reorder near-ties, so the index was re-embedded and the entire eval re-run. Every
headline metric and every per-item rank came back unchanged; one already-failing item
reordered two of its wrong answers.

---

## 4. Storage: two backends, one interface

`stores.open_store(backend, document_id)` returns either. Both expose:

```python
store.chunks                        # list[dict], ordered so position == chunk["id"]
store.search(query_vec, k)          # [(score, chunk), ...]
store.dense_scores(query_vec)       # ndarray, aligned with chunks
store.dense_score_at(chunk_id, vec) # one chunk's cosine
```

### 4.1 `store.py` — the NumPy reference

`data/vectors.npy` + `data/chunks.json`. Search is one matrix-vector product:

```python
scores = vectors @ query_vec        # cosine, because both sides are unit length
top    = np.argsort(scores)[-k:][::-1]
```

Brute force over a few hundred rows is instant. This is the **reference implementation** —
Postgres is only correct insofar as it matches it.

One subtlety: `dense_score_at` is a single-row dot product rather than an index into
`dense_scores`. A full matrix-vector product accumulates in a different order and lands
~1e-8 away — enough to make the Postgres port *look* inexact when it is not.

### 4.2 `pgstore.py` — the Postgres port

Three decisions carry the exactness, each with a plausible alternative that would have
broken it quietly:

| decision | alternative | why it matters |
|---|---|---|
| `-(embedding <#> q)` — negative inner product | `<=>` cosine distance | Vectors are already normalised, so `<#>` reproduces `vectors @ query_vec` operation for operation. `<=>` recomputes norms and reintroduces drift. |
| No ANN index | `ivfflat` / `hnsw` | Approximate search returns *different* neighbours. Speed is irrelevant at this scale. |
| `vector(384)` is float4 | float8 | Matches the float32 the embedder emits, so storage is lossless rather than merely close. |

**Verified:** identical rank, RR and top-5 pages on all 20 eval questions for all three
retrievers, zero top-5 ordering mismatches, max score delta 2 units in the last place of
float32 (2.384e-07 against an eps of 1.192e-07).

**One naming detail that prevents a whole class of bug.** Chunk dicts carry `"id"` as the
*chunk_index* — the position within the document — not the database primary key. That means
retrievers, eval and quiz index them exactly as they do NumPy chunks. The real primary key
travels alongside as `"db_id"` and is used only for foreign keys.

---

## 5. Retrieval: three strategies

All three share one signature — `retrieve(question, store, k) -> [(score, chunk), ...]` —
so `eval.py` can swap between them without knowing which is which.

**The score means different things per retriever** (cosine for dense and HyDE, a fusion
score for hybrid), so it ranks *within* a retriever but does not compare *across* them.
That is why eval records `top1_dense` separately.

### 5.1 Dense

```python
store.search(embed_query(question), k)
```

The baseline. Everything else is measured against it.

### 5.2 Hybrid — BM25 + reciprocal rank fusion

`bm25.py` implements Okapi BM25 from scratch (`k1=1.5`, `b=0.75`):

```
idf(t)   = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
score(D) = Σ  idf(t) · f(t,D)·(k1+1) / (f(t,D) + k1·(1 - b + b·|D|/avgdl))
```

Postings are stored as parallel NumPy arrays so scoring is one vectorised update per query
term. Two deliberate simplifications, left visible rather than hidden:

- **No stemmer** — "trees" does not match "tree". A stemmer meant another dependency.
- **Query terms deduplicated** — the canonical formula sums over every occurrence, letting
  a repeated word double a document's score. For natural-language questions that is noise.

**Fusion is by rank, not score.** Cosine sits around 0.6–0.9 while BM25 is unbounded, so
they are not comparable. Reciprocal rank fusion sidesteps this entirely:

```python
rrf(scores) = 1 / (RRF_K + rank)        # RRF_K = 60
fused = (1 - w)·dense_rrf + w·keyword_rrf
```

Normalising scores per query was the alternative, and it would make every query's top hit
1.0 — destroying comparability *across* questions.

**Hybrid was measured and rejected.** See [§9.3](#93-the-recorded-negative-result).

### 5.3 HyDE — Hypothetical Document Embeddings

```python
passage = llm.generate(HYDE_PROMPT.format(question=question))
store.search(embed_passage(passage), k)
```

Ask the model to *write* a plausible answer, then search with **that** instead of the
question. The intuition: a question and its answer are written in different registers, and
a hypothetical answer is register-matched to the passage being looked for.

**Why it wins here.** Every dense failure in the eval set is a question that describes a
concept without naming it. `p04` asks about walking every connection in a network exactly
once and never says *Eulerian*. The generated passage opens *"A walk that traverses every
edge of a graph exactly once (an Eulerian trail)…"* — putting the missing word into the
search vector. The item moves from a miss to rank 1.

**The cost is symmetric**: HyDE writes words the question omitted, and also words the
question did not want. That is exactly why `p01` still fails — see [§9.4](#94-the-one-remaining-miss).

**Two caches, deliberately.** Generation varies run to run even at temperature 0, so
without caching the benchmark is not reproducible.

- `hyde_cache.json` — **committed**, so eval runs are deterministic
- `hyde_cache.local.json` — **gitignored**, for questions asked through the API

Reads check both, benchmark first. Writes only ever touch the runtime file, so live usage
can never contaminate the artefact that makes the benchmark reproducible. Both are
invalidated if the model or prompt changes.

---

## 6. Answering: grounding and citation

### 6.1 The prompt

`config.GROUNDING_PROMPT` numbers the retrieved passages and imposes five rules. Rule 4 is
the load-bearing one:

> Never write a page number, a section number, or a document title. Cite passage numbers only.

That is what makes invariant 5 true. The model's entire citation vocabulary is `[1]`–`[5]`.

### 6.2 Resolving citations

```python
CITATION_RE = re.compile(r"[\[【](\d+)[\]】]")
```

Both bracket forms are accepted because the model intermittently emits full-width `【5】`.
**Assuming ASCII silently discarded 8 of 14 citations during development** — the answers
looked fine, the citation count was just wrong.

Markers outside `1..len(chunks)` are **dropped rather than trusted**, and reported in
`dropped_citations`. This is detectable precisely because the vocabulary is bounded.

### 6.3 Refusal is a prompt, not a threshold

The obvious design — refuse when top similarity is low — **does not work**, and the eval set
shows why:

| | min | max |
|---|---|---|
| answerable | 0.568 | 0.888 |
| unanswerable | 0.574 | 0.675 |

**The lowest answerable question scores below every unanswerable one.** The distributions
are not merely overlapping but *interleaved*. No cut separates them; any threshold in that
band refuses real questions to no purpose.

This is a consequence of the eval set being adversarial: the unanswerable questions are
*in-domain* algorithms the book does not cover (Fibonacci heaps, red-black trees, simplex,
Karatsuba, treaps), not off-topic questions that would separate trivially.

So refusal moved into the prompt. Retrieval always returns the top 5 regardless of score,
and the model replies with the exact sentinel `INSUFFICIENT_CONTEXT` when they do not
contain an answer. A sentinel makes refusal *detectable*; prose refusals would have to be
guessed at.

`MIN_CONTEXT_SCORE = 0.40` survives, **reframed as a crash guard** against empty or nonsense
input so a garbage query does not burn an LLM call. It is not a relevance judgement, and it
has never fired on the eval set.

**One subtlety.** The guard is measured on `top1_dense` — the question-vs-chunk cosine —
not on the retriever's own score. HyDE ranks by the hypothetical, and those scores run
higher, so gating on them would quietly change what 0.40 means.

---

## 7. Quiz generation

### 7.1 Sampling

```python
picks = np.linspace(0, len(in_range) - 1, n).round().astype(int)
```

Evenly spread across the page range rather than the first N — because chunks carry 150
characters of overlap, so consecutive ones share text and would yield near-duplicate
questions.

### 7.2 Structural pre-filter

`usable(text)` runs **before** an LLM call is spent:

| check | threshold | rationale |
|---|---|---|
| length | ≥ 200 chars | fragments have nothing to ask about |
| dot-leader ratio | ≤ 0.15 | contents pages are 57% dot leaders |
| alphanumeric density | ≥ 0.45 | tables of numbers are not teachable |

**The thresholds sit in a measured gap.** Over the clean index the dot ratio tops out at
0.051 and alphanumeric density bottoms out at 0.530; contents-page chunks ran 0.318–0.356
and 0.243–0.311. Nothing lies between, so these reject every contents chunk and no real one.

This is a **backstop in front of** the model's own `NO_QUESTION` sentinel, not a replacement
for it — on the first run the sentinel let a contents-page question through.

### 7.3 Validation

`validate()` enforces the contract and raises `Rejected`, which feeds one retry then
drop-and-backfill. It checks:

- all required fields present and non-empty
- exactly 4 options, all distinct after normalisation
- `answer_index` in range
- **`evidence` is a verbatim span of the chunk** — the traceability check
- **no self-reference** to the source (`QUIZ_BANNED_REFERENCE`)

**On verbatim matching.** `normalise()` strips everything but alphanumerics and lowercases,
because pdfplumber runs words together (`Calculatingallthosevalues`) and a faithfully copied
span would not match raw.

**On the self-reference check.** `QUIZ_PROMPT` rule 8 asks the model not to mention the
passage. That reduced violations but did not eliminate them — one sampled run came back
clean at 0/5, the next produced *"as described in the passage"* at 1/3. A rule the model
follows *most* of the time needs a check that does not depend on it. The pattern is
deliberately narrow: bare "page" is allowed (a question about a page fault is legitimate);
what is rejected is self-reference and numbered citations like "page 96".

### 7.4 Option shuffling

The model has a **strong positional bias** — it put the correct answer at A in 4 of 5
questions on the first run, which a student can exploit without reading anything. Options
are shuffled with `random.Random(chunk["id"])` — seeded per chunk, so a regenerated quiz is
reproducible.

### 7.5 Traceability

`source_chunk_id` is **attached from metadata, never generated** — the model never sees a
chunk id, so a wrong one is unrepresentable. In Postgres it is a real foreign key, so the
database rejects a question citing a chunk that does not exist.

What the model *must* supply is `evidence`: a span copied from the passage, checked against
the chunk. That is what makes an answer traceable to its source.

---

## 8. The service layer

### 8.1 Endpoints

| route | notes |
|---|---|
| `POST /upload` | 202 + background task. Ingesting 296 pages takes minutes; holding the request open would die at any proxy. Deduplicated by SHA-256. |
| `GET /documents/{id}` | Progress. Without it, background work is unobservable. |
| `POST /ask` | `qa.answer` |
| `POST /quiz/generate` | `quiz.build`, persists questions |
| `POST /quiz/submit` | **The only logic no script had** |

### 8.2 Scoring is the server's

`/quiz/generate` returns `{position, question, options, page}` — **`answer_index` is
withheld**, because the client is about to be graded on it. `correct_index` reaches the
browser only in the `/quiz/submit` response. The client cannot mark its own work.

### 8.3 `SystemExit` is not an error

`llm.generate` and `db.pool()` exit the process on failure — right for a CLI. But
`SystemExit` is a `BaseException`, so it **bypasses FastAPI's handlers entirely** and an
exhausted quota reached the client as a bare 500 with no body. The `upstream()` context
manager translates it to a 503 carrying the cause.

---

## 9. Measurement: the eval harness

### 9.1 The set

20 hand-written questions: 15 answerable with expected pages, 5 unanswerable. Three
categories — `verbatim` (5), `paraphrase` (7), `multi_page` (3).

`sections.py` is the reference for `expected_pages`: the book's *printed* page numbers are
offset from the PDF page numbers the index stores, and the two are not interchangeable.

### 9.2 Metrics

- **recall@5** — is any expected page in the top 5?
- **recall@1** — is the top hit an expected page?
- **MRR** — mean of 1/rank
- **multi_page strict** — a hit requires a chunk from *every* cluster the answer needs, not
  just one. Reported because the lenient number flatters items with wide page lists.

`top1_dense` is recorded separately for every retriever — the same measurement regardless of
what the retriever ranked by — so results stay comparable across variants.

### 9.3 The recorded negative result

Hybrid was swept across blend weights 0.0–1.0 and **not adopted**:

**No weight beats dense on any metric.** recall@5 never exceeds its 0.800 baseline, recall@1
never exceeds 0.600, paraphrase recall@1 never exceeds 0.286. At w=0.1 hybrid is
indistinguishable from dense — it does not degrade gracefully toward being useful, it
converges on being dense with extra machinery.

The trade is **structural, not a tuning artefact**: the items BM25 helps (p01, p04) and the
ones it harms (p03, p05, p06) are disjoint, and the harmed ones already sit at rank 1 where
keyword influence can only push them down.

**p07 is never retrieved at any weight, pure BM25 included.** Queried directly with
*"amortized analysis"*, BM25 returns page 87 — exactly right. But p07 asks *"why does an
operation that occasionally does a lot of work still count as cheap when you average it
out?"* and shares no vocabulary with its target. Keyword scoring cannot reach a passage it
has no term in common with, however weighted. That is the gap HyDE closes and BM25
structurally cannot.

### 9.4 The one remaining miss

`p01` asks how to precompute repeated range sums. The answer is the prefix sum array on
page 94; HyDE returns page 96, where the binary indexed tree begins.

**This is not a labelling error.** Both questionable items were audited against the book and
their expected pages left unchanged. The BIT is the *dynamic* variant, for when values change
between queries — which p01 never asks about.

The cause is inherent to the method: a hypothetical answer about precomputing range sums
naturally describes both the static prefix-sum array and the dynamic structure that
generalises it, so it carries the wrong one's vocabulary into the search vector.

**Final numbers stand at recall@5 0.933 and recall@1 0.867, with p01 counted as a genuine
miss rather than relabelled away.**

---

## 10. Deployment

| | | why |
|---|---|---|
| API | Render, Docker | `render.yaml`; builds with `BAKE_MODEL=true` |
| Frontend | Vercel | `vercel.json` rewrites `/api/*` to the Render origin |
| Postgres | Neon | Render's free Postgres has no `pgvector` |

**The Vercel rewrite is what keeps `api.py` free of CORS middleware** — the browser only ever
sees a same-origin `/api` path, in production exactly as in development.

**`BAKE_MODEL`** defaults to `true`: a host with an ephemeral filesystem re-downloads a
volume-cached model on every cold start, and a demo that stalls 40s on first click reads as
broken. `docker-compose.yml` passes `false`, because there a named volume persists and a
smaller image is worth more.

---

## 11. Bugs worth knowing about

Every one of these was invisible locally and appeared only when the scripts were put behind
a service. This section is the most interview-useful part of the document.

**1. A module-level store assigned per request.** `/ask` resolved the caller's document then
assigned it to a global. FastAPI runs sync handlers in a threadpool, so two overlapping
requests shared it — the second overwrote the global before the first read it, and a
question about one document was answered *and cited* from another. The SQL had been filtered
by `document_id` the whole time; the plumbing choosing *which* document was not.

**2. A globally cached BM25 index.** Built once from whichever document was indexed first, so
with more than one document every keyword query would score against the wrong vocabulary.
Latent rather than live — `RETRIEVER` is HyDE, which never touches it — but waiting for a
config change to become real.

**3. `config.STORE` resolving to the most recent upload.** `qa.py` opened its store through a
lazy singleton with no arguments. With `STORE=postgres` set, regenerating `qa_run.json` would
have scored all 20 eval questions against a 40-chunk Bluetooth manual — and written a file
indistinguishable from a valid run. Fixed by removing the fallback entirely: `answer()` now
requires `store` as a keyword argument.

**4. The schema could not bootstrap itself.** `apply_schema()` went through the connection
pool, and the pool configures every connection with `register_vector`, which fails until
`CREATE EXTENSION vector` has run — the first line of `schema.sql`. Local Docker hid it
completely: compose mounts the schema into `/docker-entrypoint-initdb.d/`, so Postgres
applies it during volume init and the extension always exists before the app connects. A
managed database has no such hook, and the first deploy against an empty Neon instance failed
with a `PoolTimeout` that named nothing useful.

**5. Truncation that moved no metric.** At `LLM_MAX_TOKENS=512`, one answer stopped
mid-sentence. Every metric still passed — it cited a page earlier in the answer, so `uncited`
stayed `False`. **Truncation degraded the text without moving a single number.** It was found
by reading the answers, not by measuring them.

**6. Front matter skipped by a fixed page count.** Ingestion skipped a prefix derived from
one book's layout. Applied to a 16-page upload whose page 1 was already body text, it
discarded pages 1–10 — two thirds of the document, missing with no error.

**The pattern across all six:** single-document, single-threaded, fail-by-exiting, and
"whatever happened last" are assumptions a script can hold silently and a service cannot.
The retrieval maths needed nothing.

---

## 12. If someone asks you to defend it

**"Why not LangChain?"** The retrieval layer is the part of this project worth being able to
explain. Chunking, BM25, rank fusion and the citation contract are 560 lines total, and
every one of them is a decision I can point at a measurement for.

**"Why no vector database?"** 673 chunks. A sequential scan is instant, and an ANN index
would make results *approximate* — which would break the guarantee that the Postgres port
returns exactly what the NumPy reference does.

**"Why is hybrid retrieval in the repo if you don't use it?"** Because the negative result is
the result. It was swept across 11 weights, beat dense at none of them, and the reason is
structural rather than a tuning failure. Deleting it would leave the README claiming HyDE is
best without evidence that the obvious alternative was tried.

**"How do you know the citations are right?"** The model cannot emit a page number — it emits
passage indices, bounded by the number of passages it was given, and out-of-range markers are
dropped and counted. Pages are looked up from chunk metadata. A wrong page is not unlikely,
it is unrepresentable.

**"How do you know retrieval didn't regress?"** Every retrieval change is measured against
the eval set before it is kept, and every run is committed to `eval_runs/`. The fastembed
migration is the clearest example: the vectors changed by 8.8e-04 per dimension, so the index
was rebuilt and all three retrievers re-run rather than assumed to carry over.

**"What's still broken?"** `p01`, one retrieval miss with a documented and inherent cause.
Front matter is indexed rather than detected. BM25 has no stemmer. The frontend has never
been visually verified. The eval set is 20 questions written by one person — enough to catch
the failures documented here, not enough to claim a number generalises.
