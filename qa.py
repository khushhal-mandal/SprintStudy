"""Grounded Q&A: retrieve, answer from the passages only, cite by page.

Refusal is the model's job, decided in the grounding prompt from the passages
it was given. Retrieval always returns the top K regardless of score, because
top-1 similarity does not separate answerable from unanswerable questions on
the eval set - the two distributions overlap.

The LLM never emits a page number. It cites passage indices, which are looked
up against chunk metadata on the way out, so a fabricated page is not merely
unlikely but unrepresentable.

usage: python qa.py "your question"
       python qa.py --all          score every question in eval.json
"""

import json
import re
import sys

import llm
from config import (
    ANSWER_K,
    EVAL_PATH,
    GROQ_MODEL,
    GROUNDING_PROMPT,
    MIN_CONTEXT_SCORE,
    QA_RUN_PATH,
    RETRIEVER,
    REFUSAL_MESSAGE,
    REFUSAL_SENTINEL,
)
from embedder import embed_query
from retrievers import RETRIEVERS
from stores import open_store

# The model intermittently emits full-width 【5】 instead of [5] - same
# citation, different bracket - so both forms are accepted. A mixed pair
# like [5】 is tolerated rather than treated as a separate case.
CITATION_RE = re.compile(r"[\[【](\d+)[\]】]")

_store = None


def _load_index():
    """The configured store, opened once and reused across questions.

    Only for the CLI, which is single-threaded and serves one document. Callers
    that already know their store pass it to answer() instead - see the note
    there.
    """
    global _store
    if _store is None:
        _store = open_store()
    return _store


def build_prompt(question: str, chunks: list[dict]) -> str:
    passages = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))
    return GROUNDING_PROMPT.format(
        sentinel=REFUSAL_SENTINEL, passages=passages, question=question
    )


def parse_citations(answer: str, chunks: list[dict]) -> tuple[list[dict], list[int]]:
    """Map [n] markers to chunk metadata.

    Returns (citations, dropped). A marker outside 1..len(chunks) is dropped
    rather than trusted - it is the one way this can go wrong, and it is
    detectable because the model's whole citation vocabulary is bounded.
    """
    citations: list[dict] = []
    dropped: list[int] = []

    for marker in dict.fromkeys(int(m) for m in CITATION_RE.findall(answer)):
        if 1 <= marker <= len(chunks):
            chunk = chunks[marker - 1]
            citations.append(
                {"marker": marker, "page": chunk["page"], "chunk_id": chunk["id"]}
            )
        else:
            dropped.append(marker)

    return citations, dropped


def answer(question: str, k: int = ANSWER_K, store=None) -> dict:
    """Answer from one document's chunks.

    `store` is an argument rather than module state because the API serves
    requests concurrently. Assigning a module-level store per request let two
    overlapping requests interleave: the second overwrote the global before the
    first read it, so a question about one document was answered - and cited -
    from another. Passing it down keeps each request on its own document.
    """
    store = store if store is not None else _load_index()
    hits = RETRIEVERS[RETRIEVER](question, store, k)

    # MIN_CONTEXT_SCORE is calibrated on question-vs-chunk cosine: the dense
    # floor over the eval set is 0.623. HyDE ranks by the hypothetical instead,
    # and those scores run higher (floor 0.701), so gating on the retriever's
    # own score would quietly change what 0.55 means. Measure the guard on the
    # quantity it was calibrated on, whichever retriever produced the hits.
    top1_dense = store.dense_score_at(hits[0][1]["id"], embed_query(question))

    retrieved = [
        {"rank": i, "score": score, "page": c["page"], "chunk_id": c["id"]}
        for i, (score, c) in enumerate(hits, start=1)
    ]
    result = {
        "question": question,
        "answered": False,
        "answer": None,
        "refusal_reason": None,
        "citations": [],
        "dropped_citations": [],
        "uncited": False,
        "retrieved": retrieved,
        "top1_score": hits[0][0],
        "top1_dense": top1_dense,
    }

    if top1_dense < MIN_CONTEXT_SCORE:
        result["refusal_reason"] = "pre_filter"
        return result

    raw = llm.generate(build_prompt(question, [c for _, c in hits]))

    if REFUSAL_SENTINEL in raw:
        result["refusal_reason"] = "model"
        return result

    citations, dropped = parse_citations(raw, [c for _, c in hits])
    result.update(
        answered=True,
        answer=raw,
        citations=citations,
        dropped_citations=dropped,
        uncited=not citations,
    )
    return result


def print_result(result: dict) -> None:
    print(f'\nQ: {result["question"]}')

    if not result["answered"]:
        reason = {
            "pre_filter": f"below the {MIN_CONTEXT_SCORE} context floor, no LLM call",
            "model": "model found no answer in the passages",
        }[result["refusal_reason"]]
        print(f"\n{REFUSAL_MESSAGE}")
        print(f"  (refused: {reason}; top-1 dense {result['top1_dense']:.3f})")
        return

    print(f"\n{result['answer']}\n")

    if result["citations"]:
        print("Sources")
        for c in result["citations"]:
            print(f"  [{c['marker']}] page {c['page']:<4} chunk #{c['chunk_id']}")
    else:
        print("  warning: answered without citing any passage")

    if result["dropped_citations"]:
        print(f"  warning: dropped out-of-range markers {result['dropped_citations']}")


def run_all() -> None:
    """Every eval question through the full pipeline. Milestone 3's acceptance test.

    Writes the run to QA_RUN_PATH so the numbers in the repo are a run that
    happened, not a claim. Model output is non-deterministic even at
    temperature 0, so this file is expected to change between runs.
    """
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    chunks = _load_index().chunks

    print(f"\n{len(items)} questions through the full pipeline\n")
    print(f"  {'id':<5} {'category':<12} {'dense':>6} {'outcome':<10} {'cites':>5}  pages cited")
    print(f"  {'-' * 68}")

    records = []
    for item in items:
        r = answer(item["question"])
        leaked = bool(r["answered"] and re.search(r"\bpage\s+\d+", r["answer"], re.I))
        pages = [c["page"] for c in r["citations"]]

        records.append(
            {
                "id": item["id"],
                "category": item["category"],
                "answerable": item["answerable"],
                "top1_score": round(r["top1_score"], 4),
                "top1_dense": round(r["top1_dense"], 4),
                "answered": r["answered"],
                "refusal_reason": r["refusal_reason"],
                "citation_count": len(r["citations"]),
                "cited_pages": pages,
                "dropped_citations": r["dropped_citations"],
                "uncited": r["uncited"],
                "wrote_page_number": leaked,
                "answer": r["answer"],
            }
        )

        if r["answered"]:
            outcome, detail = "answered", ",".join(map(str, pages)) or "NONE CITED"
            cites = str(len(r["citations"]))
        else:
            outcome, detail, cites = "refused", f"({r['refusal_reason']})", "-"

        print(
            f"  {item['id']:<5} {item['category']:<12} {r['top1_dense']:>6.3f} "
            f"{outcome:<10} {cites:>5}  {detail}"
        )

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    summary = {
        "items": len(records),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "unanswerable_refused": sum(1 for r in unanswerable if not r["answered"]),
        "answerable_answered": sum(1 for r in answerable if r["answered"]),
        "pre_filter_fired": sum(1 for r in records if r["refusal_reason"] == "pre_filter"),
        "citations_total": sum(r["citation_count"] for r in records),
        "answered_without_citation": sum(1 for r in records if r["uncited"]),
        "citations_dropped": sum(len(r["dropped_citations"]) for r in records),
        "page_number_leaks": sum(1 for r in records if r["wrote_page_number"]),
    }

    QA_RUN_PATH.write_text(
        json.dumps(
            {
                "model": GROQ_MODEL,
                "retriever": RETRIEVER,
                "answer_k": ANSWER_K,
                "min_context_score": MIN_CONTEXT_SCORE,
                "chunks": len(chunks),
                "summary": summary,
                "results": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n  unanswerable refused      {summary['unanswerable_refused']}/{len(unanswerable)}")
    print(f"  answerable answered       {summary['answerable_answered']}/{len(answerable)}")
    print(f"  pre-filter fired          {summary['pre_filter_fired']}  (expected 0)")
    print(f"  citations attached        {summary['citations_total']}")
    print(f"  answered without cite     {summary['answered_without_citation']}")
    print(f"  citations dropped         {summary['citations_dropped']}")
    print(f"  wrote a page number       {summary['page_number_leaks']}")
    print(f"\n  run written to {QA_RUN_PATH.name}\n")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        run_all()
    elif len(sys.argv) == 2:
        print_result(answer(sys.argv[1]))
        print()
    else:
        sys.exit('usage: python qa.py "your question"  |  python qa.py --all')
