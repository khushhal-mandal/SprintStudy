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
import store
from config import (
    ANSWER_K,
    EVAL_PATH,
    GROUNDING_PROMPT,
    MIN_CONTEXT_SCORE,
    REFUSAL_MESSAGE,
    REFUSAL_SENTINEL,
)
from embedder import embed_query

CITATION_RE = re.compile(r"\[(\d+)\]")

_index = None


def _load_index():
    """chunks + vectors, loaded once and reused across questions."""
    global _index
    if _index is None:
        _index = store.load()
    return _index


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


def answer(question: str, k: int = ANSWER_K) -> dict:
    chunks, vectors = _load_index()
    hits = store.search(embed_query(question), chunks, vectors, k)

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
    }

    if hits[0][0] < MIN_CONTEXT_SCORE:
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
        print(f"  (refused: {reason}; top-1 {result['top1_score']:.3f})")
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
    """Every eval question through the full pipeline. Milestone 3's acceptance test."""
    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    print(f"\n{len(items)} questions through the full pipeline\n")
    print(f"  {'id':<5} {'category':<12} {'top1':>6} {'outcome':<10} pages cited")
    print(f"  {'-' * 62}")

    results = []
    for item in items:
        r = answer(item["question"])
        results.append((item, r))

        if r["answered"]:
            outcome = "answered"
            detail = ",".join(str(c["page"]) for c in r["citations"]) or "NONE CITED"
        else:
            outcome = "refused"
            detail = f"({r['refusal_reason']})"

        print(
            f"  {item['id']:<5} {item['category']:<12} {r['top1_score']:>6.3f} "
            f"{outcome:<10} {detail}"
        )

    unanswerable = [(i, r) for i, r in results if not i["answerable"]]
    answerable = [(i, r) for i, r in results if i["answerable"]]
    refused_bad = sum(1 for _, r in unanswerable if not r["answered"])
    answered_ok = sum(1 for _, r in answerable if r["answered"])
    prefiltered = sum(1 for _, r in results if r["refusal_reason"] == "pre_filter")
    uncited = sum(1 for _, r in results if r["uncited"])

    print(f"\n  unanswerable refused   {refused_bad}/{len(unanswerable)}")
    print(f"  answerable answered    {answered_ok}/{len(answerable)}")
    print(f"  pre-filter fired       {prefiltered}  (expected 0)")
    print(f"  answered without cite  {uncited}")

    leaked = [
        i["id"]
        for i, r in results
        if r["answered"] and re.search(r"\bpage\s+\d+", r["answer"], re.I)
    ]
    print(f"  wrote a page number    {len(leaked)}  {leaked or ''}")
    print()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        run_all()
    elif len(sys.argv) == 2:
        print_result(answer(sys.argv[1]))
        print()
    else:
        sys.exit('usage: python qa.py "your question"  |  python qa.py --all')
