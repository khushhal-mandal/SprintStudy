"""Sweep the hybrid blend weight and record the curve.

A recorded negative result rather than a tuning tool: no weight beats dense on
this eval set, and the README says so. It lives in the repo because a claim that
strong needs to be reproducible, not because the weight is meant to be tuned.

Needs no LLM - dense scores plus BM25 - so it runs whatever the token quota is
doing.

usage: python sweep.py
"""

import json

import numpy as np

import retrievers
from config import EVAL_K, EVAL_PATH, EVAL_RUNS_DIR
from embedder import embed_query
from stores import open_store

WEIGHTS = [round(0.1 * i, 1) for i in range(11)]
CATEGORIES = ("verbatim", "paraphrase", "multi_page")


def main() -> None:
    store = open_store("numpy")
    items = [i for i in json.loads(EVAL_PATH.read_text(encoding="utf-8")) if i["answerable"]]

    # Rank vectors once per question; varying w is then pure arithmetic, so the
    # sweep measures the blend rather than re-measuring the retrievers.
    prepared = []
    for item in items:
        dense = retrievers._rrf(store.dense_scores(embed_query(item["question"])))
        keyword = retrievers._rrf(retrievers._get_bm25(store).scores(item["question"]))
        prepared.append((item, dense, keyword))

    rows = []
    for w in WEIGHTS:
        ranks = {}
        for item, dense, keyword in prepared:
            fused = (1 - w) * dense + w * keyword
            top = np.argsort(fused)[-EVAL_K:][::-1]
            pages = [store.chunks[i]["page"] for i in top]
            expected = set(item["expected_pages"])
            ranks[item["id"]] = next(
                (i for i, p in enumerate(pages, start=1) if p in expected), None
            )

        n = len(items)
        row = {
            "w": w,
            f"recall@{EVAL_K}": sum(1 for r in ranks.values() if r) / n,
            "recall@1": sum(1 for r in ranks.values() if r == 1) / n,
            "mrr": sum(1 / r for r in ranks.values() if r) / n,
            "by_category": {},
            "ranks": ranks,
        }
        for cat in CATEGORIES:
            ids = [i["id"] for i in items if i["category"] == cat]
            row["by_category"][cat] = {
                f"recall@{EVAL_K}": sum(1 for i in ids if ranks[i]) / len(ids),
                "recall@1": sum(1 for i in ids if ranks[i] == 1) / len(ids),
            }
        rows.append(row)

    print(f"\nhybrid blend weight sweep, K={EVAL_K}, {len(items)} answerable items")
    print("w = keyword's share of the fused rank score\n")
    print(f"  {'w':>4} {'recall@5':>9} {'recall@1':>9} {'MRR':>7} {'para r@5':>9} {'para r@1':>9}")
    print(f"  {'-' * 52}")
    for row in rows:
        note = {0.0: "  <- pure dense", 0.5: "  <- committed", 1.0: "  <- pure BM25"}.get(row["w"], "")
        para = row["by_category"]["paraphrase"]
        print(
            f"  {row['w']:>4.1f} {row[f'recall@{EVAL_K}']:>9.3f} {row['recall@1']:>9.3f} "
            f"{row['mrr']:>7.3f} {para[f'recall@{EVAL_K}']:>9.3f} {para['recall@1']:>9.3f}{note}"
        )

    print(f"\n  per-question rank by weight (- = not in top {EVAL_K})\n")
    print(f"  {'id':<5}" + "".join(f"{w:>6.1f}" for w in WEIGHTS))
    print(f"  {'-' * (5 + 6 * len(WEIGHTS))}")
    never = []
    for item in items:
        cells = "".join(
            f"{(str(r['ranks'][item['id']]) if r['ranks'][item['id']] else '-'):>6}" for r in rows
        )
        print(f"  {item['id']:<5}{cells}")
        if all(r["ranks"][item["id"]] is None for r in rows):
            never.append(item["id"])

    print(f"\n  never retrieved at any weight: {', '.join(never) if never else 'none'}")

    EVAL_RUNS_DIR.mkdir(exist_ok=True)
    path = EVAL_RUNS_DIR / "hybrid-sweep.json"
    path.write_text(
        json.dumps(
            {
                "k": EVAL_K,
                "chunks": len(store.chunks),
                "items": len(items),
                "never_retrieved": never,
                "sweep": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n  saved to {path}\n")


if __name__ == "__main__":
    main()
