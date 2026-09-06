"""Measure retrieval against the eval set.

Reports recall@K, recall@1 and MRR over the answerable items, a stricter
recall for multi_page items that requires every cluster to be represented, and
the top-1 similarity distribution for answerable vs unanswerable items.

Those last two distributions overlap on this eval set, which is why refusal
lives in the grounding prompt rather than in a cut on the similarity score.

usage: python eval.py [--retriever dense|hybrid|hyde] [eval.json]
       python eval.py --compare
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from config import EVAL_DEFAULT_RETRIEVER, EVAL_K, EVAL_PATH, EVAL_RUNS_DIR
from stores import open_store
from embedder import embed_query
from retrievers import RETRIEVERS

CATEGORIES = ("verbatim", "paraphrase", "multi_page", "unanswerable")

SCHEMA_HELP = """
Create it as a JSON list. Schema per item:

  {
    "id": "q01",
    "question": "...",
    "expected_pages": [99],
    "answerable": true,
    "category": "verbatim"
  }

expected_pages are PDF page numbers - run `python sections.py` for the mapping
from printed page to PDF page.

category is one of: verbatim | paraphrase | multi_page | unanswerable
Unanswerable items take "expected_pages": [], "answerable": false.

multi_page items also need "clusters": a list of page lists partitioning
expected_pages into the distinct parts of the document the answer needs, e.g.
"clusters": [[96, 97, 98], [99, 100, 101, 102]]
"""


def load_items(path: Path, indexed_pages: set[int]) -> list[dict]:
    """Parse and validate the eval set. Exits on any schema error."""
    if not path.exists():
        raise SystemExit(f"No eval set at {path}.\n{SCHEMA_HELP}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}")

    if not isinstance(raw, list):
        raise SystemExit(f"{path} must contain a JSON list.\n{SCHEMA_HELP}")
    if not raw:
        raise SystemExit(f"{path} is empty.\n{SCHEMA_HELP}")

    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for i, item in enumerate(raw):
        where = f"item {i}"
        if not isinstance(item, dict):
            errors.append(f"{where}: not an object")
            continue

        item_id = item.get("id")
        where = f"item {i} (id={item_id!r})"
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{where}: 'id' must be a non-empty string")
        elif item_id in seen:
            errors.append(f"{where}: duplicate id")
        else:
            seen.add(item_id)

        if not isinstance(item.get("question"), str) or not item["question"].strip():
            errors.append(f"{where}: 'question' must be a non-empty string")

        answerable = item.get("answerable")
        if not isinstance(answerable, bool):
            errors.append(f"{where}: 'answerable' must be true or false")

        category = item.get("category")
        if category not in CATEGORIES:
            errors.append(f"{where}: 'category' must be one of {' | '.join(CATEGORIES)}")

        pages = item.get("expected_pages")
        if not isinstance(pages, list) or not all(
            isinstance(p, int) and not isinstance(p, bool) for p in pages
        ):
            errors.append(f"{where}: 'expected_pages' must be a list of ints")
            pages = []

        # Cross-field consistency. These are the mistakes that silently produce
        # meaningless numbers rather than crashing.
        if answerable is True and not pages:
            errors.append(f"{where}: answerable item needs at least one expected page")
        if answerable is False and pages:
            errors.append(f"{where}: unanswerable item must have empty expected_pages")
        if (category == "unanswerable") != (answerable is False):
            errors.append(f"{where}: category 'unanswerable' and answerable=false must agree")
        if category == "multi_page" and len(pages) < 2:
            warnings.append(f"{where}: category is multi_page but only {len(pages)} page listed")

        # Clusters carve expected_pages into the distinct parts of the document
        # an answer needs. They cannot be derived - m01's two halves sit in
        # different chapters, m03's sit in adjacent sections of the same one -
        # so they are declared, and required, rather than guessed at.
        clusters = item.get("clusters")
        if category == "multi_page":
            if clusters is None:
                errors.append(f"{where}: multi_page item needs a 'clusters' field")
            elif (
                not isinstance(clusters, list)
                or len(clusters) < 2
                or not all(
                    isinstance(c, list)
                    and c
                    and all(isinstance(p, int) and not isinstance(p, bool) for p in c)
                    for c in clusters
                )
            ):
                errors.append(f"{where}: 'clusters' must be 2+ non-empty lists of ints")
            else:
                union = {p for c in clusters for p in c}
                if union != set(pages):
                    missing = sorted(set(pages) - union)
                    extra = sorted(union - set(pages))
                    errors.append(
                        f"{where}: clusters must partition expected_pages exactly "
                        f"(missing from clusters: {missing or 'none'}, "
                        f"not in expected_pages: {extra or 'none'})"
                    )
        elif clusters is not None:
            errors.append(f"{where}: 'clusters' is only valid on multi_page items")

        for p in pages:
            if p not in indexed_pages:
                errors.append(
                    f"{where}: page {p} is not in the index "
                    "(out of range, or no extractable text - recall for it would always be 0)"
                )

    if warnings:
        print("warnings:")
        for w in warnings:
            print(f"  {w}")
        print()

    if errors:
        sys.stdout.flush()  # keep warnings above errors when the two streams merge
        print(f"{len(errors)} problem(s) in {path}:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)

    return raw


def run(items: list[dict], store, k: int, retrieve) -> list[dict]:
    """Retrieve for every item with the given retriever and score it."""
    rows = []
    for item in items:
        hits = retrieve(item["question"], store, k)
        pages = [c["page"] for _, c in hits]
        expected = set(item["expected_pages"])
        rank = next((i for i, p in enumerate(pages, start=1) if p in expected), None)

        # top1_score is whatever the retriever ranked by: a cosine for dense, an
        # RRF score for hybrid, and for HyDE a cosine against the hypothetical
        # rather than the question. top1_dense is the same measurement for every
        # retriever - how close the winning chunk is to the original question -
        # so it stays comparable across variants and back to the baseline.
        # embed_query applies QUERY_PREFIX; embed_chunks does not.
        top1_dense = store.dense_score_at(hits[0][1]["id"], embed_query(item["question"]))
        rows.append(
            {
                **item,
                "top_pages": pages,
                "top1_score": hits[0][0],
                "top1_dense": top1_dense,
                "rank": rank,
                "rr": 1.0 / rank if rank else 0.0,
            }
        )
    return rows


def print_table(rows: list[dict], k: int) -> None:
    print(f"\nper-question results (K={k})\n")

    expected = {
        r["id"]: ",".join(str(p) for p in r["expected_pages"]) if r["answerable"] else "-"
        for r in rows
    }
    pages = {r["id"]: " ".join(str(p) for p in r["top_pages"]) for r in rows}
    ew = max(len("expected"), *(len(s) for s in expected.values()))
    pw = max(len(f"top-{k} pages"), *(len(s) for s in pages.values()))

    head = (
        f"  {'':1} {'id':<5} {'category':<12} {'expected':<{ew}} "
        f"{'top-' + str(k) + ' pages':<{pw}} {'rank':>4} {'RR':>5} {'dense':>6}"
    )
    print(head)
    print(f"  {'-' * (len(head) - 2)}")

    for r in rows:
        if r["answerable"]:
            mark = "." if r["rank"] else "X"
            rank = str(r["rank"]) if r["rank"] else "-"
            rr = f"{r['rr']:.3f}"
        else:
            mark, rank, rr = " ", "-", "-"

        print(
            f"  {mark:1} {r['id']:<5} {r['category']:<12} {expected[r['id']]:<{ew}} "
            f"{pages[r['id']]:<{pw}} {rank:>4} {rr:>5} {r['top1_dense']:>6.3f}"
        )
    print("\n  X = no expected page in the top K")


def print_metrics(rows: list[dict], k: int) -> None:
    answerable = [r for r in rows if r["answerable"]]
    if not answerable:
        print("\nno answerable items - skipping retrieval metrics")
        return

    n = len(answerable)
    at_k = sum(1 for r in answerable if r["rank"])
    at_1 = sum(1 for r in answerable if r["rank"] == 1)
    mrr = sum(r["rr"] for r in answerable) / n

    print(f"\nretrieval over {n} answerable items")
    print(f"  recall@{k}   {at_k / n:.3f}  ({at_k}/{n})")
    print(f"  recall@1   {at_1 / n:.3f}  ({at_1}/{n})")
    print(f"  MRR@{k}      {mrr:.3f}")

    present = [c for c in CATEGORIES if any(r["category"] == c for r in answerable)]
    if len(present) > 1:
        print(f"\n  {'category':<12} {'n':>3} {'recall@' + str(k):>9} {'recall@1':>9} {'MRR':>6}")
        for cat in present:
            group = [r for r in answerable if r["category"] == cat]
            g = len(group)
            print(
                f"  {cat:<12} {g:>3} "
                f"{sum(1 for r in group if r['rank']) / g:>9.3f} "
                f"{sum(1 for r in group if r['rank'] == 1) / g:>9.3f} "
                f"{sum(r['rr'] for r in group) / g:>6.3f}"
            )


def print_strict_multi_page(rows: list[dict], k: int) -> None:
    """Score multi_page items so a hit needs every cluster represented.

    The lenient number counts a hit on any single matching page, which a wide
    expected_pages list makes easy. This asks what the category was written to
    ask: did retrieval surface all the distinct parts the answer draws on?

    There is no strict recall@1 - with two or more clusters it is 0 by
    construction, so it would report nothing.
    """
    items = [r for r in rows if r["category"] == "multi_page"]
    if not items:
        return

    print(f"\nmulti_page strict recall@{k} (a hit needs every cluster present)")
    hits = 0
    for r in items:
        top = set(r["top_pages"])
        n = len(r["clusters"])
        covered = sum(1 for c in r["clusters"] if top & set(c))
        hits += covered == n
        print(
            f"  {r['id']:<5} {n} clusters   {covered}/{n} covered   "
            f"{'HIT' if covered == n else 'MISS'}"
        )

    lenient = sum(1 for r in items if r["rank"])
    total = len(items)
    print(
        f"  strict {hits / total:.3f} ({hits}/{total})   "
        f"lenient {lenient / total:.3f} ({lenient}/{total})"
    )


def _stats(scores: list[float]) -> str:
    a = np.array(scores)
    return (
        f"{len(a):>3} {a.min():>7.3f} {np.percentile(a, 25):>7.3f} "
        f"{np.median(a):>7.3f} {a.mean():>7.3f} {np.percentile(a, 75):>7.3f} {a.max():>7.3f}"
    )


def print_separation(rows: list[dict]) -> None:
    yes = [r["top1_dense"] for r in rows if r["answerable"]]
    no = [r["top1_dense"] for r in rows if not r["answerable"]]

    print("\ntop-1 dense similarity distribution (question vs winning chunk)")
    print(f"  {'':<13}{'n':>3} {'min':>7} {'p25':>7} {'med':>7} {'mean':>7} {'p75':>7} {'max':>7}")
    if yes:
        print(f"  {'answerable':<13}{_stats(yes)}")
    if no:
        print(f"  {'unanswerable':<13}{_stats(no)}")

    if not (yes and no):
        print("\n  need both answerable and unanswerable items to measure the gap")
        return

    lo, hi = min(yes), max(no)
    if lo > hi:
        print(
            f"\n  clean separation: lowest answerable {lo:.3f} > "
            f"highest unanswerable {hi:.3f}  (gap {lo - hi:+.3f})"
        )
    else:
        overlap = sum(1 for s in yes if s <= hi) + sum(1 for s in no if s >= lo)
        print(
            f"\n  OVERLAP: lowest answerable {lo:.3f} <= highest unanswerable {hi:.3f} "
            f"({overlap} items in the overlap band)"
        )
        print("  top-1 score is not a usable refusal signal on its own, which is why")
        print("  refusal is the grounding prompt's job rather than a cut on this number")


def summarise(rows: list[dict], k: int) -> dict:
    """Headline and per-category metrics, for saving and for --compare."""
    answerable = [r for r in rows if r["answerable"]]
    n = len(answerable)

    def block(group):
        g = len(group)
        if not g:
            return None
        return {
            "n": g,
            f"recall@{k}": sum(1 for r in group if r["rank"]) / g,
            "recall@1": sum(1 for r in group if r["rank"] == 1) / g,
            "mrr": sum(r["rr"] for r in group) / g,
        }

    strict = [r for r in rows if r["category"] == "multi_page"]
    strict_hits = sum(
        1
        for r in strict
        if all(set(r["top_pages"]) & set(c) for c in r["clusters"])
    )

    return {
        "overall": block(answerable),
        "by_category": {
            cat: block([r for r in answerable if r["category"] == cat])
            for cat in CATEGORIES
            if cat != "unanswerable"
            and any(r["category"] == cat for r in answerable)
        },
        "multi_page_strict": strict_hits / len(strict) if strict else None,
        "answerable_top1_dense_mean": (
            sum(r["top1_dense"] for r in answerable) / n if n else None
        ),
        "unanswerable_top1_dense_mean": (
            sum(r["top1_dense"] for r in rows if not r["answerable"])
            / max(1, len(rows) - n)
        ),
    }


def save_run(name: str, rows: list[dict], k: int, backend: str = "numpy") -> Path:
    """One file per retriever+backend.

    The backend is in the filename because a Postgres run and a NumPy run of the
    same retriever are different measurements - writing both to one path silently
    replaced the reference numbers the port is checked against.
    """
    EVAL_RUNS_DIR.mkdir(exist_ok=True)
    stem = name if backend == "numpy" else f"{name}-{backend}"
    path = EVAL_RUNS_DIR / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "retriever": name,
                "store": backend,
                "k": k,
                "summary": summarise(rows, k),
                "results": [
                    {
                        "id": r["id"],
                        "category": r["category"],
                        "answerable": r["answerable"],
                        "expected_pages": r["expected_pages"],
                        "top_pages": r["top_pages"],
                        "rank": r["rank"],
                        "rr": round(r["rr"], 4),
                        "top1_score": round(r["top1_score"], 6),
                        "top1_dense": round(r["top1_dense"], 4),
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def compare() -> None:
    """Every saved run side by side, and which items each variant moves."""
    paths = sorted(EVAL_RUNS_DIR.glob("*.json")) if EVAL_RUNS_DIR.exists() else []
    if not paths:
        raise SystemExit(
            f"No saved runs in {EVAL_RUNS_DIR.name}/. "
            "Run: python eval.py --retriever dense|hybrid|hyde"
        )

    runs = []
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        run["label"] = path.stem
        runs.append(run)
    # Baseline first, then the variants, so deltas read left to right.
    runs.sort(key=lambda r: (r["label"] != "dense", r["label"]))
    names = [r["label"] for r in runs]
    k = runs[0]["k"]

    print(f"\n{len(runs)} runs, K={k}\n")
    w = 9
    print(f"  {'metric':<24}" + "".join(f"{n:>{w}}" for n in names))
    print(f"  {'-' * (24 + w * len(names))}")

    for label, key in ((f"recall@{k}", f"recall@{k}"), ("recall@1", "recall@1"), ("MRR", "mrr")):
        cells = "".join(f"{r['summary']['overall'][key]:>{w}.3f}" for r in runs)
        print(f"  {label:<24}{cells}")

    cells = "".join(
        f"{r['summary']['multi_page_strict']:>{w}.3f}"
        if r["summary"]["multi_page_strict"] is not None
        else f"{'-':>{w}}"
        for r in runs
    )
    print(f"  {'multi_page strict':<24}{cells}")

    print()
    for cat in CATEGORIES:
        if cat == "unanswerable" or not any(cat in r["summary"]["by_category"] for r in runs):
            continue
        for label, key in ((f"recall@{k}", f"recall@{k}"), ("recall@1", "recall@1")):
            cells = "".join(
                f"{r['summary']['by_category'][cat][key]:>{w}.3f}"
                if cat in r["summary"]["by_category"]
                else f"{'-':>{w}}"
                for r in runs
            )
            print(f"  {cat + ' ' + label:<24}{cells}")

    print()
    print(f"  {'mean top1 dense, answerable':<24}"
          + "".join(f"{r['summary']['answerable_top1_dense_mean']:>{w}.3f}" for r in runs))
    print(f"  {'  same, unanswerable':<24}"
          + "".join(f"{r['summary']['unanswerable_top1_dense_mean']:>{w}.3f}" for r in runs))

    # Per-question ranks. A dash is a miss; moved rows are the point of this.
    by_id = {r["label"]: {x["id"]: x for x in r["results"]} for r in runs}
    order = [x["id"] for x in runs[0]["results"]]

    print(f"\n  rank of first expected page (- = not in top {k})\n")
    print(f"  {'id':<5} {'category':<12}" + "".join(f"{n:>9}" for n in names) + "   moved")
    print(f"  {'-' * (17 + 9 * len(names) + 8)}")

    for qid in order:
        rows = [by_id[n][qid] for n in names]
        if not rows[0]["answerable"]:
            continue
        ranks = [r["rank"] for r in rows]
        cells = "".join(f"{(str(x) if x else '-'):>9}" for x in ranks)
        moved = "   <-" if len(set(ranks)) > 1 else ""
        print(f"  {qid:<5} {rows[0]['category']:<12}{cells}{moved}")
    print()


def main(path: Path, retriever: str, backend: str) -> None:
    store = open_store(backend)
    items = load_items(path, {c["page"] for c in store.chunks})

    rows = run(items, store, EVAL_K, RETRIEVERS[retriever])

    print(
        f"\n{len(items)} eval items against {len(store.chunks)} chunks  "
        f"[retriever: {retriever}, store: {store.name}]"
    )
    print_table(rows, EVAL_K)
    print_metrics(rows, EVAL_K)
    print_strict_multi_page(rows, EVAL_K)
    print_separation(rows)
    print(f"\nrun saved to {save_run(retriever, rows, EVAL_K, store.name)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_set", nargs="?", type=Path, default=EVAL_PATH)
    parser.add_argument(
        "--retriever", choices=sorted(RETRIEVERS), default=EVAL_DEFAULT_RETRIEVER
    )
    parser.add_argument(
        "--store", choices=("numpy", "postgres"), default=None,
        help="which backend serves vector search (default: config.STORE)",
    )
    parser.add_argument(
        "--compare", action="store_true", help="show every saved run side by side"
    )
    args = parser.parse_args()

    if args.compare:
        compare()
    else:
        main(args.eval_set, args.retriever, args.store)
