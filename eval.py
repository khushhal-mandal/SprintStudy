"""Measure retrieval against the eval set.

Reports recall@K, recall@1 and MRR over the answerable items, a stricter
recall for multi_page items that requires every cluster to be represented, and
the top-1 similarity distribution for answerable vs unanswerable items.

Those last two distributions overlap on this eval set, which is why refusal
lives in the grounding prompt rather than in a cut on the similarity score.

usage: python eval.py [eval.json]
"""

import json
import sys
from pathlib import Path

import numpy as np

import store
from config import EVAL_K, EVAL_PATH
from embedder import embed_query

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


def run(items: list[dict], chunks: list[dict], vectors: np.ndarray, k: int) -> list[dict]:
    """Retrieve for every item and score it."""
    rows = []
    for item in items:
        # embed_query applies QUERY_PREFIX; embed_chunks does not. Questions must
        # go through this path even though it costs one forward pass per item.
        hits = store.search(embed_query(item["question"]), chunks, vectors, k)
        pages = [c["page"] for _, c in hits]
        expected = set(item["expected_pages"])

        rank = next((i for i, p in enumerate(pages, start=1) if p in expected), None)
        rows.append(
            {
                **item,
                "top_pages": pages,
                "top1_score": hits[0][0],
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
        f"{'top-' + str(k) + ' pages':<{pw}} {'rank':>4} {'RR':>5} {'top1':>6}"
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
            f"{pages[r['id']]:<{pw}} {rank:>4} {rr:>5} {r['top1_score']:>6.3f}"
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
    yes = [r["top1_score"] for r in rows if r["answerable"]]
    no = [r["top1_score"] for r in rows if not r["answerable"]]

    print("\ntop-1 similarity distribution")
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


def main(path: Path) -> None:
    chunks, vectors = store.load()
    items = load_items(path, {c["page"] for c in chunks})

    rows = run(items, chunks, vectors, EVAL_K)

    print(f"\n{len(items)} eval items against {len(chunks)} chunks")
    print_table(rows, EVAL_K)
    print_metrics(rows, EVAL_K)
    print_strict_multi_page(rows, EVAL_K)
    print_separation(rows)
    print()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_PATH)
