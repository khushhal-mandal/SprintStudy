"""Generate an MCQ quiz from a page range.

One question per chunk, one LLM call per chunk. The model never sees a chunk
id: source_chunk_id and page are attached from metadata afterwards, so a
question cannot cite a source it did not come from.

What the model must supply is `evidence` - a span copied out of the passage.
It is checked against the chunk text, which is what makes each answer traceable
to its source rather than merely claimed to be.

usage: python quiz.py <start_page> <end_page> [n]
"""

import json
import random
import re
import sys
from collections import Counter

import numpy as np

import llm
import store
from config import (
    QUIZ_DECLINE_SENTINEL,
    QUIZ_MAX_DOT_RATIO,
    QUIZ_MIN_ALNUM_RATIO,
    QUIZ_MIN_CHARS,
    QUIZ_DEFAULT_N,
    QUIZ_MAX_RETRIES,
    QUIZ_OPTIONS,
    QUIZ_PATH,
    QUIZ_PROMPT,
)


class Rejected(Exception):
    """The model's output did not meet the contract."""


def normalise(text: str) -> str:
    """Strip everything but alphanumerics, lowercased.

    pdfplumber runs words together - 'Calculatingallthosevalues' - so a quoted
    span will not match the chunk raw even when it was copied faithfully. This
    also absorbs the unicode dashes and narrow spaces the model emits.
    """
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def select(chunks: list[dict], start: int, end: int, n: int) -> list[dict]:
    """N chunks spread evenly across the page range.

    Evenly rather than the first N: chunks carry 150 characters of overlap, so
    consecutive ones share text and would yield near-duplicate questions.
    """
    in_range = [c for c in chunks if start <= c["page"] <= end]
    if not in_range:
        raise SystemExit(f"No chunks on pages {start}-{end}.")
    if n >= len(in_range):
        return in_range

    picks = np.linspace(0, len(in_range) - 1, n).round().astype(int)
    return [in_range[i] for i in dict.fromkeys(picks.tolist())]


def parse(raw: str) -> dict:
    """Tolerant parse, strict validation comes after."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise Rejected("no JSON object in the response")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise Rejected(f"invalid JSON: {exc}")


def validate(mcq: dict, chunk: dict) -> dict:
    """Enforce the contract. Raises Rejected with a reason."""
    if not isinstance(mcq, dict):
        raise Rejected("not a JSON object")

    for field in ("question", "evidence", "explanation"):
        if not isinstance(mcq.get(field), str) or not mcq[field].strip():
            raise Rejected(f"'{field}' must be a non-empty string")

    options = mcq.get("options")
    if not isinstance(options, list) or len(options) != QUIZ_OPTIONS:
        raise Rejected(f"'options' must be a list of exactly {QUIZ_OPTIONS}")
    if not all(isinstance(o, str) and o.strip() for o in options):
        raise Rejected("every option must be a non-empty string")
    if len({normalise(o) for o in options}) != QUIZ_OPTIONS:
        raise Rejected("options must be distinct")

    index = mcq.get("answer_index")
    if not isinstance(index, int) or isinstance(index, bool):
        raise Rejected("'answer_index' must be an integer")
    if not 0 <= index < QUIZ_OPTIONS:
        raise Rejected(f"'answer_index' must be 0-{QUIZ_OPTIONS - 1}, got {index}")

    # The traceability check: the quoted span has to actually be in the chunk.
    if normalise(mcq["evidence"]) not in normalise(chunk["text"]):
        raise Rejected("'evidence' is not a verbatim span of the chunk")

    return {
        "question": mcq["question"].strip(),
        "options": [o.strip() for o in options],
        "answer_index": index,
        "evidence": mcq["evidence"].strip(),
        "explanation": mcq["explanation"].strip(),
        # Attached, never generated.
        "source_chunk_id": chunk["id"],
        "page": chunk["page"],
    }


def usable(text: str) -> tuple[bool, str]:
    """Structural check before an LLM call is spent on a chunk.

    A backstop for the model's own NO_QUESTION sentinel, which let a
    contents-page question through on the first run. This judges shape rather
    than content, so it cannot be talked out of a decision.
    """
    if len(text) < QUIZ_MIN_CHARS:
        return False, f"only {len(text)} chars"

    dots = text.count(".") / len(text)
    if dots > QUIZ_MAX_DOT_RATIO:
        return False, f"dot leaders {dots:.0%}"

    alnum = sum(c.isalnum() for c in text) / len(text)
    if alnum < QUIZ_MIN_ALNUM_RATIO:
        return False, f"alphanumeric density {alnum:.0%}"

    return True, ""


def shuffle_options(mcq: dict, rng: random.Random) -> dict:
    """Randomise option order and re-point answer_index at the correct text.

    The model has a strong positional bias - it put the correct answer at A in
    4 of 5 questions on the first run, which a student can exploit without
    reading anything. Seeded per chunk so a regenerated quiz is reproducible.
    Options are validated distinct, so index() is unambiguous.
    """
    correct = mcq["options"][mcq["answer_index"]]
    options = list(mcq["options"])
    rng.shuffle(options)
    mcq["options"] = options
    mcq["answer_index"] = options.index(correct)
    return mcq


def generate_one(chunk: dict) -> tuple[dict | None, str, int]:
    """Return (mcq or None, outcome, retries). Outcome is ok|declined|rejected."""
    prompt = QUIZ_PROMPT.format(sentinel=QUIZ_DECLINE_SENTINEL, passage=chunk["text"])

    last = ""
    for attempt in range(QUIZ_MAX_RETRIES + 1):
        raw = llm.generate(prompt)
        if QUIZ_DECLINE_SENTINEL in raw:
            return None, "declined", attempt
        try:
            mcq = validate(parse(raw), chunk)
            return shuffle_options(mcq, random.Random(chunk["id"])), "ok", attempt
        except Rejected as exc:
            last = str(exc)

    print(f"    chunk #{chunk['id']}: dropped after retry ({last})")
    return None, "rejected", QUIZ_MAX_RETRIES


def build(chunks: list[dict], start: int, end: int, n: int) -> tuple[list[dict], dict]:
    """Generate n questions, backfilling past chunks the model declines."""
    in_range = [c for c in chunks if start <= c["page"] <= end]
    by_id = {c["id"]: i for i, c in enumerate(in_range)}
    selected = select(chunks, start, end, n)
    used: set[int] = set()

    quiz: list[dict] = []
    stats = Counter()

    for chunk in selected:
        position = by_id[chunk["id"]]
        candidate = chunk

        # Search outward from the declined slot so the quiz keeps its spread.
        for offset in _outward(position, len(in_range)):
            candidate = in_range[offset]
            if candidate["id"] in used:
                continue
            used.add(candidate["id"])

            ok, why = usable(candidate["text"])
            if not ok:
                stats["filtered"] += 1
                print(f"    chunk #{candidate['id']}: pre-filtered ({why})")
                continue

            mcq, outcome, retries = generate_one(candidate)
            stats[outcome] += 1
            stats["retries"] += retries
            if outcome == "ok":
                quiz.append(mcq)
                break
        else:
            print(f"    no usable chunk left near page {chunk['page']}")

    quiz.sort(key=lambda q: q["source_chunk_id"])
    return quiz, stats


def _outward(start: int, size: int):
    """start, start+1, start-1, start+2, ... clipped to the range."""
    yield start
    for step in range(1, size):
        for candidate in (start + step, start - step):
            if 0 <= candidate < size:
                yield candidate


def print_quiz(quiz: list[dict]) -> None:
    for i, q in enumerate(quiz, start=1):
        print(f"\n{i}. {q['question']}")
        for j, option in enumerate(q["options"]):
            mark = "*" if j == q["answer_index"] else " "
            print(f"   {mark} {chr(65 + j)}. {option}")
        print(f"     page {q['page']}, chunk #{q['source_chunk_id']}")
        print(f"     evidence: \"{q['evidence'][:100]}\"")


def print_stats(quiz: list[dict], stats: Counter) -> None:
    """Cheap tells that a generated quiz is bad. Reported, not asserted away."""
    print(f"\n{len(quiz)} questions")
    print(f"  pre-filtered  {stats['filtered']}")
    print(f"  declined      {stats['declined']}")
    print(f"  rejected      {stats['rejected']}")
    print(f"  retries       {stats['retries']}")

    if not quiz:
        return

    positions = Counter(q["answer_index"] for q in quiz)
    spread = " ".join(f"{chr(65 + i)}:{positions.get(i, 0)}" for i in range(QUIZ_OPTIONS))
    longest = sum(
        1
        for q in quiz
        if len(q["options"][q["answer_index"]]) == max(len(o) for o in q["options"])
    )
    leaks = [q["source_chunk_id"] for q in quiz if _leaks_page(q)]

    print(f"  answer spread {spread}  (shuffled after generation)")
    print(f"  correct is longest option  {longest}/{len(quiz)}  (want about 1 in 4)")
    print(f"  wrote a page number        {len(leaks)}  {leaks or ''}")


def _leaks_page(q: dict) -> bool:
    text = " ".join([q["question"], *q["options"], q["explanation"]])
    return bool(re.search(r"\b(page|section)\s+\d", text, re.I))


def main(start: int, end: int, n: int) -> None:
    chunks, _ = store.load()
    print(f"\nquiz from pages {start}-{end}, {n} questions")

    quiz, stats = build(chunks, start, end, n)
    print_quiz(quiz)
    print_stats(quiz, stats)

    QUIZ_PATH.write_text(
        json.dumps(
            {"pages": [start, end], "requested": n, "questions": quiz},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nsaved to {QUIZ_PATH.name}\n")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: python quiz.py <start_page> <end_page> [n]")
    try:
        first, last = int(sys.argv[1]), int(sys.argv[2])
        count = int(sys.argv[3]) if len(sys.argv) == 4 else QUIZ_DEFAULT_N
    except ValueError:
        sys.exit("pages and count must be integers")
    main(first, last, count)
