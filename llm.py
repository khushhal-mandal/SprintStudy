"""Groq chat completion behind a single function.

Groq's endpoint is OpenAI-compatible, so this is one POST. Kept deliberately
thin: the only thing the rest of the project knows about the LLM is
generate(prompt) -> str.

Timeouts are the fiddly part. requests' `timeout` bounds individual socket
operations, not the call, so a response that stalls after the headers arrive
hangs forever - which it did, for 18 minutes, holding an open connection at
zero CPU. The body is therefore streamed against a wall-clock deadline, and
every network fault is retried rather than raised.
"""

import json
import os
import re
import time

import requests

from config import (
    GROQ_MODEL,
    GROQ_URL,
    LLM_CONNECT_TIMEOUT,
    LLM_DEADLINE,
    LLM_MAX_RETRY_WAIT,
    LLM_MAX_TOKENS,
    LLM_READ_TIMEOUT,
    LLM_TEMPERATURE,
)

API_KEY_VAR = "GROQ_API_KEY"

# The free tier caps tokens per minute, and one grounded prompt is ~1.7k tokens,
# so a batch run hits the cap within a handful of calls. Groq says how long to
# wait; honour that rather than guessing.
MAX_ATTEMPTS = 6
FALLBACK_WAIT = 10.0
RETRY_STATUSES = (408, 429, 500, 502, 503, 504)

# Groq writes waits as "6.48s" but also "7m22.368s", and occasionally with
# an hours part. Parsing only the seconds form silently fell back to the
# default and burned attempts against a quota that needed minutes.
RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s")


def generate(prompt: str) -> str:
    """Send one prompt, return the model's text. Retries transient failures."""
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise SystemExit(
            f"{API_KEY_VAR} is not set.\n"
            f"Get a key at https://console.groq.com/keys, then:\n"
            f'  export {API_KEY_VAR}="..."'
        )

    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            status, body, headers = _request(prompt, key)
        except requests.exceptions.RequestException as exc:
            # The bug this file exists to fix: a timeout or dropped connection
            # used to propagate straight out of generate(), so the retry loop
            # below only ever saw HTTP status codes and never a network fault.
            last = f"{type(exc).__name__}: {exc}"
            if attempt == MAX_ATTEMPTS:
                break
            time.sleep(min(2**attempt, FALLBACK_WAIT))
            continue

        if status == 200:
            return json.loads(body)["choices"][0]["message"]["content"].strip()

        last = f"HTTP {status}: {body[:300]}"
        if status not in RETRY_STATUSES:
            raise SystemExit(f"Groq returned {status}: {body[:400]}")
        if attempt == MAX_ATTEMPTS:
            break

        wait = _retry_after(headers, body)
        if wait > LLM_MAX_RETRY_WAIT:
            raise SystemExit(
                f"Groq returned {status} and asked for {wait:.0f}s, over the "
                f"{LLM_MAX_RETRY_WAIT}s cap. Not waiting.\n{body[:300]}"
            )
        time.sleep(wait)

    raise SystemExit(f"Groq failed after {MAX_ATTEMPTS} attempts. Last: {last}")


def _request(prompt: str, key: str) -> tuple[int, str, dict]:
    """One POST, with the body read under a hard wall-clock deadline.

    stream=True so the body arrives in pieces we can time. Without it, requests
    reads to completion inside a single call and the deadline cannot be checked.
    """
    deadline = time.monotonic() + LLM_DEADLINE

    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT),
        stream=True,
    )

    try:
        pieces = []
        for piece in resp.iter_content(chunk_size=8192):
            if time.monotonic() > deadline:
                raise requests.exceptions.Timeout(
                    f"response exceeded LLM_DEADLINE of {LLM_DEADLINE}s"
                )
            pieces.append(piece)
        return resp.status_code, b"".join(pieces).decode("utf-8", "replace"), resp.headers
    finally:
        resp.close()


def _retry_after(headers, body: str) -> float:
    """Seconds to wait, from the header, then the message, then a default."""
    value = headers.get("retry-after") if headers else None
    if value:
        try:
            return float(value) + 0.5
        except ValueError:
            pass

    m = RETRY_AFTER_RE.search(body)
    if not m:
        return FALLBACK_WAIT

    hours, minutes, seconds = m.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(seconds) + 0.5
