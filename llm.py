"""Groq chat completion behind a single function.

Groq's endpoint is OpenAI-compatible, so this is one POST. Kept deliberately
thin: the only thing the rest of the project knows about the LLM is
generate(prompt) -> str.
"""

import os
import re
import time

import requests

from config import GROQ_MODEL, GROQ_URL, LLM_MAX_TOKENS, LLM_TEMPERATURE, LLM_TIMEOUT

API_KEY_VAR = "GROQ_API_KEY"

# The free tier caps tokens per minute, and one grounded prompt is ~1.7k tokens,
# so a batch run hits the cap within a handful of calls. Groq says how long to
# wait; honour that rather than guessing.
MAX_ATTEMPTS = 6
FALLBACK_WAIT = 10.0


def generate(prompt: str) -> str:
    """Send one prompt, return the model's text. Raises on any failure."""
    key = os.environ.get(API_KEY_VAR)
    if not key:
        raise SystemExit(
            f"{API_KEY_VAR} is not set.\n"
            f"Get a key at https://console.groq.com/keys, then:\n"
            f'  export {API_KEY_VAR}="..."'
        )

    for attempt in range(1, MAX_ATTEMPTS + 1):
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": LLM_TEMPERATURE,
                "max_tokens": LLM_MAX_TOKENS,
            },
            timeout=LLM_TIMEOUT,
        )

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()

        if resp.status_code in (429, 500, 502, 503) and attempt < MAX_ATTEMPTS:
            time.sleep(_retry_after(resp))
            continue

        raise SystemExit(f"Groq returned {resp.status_code}: {resp.text[:400]}")

    raise SystemExit(f"Groq still failing after {MAX_ATTEMPTS} attempts.")


def _retry_after(resp: requests.Response) -> float:
    """Seconds to wait, from the header, then the message, then a default."""
    header = resp.headers.get("retry-after")
    if header:
        try:
            return float(header) + 0.5
        except ValueError:
            pass

    m = re.search(r"try again in ([\d.]+)s", resp.text)
    return float(m.group(1)) + 0.5 if m else FALLBACK_WAIT
