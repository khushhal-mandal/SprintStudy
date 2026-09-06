#!/bin/sh
# Fetch the embedding model into the mounted cache, then hand off to CMD.
#
# The model is not baked into the image. It lives in the study-models volume
# mounted at $HF_HOME, so the first `docker compose up` downloads it (128MB of
# weights) and every start after that is a cache hit costing a few seconds.
#
# Why here and not lazily on first use: embedder loads the model on the first
# call to embed_chunks, which happens inside /upload's background task. A
# download there is invisible - the document just sits in 'processing' for
# minutes with nothing in the log to explain it. Doing it before uvicorn binds
# the port puts the fetch in `docker compose logs api`, and means the API never
# accepts a request it cannot yet serve.
set -e

python - <<'PY'
from config import EMBED_MODEL
from sentence_transformers import SentenceTransformer

# The model id comes from config.py, not a second copy of the string here:
# changing it in one place must change what the container fetches.
print(f"entrypoint: loading {EMBED_MODEL} (downloads on a cold cache) ...", flush=True)
SentenceTransformer(EMBED_MODEL)
print("entrypoint: model ready", flush=True)
PY

exec "$@"
