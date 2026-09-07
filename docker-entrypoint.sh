#!/bin/sh
# Fetch the embedding model into the mounted cache, then hand off to CMD.
#
# The model is not baked into the image. It lives in the study-models volume
# mounted at $EMBED_CACHE_DIR, so the first `docker compose up` downloads it
# and every start after that is a cache hit costing a few seconds.
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
from embedder import embed_passage

# Goes through embedder rather than constructing TextEmbedding directly, so the
# warm-up exercises exactly the path that serves requests - same model id, same
# cache dir, same normalisation. A warm-up that used its own settings could
# succeed while the real path still had to download.
print(f"entrypoint: loading {EMBED_MODEL} (downloads on a cold cache) ...", flush=True)
vector = embed_passage("warm up")
print(f"entrypoint: model ready, {vector.shape[0]} dims", flush=True)
PY

exec "$@"
