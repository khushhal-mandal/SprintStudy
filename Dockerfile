# The API image.
#
# Layer order is the whole design here. The pip install is the slow step, so it
# sits above the application code: editing qa.py rebuilds one small layer, not
# the dependency tree.
#
# There is no torch in this image. Embedding runs on onnxruntime through
# fastembed, which is what makes the image small enough for a free-tier instance
# to hold - see the note in embedder.py for the measurements that made the swap
# safe to do.
#
# The embedding model is deliberately NOT in this image either. It is fetched on
# first container start into a volume mounted at EMBED_CACHE_DIR - see
# docker-entrypoint.sh and the study-models volume in docker-compose.yml. Baking
# it cost a 136MB layer carrying a copy of a third-party artefact versioned
# independently of this repo.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBED_CACHE_DIR=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The mount point for the model cache. Created here so the named volume inherits
# a directory that exists; nothing is written into it at build time.
RUN mkdir -p /models

# Application code last - changes constantly, invalidates nothing above.
# schema.sql must land in the workdir: db.apply_schema() opens it by relative path.
COPY schema.sql config.py db.py store.py pgstore.py stores.py ./
COPY pdf_parser.py chunker.py embedder.py bm25.py retrievers.py llm.py ./
COPY qa.py quiz.py api.py ./
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Whether the model ships inside the image.
#
# Defaults to true because the deployment target that cannot be configured is
# the one that has to work: a host with an ephemeral filesystem re-downloads a
# volume-cached model on every cold start, and a demo that stalls for 40s on
# first click reads as broken. docker-compose.yml passes false, because there a
# named volume persists and keeping the image small is worth more.
#
# Placed after the application code deliberately. It invalidates on any code
# change, which costs nothing on a host that builds from scratch each deploy,
# and the alternative - baking earlier - means repeating the model id here
# instead of reading it from config.py.
ARG BAKE_MODEL=true
RUN if [ "$BAKE_MODEL" = "true" ]; then \
        python -c "from embedder import embed_passage; embed_passage('bake')"; \
    fi

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
