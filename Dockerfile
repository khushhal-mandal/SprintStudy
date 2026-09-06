# The API image.
#
# Layer order is the whole design here. Torch and the pip install are the slow
# steps, so they sit above the application code: editing qa.py rebuilds one
# small layer, not the dependency tree.
#
# The embedding model is deliberately NOT in this image. It is fetched on first
# container start into a volume mounted at HF_HOME - see docker-entrypoint.sh
# and the study-models volume in docker-compose.yml. Baking it cost a 136MB
# layer (80MB compressed) carrying a copy of a third-party artefact versioned
# independently of this repo; measured, the image goes 1.72GB -> 1.58GB.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Torch first, from the CPU wheel index. The default wheel ships CUDA libraries
# this project will never use - roughly a gigabyte of them. sentence-transformers
# below then finds torch already satisfied.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
