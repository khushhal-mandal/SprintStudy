from pathlib import Path

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# bge-v1.5 models are trained with this instruction on the QUERY side only.
# Chunks are embedded as-is. Never add it to chunks.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# A text PDF gives far more than this per page. Anything below means it's scanned.
MIN_CHARS_PER_PAGE = 100

DATA_DIR = Path(__file__).parent / "data"
