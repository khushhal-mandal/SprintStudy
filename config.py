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

# Eval set. K is the retrieval depth every metric is measured at - changing it
# changes what recall@K means, so past numbers stop being comparable.
EVAL_PATH = Path(__file__).parent / "eval.json"
EVAL_K = 5

# Record of the last full qa.py --all run. Committed, so the numbers in the
# repo are a run that actually happened rather than a claim in a README.
QA_RUN_PATH = Path(__file__).parent / "qa_run.json"

# --- answering ---

# Separate from EVAL_K on purpose. They are both 5, but changing retrieval
# depth for answering must not silently redefine what recall@K measured.
ANSWER_K = 5

# Cost floor, not the refusal mechanism. Refusal is the grounding prompt's job.
# The lowest top-1 on the eval set is 0.623, so this should never fire there -
# if it does, something upstream has changed and is worth looking at.
MIN_CONTEXT_SCORE = 0.55

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# The llama-3.x chat models are not reachable on this account; gpt-oss-120b is
# the strongest general model the key can see. Check with:
#   curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512
LLM_TIMEOUT = 30

# The model emits this exact string instead of an answer it cannot ground.
# A sentinel makes refusal detectable; prose refusals would have to be guessed at.
REFUSAL_SENTINEL = "INSUFFICIENT_CONTEXT"
REFUSAL_MESSAGE = "That isn't covered in this document."

# Rule 4 is what keeps the citation invariant true: the model only ever emits
# passage indices, so a page number cannot be generated, only looked up.
GROUNDING_PROMPT = """You are answering questions about a single document. You are given numbered
passages from that document. They are the only information you may use.

Rules:
1. Answer only from the passages. Do not use outside knowledge, even if you
   are confident it is correct.
2. If the passages do not contain enough to answer, reply with exactly
   {sentinel} and nothing else. Do not guess, do not partially
   answer, and do not explain what is missing.
3. Cite the passages you used with bracketed numbers: [1], or [2][4] when a
   sentence draws on more than one. Put them at the end of the sentence they
   support.
4. Never write a page number, a section number, or a document title. Cite
   passage numbers only.
5. Be concise. Two or three sentences unless the question needs more.

Passages:
{passages}

Question: {question}

Answer:"""
