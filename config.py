import os
from pathlib import Path

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# fastembed streams in batches; 32 was sentence-transformers' setting and is
# kept so ingestion memory stays flat on a small instance.
EMBED_BATCH = 32

# Where the ONNX weights live. Overridden in the container to a mounted volume
# so the model is fetched once rather than baked into the image.
EMBED_CACHE_DIR = os.environ.get("EMBED_CACHE_DIR", str(Path(__file__).parent / ".models"))

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

# Crash guard against empty or nonsense input. NOT a relevance judgement -
# refusal is the grounding prompt's job and nothing else's.
#
# Under HyDE this number cannot mean anything more than that. Question-vs-chunk
# cosine runs 0.568-0.888 on answerable items and 0.574-0.675 on unanswerable
# ones: the lowest answerable question scores below every unanswerable one, so
# the two are not merely overlapping but interleaved. Any threshold placed near
# them would be arbitrary, and would refuse real questions to no purpose.
#
# 0.40 sits below anything a genuine question produces. It exists so a garbage
# query does not burn an LLM call, and for nothing else.
MIN_CONTEXT_SCORE = 0.40

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# The llama-3.x chat models are not reachable on this account; gpt-oss-120b is
# the strongest general model the key can see. Check with:
#   curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
GROQ_MODEL = "openai/gpt-oss-120b"
LLM_TEMPERATURE = 0.0
# Raised from 512, which truncated. One answer in the previous qa.py --all run
# stopped mid-sentence at the cap - p03 ended "...while different hash values
# guarantee they differ", losing its closing clause. Every metric still passed
# (p03 cites page 255 earlier in the answer, so uncited stayed False), which is
# the point: truncation degraded the text without moving a single number, so
# nothing in the harness would have caught it. A quiz question plus four options
# and an explanation is the longer shape and has less headroom still.
LLM_MAX_TOKENS = 1024

# requests' timeout is per socket operation, not per call: a response that
# stalls after its headers arrive, or trickles a byte before each deadline,
# never trips it. That hung a quiz run for 18 minutes with an open connection
# and no CPU. Connect and read are bounded separately, and LLM_DEADLINE bounds
# the whole call regardless of how the bytes arrive.
LLM_CONNECT_TIMEOUT = 10
LLM_READ_TIMEOUT = 30
LLM_DEADLINE = 120

# Groq's 429 body says how long to wait, and for a daily-quota breach that can
# be minutes ("try again in 7m22.368s"). Sleeping that long inside a request
# handler is the same failure the deadline above exists to prevent, so a wait
# longer than this is refused outright rather than honoured.
LLM_MAX_RETRY_WAIT = 60

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

# --- retrieval variants ---

# What qa.py answers with. eval.py has its own default below; the two are
# separate so benchmarking a variant never silently changes what users get.
RETRIEVER = "hyde"

# Default for eval.py's --retriever flag. Stays dense: it is the baseline
# every other run is measured against.
EVAL_DEFAULT_RETRIEVER = "dense"
EVAL_RUNS_DIR = Path(__file__).parent / "eval_runs"

# Okapi BM25. Standard values, not fitted to the eval set.
BM25_K1 = 1.5
BM25_B = 0.75

# Dense cosine and BM25 are on incomparable scales, so hybrid fuses ranks
# rather than scores. RRF_K damps the head of each list; 60 is the usual value.
RRF_K = 60

# Keyword's share of the fused score. 0.5 is plain unweighted RRF - the
# textbook baseline, carrying no judgement fitted to these 20 questions.
HYBRID_WEIGHT = 0.5

# HyDE: answer the question first, then search with the hypothetical answer.
# Cached so the benchmark is reproducible - generation varies run to run even
# at temperature 0. Delete the cache file to regenerate.
# Two caches, deliberately. The benchmark cache is committed so eval runs are
# reproducible; the runtime one is gitignored, because questions asked through
# the API are the user's data and would otherwise accumulate in the artefact
# that makes the benchmark deterministic. Reads check both, writes only ever
# touch the runtime file.
HYDE_CACHE_PATH = Path(__file__).parent / "hyde_cache.json"
HYDE_RUNTIME_CACHE_PATH = Path(__file__).parent / "hyde_cache.local.json"
HYDE_PROMPT = """Write a short passage that answers the question below, as it would appear
in a technical reference. Two or three sentences. State it plainly as fact -
do not hedge and do not say whether you are certain.

Question: {question}

Passage:"""

# --- quiz generation ---

QUIZ_PATH = Path(__file__).parent / "quiz.json"
QUIZ_DEFAULT_N = 5
QUIZ_OPTIONS = 4

# One regeneration attempt before a chunk is dropped and backfilled.
QUIZ_MAX_RETRIES = 1

# Returned bare when a chunk has no teachable content - the table-of-contents
# pages are 57% dot leaders. Same sentinel pattern as REFUSAL_SENTINEL.
QUIZ_DECLINE_SENTINEL = "NO_QUESTION"

# The model never sees a chunk id: source_chunk_id is attached from metadata
# afterwards, so a wrong one is unrepresentable. What it must supply is
# `evidence`, a span copied out of the passage, which is checked against the
# chunk. That is what makes a question's answer traceable to its source.
QUIZ_PROMPT = """Write one multiple-choice question testing understanding of the passage below.

Rules:
1. The question must be answerable from the passage alone, and the correct
   answer must be stated in or follow from it. Do not use outside knowledge.
2. Test understanding, not recall. The question must be answerable by someone
   who understood the passage but memorised none of its specific example
   values. If the correct answer is a number lifted from a worked example, do
   not ask it. Ask why that result arises, or what would change it.
3. Prefer "why", "how", or "what would happen if" over "what is". A question
   whose answer can be found by scanning the passage for a matching word is a
   weak question.
4. Give exactly 4 options. Exactly one is correct.
5. Each wrong option must be a plausible misunderstanding of the mechanism -
   a confused complexity, a step applied in the wrong order, a condition
   inverted, a structure mistaken for a similar one. Never use arbitrary
   near-values, and never make the wrong options numbers close to the right
   number.
6. Keep all four options about the same length. Do not make the correct one
   longer or more detailed than the others.
7. "evidence" must be copied word for word from the passage - the sentence or
   clause that makes the correct answer correct. Do not paraphrase it.
8. Never mention a page number, a section number, the document, or the passage
   itself. The question and its options must stand alone: whoever answers them
   sees only the question, so "According to the passage..." points at nothing
   they can read. Write the substance in rather than referring to it.
9. If the passage has no teachable content - a table of contents, an index, a
   fragment, a bare list of numbers - reply with exactly {sentinel} and
   nothing else. Do not invent a question from nothing.

Reply with a single JSON object and no other text:

{{
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "answer_index": 0,
  "evidence": "...",
  "explanation": "..."
}}

Passage:
{passage}
"""

# --- PDF glyph repair ---

# The book is typeset with Fourier math fonts plus AMS symbol fonts, whose
# built-in Type 1 encodings pdfminer cannot resolve to unicode. It emits
# "(cid:N)" instead, corrupting 110 of 680 chunks across 62 pages.
#
# Every code was enumerated from the PDF and identified from its font and
# surrounding text. Fourier-Math-Extension is a CMEX-style extension font, so
# the same delimiter appears at several codes in different display sizes -
# hence the repeated brackets below.
CID_MAP = {
    # Fourier-Math-Extension: large operators and delimiters
    88: "∑",   # sum, "Each sum of the form n (cid:88) xk=1k+2k+..."
    80: "∑",   # sum, smaller size
    89: "∏",   # product, "The factorial n! can be defined n (cid:89) x=1*2*3*..."
    112: "√",  # radical, "Binet's formula: (cid:112) (1+ 5)n"
    113: "√",  # radical, larger size
    40: "{",        # cases brace, "(cid:40) true x=0 possible(x,0)= false"
    183: "[",       # matrix bracket, "(cid:183) 6 1 4(cid:184)"
    184: "]",
    195: "(",       # binomial delimiters, "(cid:195) (cid:33) n n-1"
    33: ")",
    161: "(",       # binomial delimiters, smaller
    162: ")",
    179: "(",       # parenthesised quotient, "(cid:179)a(cid:180) log ="
    180: ")",
    # Underbrace pieces. Four codes assemble one horizontal brace, so each
    # carries no text of its own - dropped rather than repeated four times.
    122: "",
    123: "",
    124: "",
    125: "",
    # Fourier-Math-Symbols
    48: "′",   # prime, "p(cid:48) (u)=p(u+h)"
    98: "⌊",   # floor left, "(cid:98)x(cid:99) rounds the number x down"
    99: "⌋",   # floor right
    100: "⌈",  # ceiling left, "(cid:100)x(cid:101) rounds the number up"
    101: "⌉",  # ceiling right
    59: "∅",   # empty set, "the symbol (cid:59) denotes an empty set"
    54: "≠",   # see NEGATION_PAIR below
    # Tiling notation, section 7.6. Page 86 states the representation uses "the
    # upper square of a vertical tile", which fixes 117 as the upper half; the
    # example grid stacks 117 directly above 116 in every column.
    117: "⊓",  # upper square of a vertical tile
    116: "⊔",  # lower square
    64: "⊏",   # left square of a horizontal tile (always paired with 65)
    65: "⊐",   # right square
    3: "□",    # placeholder square standing for any of the other three
    # Fourier-Math-BlackBoard
    78: "ℕ",   # N, "(cid:78) (natural numbers)"
    90: "ℤ",   # Z, "(cid:90) (integers)"
    81: "ℚ",   # Q, "(cid:81) (rational numbers)"
    82: "ℝ",   # R, "(cid:82) (real numbers)"
    # Fourier-Math-Letters-Italic
    178: "ε",  # epsilon, "|a-b|<(cid:178), where (cid:178) is a small number"
    # MSBM10
    45: "∤",   # does not divide, "otherwise we write a(cid:45)b"
}

# cid 54 is a negation slash drawn *over* the following "=", so the raw stream
# reads "x(cid:54)=0" and means "x != 0". Substituting the code alone would give
# "x!==0", so the pair is replaced together. Getting this wrong inverts meaning.
NEGATION_PAIR = ("(cid:54)=", "≠")


# Structural pre-filter, in front of the model's NO_QUESTION sentinel rather
# than instead of it: on the first run the sentinel let a contents-page question
# through, so it needs a backstop that does not depend on the model's judgement.
#
# Thresholds sit in the measured gap. Over the clean index the dot-leader ratio
# tops out at 0.051 and alphanumeric density bottoms out at 0.530; the
# contents-page chunks ran 0.318-0.356 and 0.243-0.311 respectively. Nothing
# lies between, so these reject every contents chunk and no real one.
QUIZ_MIN_CHARS = 200
QUIZ_MAX_DOT_RATIO = 0.15
QUIZ_MIN_ALNUM_RATIO = 0.45

# --- database ---

# Never hardcoded: the default is a local convenience matching docker-compose.yml,
# and any real deployment overrides it through the environment.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://study:study@localhost:5432/study"
)

# Which backend the retrievers read from. The NumPy store stays the reference
# implementation - the Postgres one is only correct insofar as it matches it.
STORE = os.environ.get("STORE", "numpy")
