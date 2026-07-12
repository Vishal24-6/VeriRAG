"""
config.py — Central Configuration Hub

All model names, thresholds, and hyperparameters in one place.
Every module imports from here — this is the single source of truth.
"""

# ──────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────

BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

# ──────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────

CHUNK_SIZE = 800          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks

# ──────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────

TOP_K = 5                 # final results to return
RRF_K = 60                # RRF constant (from Cormack et al. 2009)
RRF_THRESHOLD = 0.3       # remove chunks scoring < 30% of top RRF score
DRIFT_THRESHOLD = 0.85    # cosine similarity floor for query rewrite acceptance

# ──────────────────────────────────────────────────────────
# Confidence Gating
# ──────────────────────────────────────────────────────────

CONFIDENCE_MEAN_THRESHOLD = 0.35   # force web fallback if mean < this
CONFIDENCE_MAX_THRESHOLD = 0.4     # force web fallback if max < this

# ──────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────

LLM_TEMPERATURE = 0.2
MAX_CONTEXT_TOKENS = 4096          # ~16K chars at 4 chars/token
MAX_CONTEXT_CHARS = 3500           # context budget for app.py
LLM_MAX_RETRIES = 2
LLM_RETRY_DELAY = 2               # seconds

SYSTEM_PROMPT = (
    "You are VeriRAG, a precise document QA assistant. "
    "Answer ONLY from the provided context. If the context doesn't "
    "contain the answer, say so clearly. Always cite sources using [1], [2] notation."
)

# ──────────────────────────────────────────────────────────
# Web Search
# ──────────────────────────────────────────────────────────

WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TIMEOUT = 10            # seconds

# ──────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────

APP_VERSION = "2.0.0"
APP_TITLE = "🧠 Hybrid Multi-Document RAG System"
