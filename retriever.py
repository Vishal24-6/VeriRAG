import numpy as np
import time
import faiss
import logging
import hashlib
from rank_bm25 import BM25Okapi
import re

from chunking import chunk_text
import config
from models import ModelManager
from cache import QueryCache


# -------------------------------
# Logging (debug only, no UI)
# -------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# -------------------------------
# Singleton Model Manager + Cache
# -------------------------------

_models = ModelManager()
_query_cache = QueryCache()


# -------------------------------
# Retrieval stats (module-level)
# -------------------------------

_stats_total_chunks = 0
_stats_candidates_after_rrf = 0
_stats_candidates_after_dedup = 0
_stats_rerank_time_ms = 0.0


def get_bi_encoder():
    """Access bi-encoder via ModelManager singleton."""
    return _models.bi_encoder


def get_cross_encoder():
    """Access cross-encoder via ModelManager singleton."""
    return _models.cross_encoder


# -------------------------------
# Text preprocessing
# -------------------------------

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.split()


def _chunk_hash(text):
    """Fingerprint a chunk for near-duplicate detection."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


# -------------------------------
# Adaptive Top-K
# -------------------------------

def _adaptive_k(question, base_k=10):
    """Return wider initial_k for longer/complex queries."""
    word_count = len(question.split())
    if word_count > 15:
        return base_k * 3   # complex query
    elif word_count > 8:
        return base_k * 2   # medium query
    return base_k           # simple query


# -------------------------------
# Build Hybrid Index
# -------------------------------

def build_index(documents):

    chunks = []
    sources = []

    for doc_name, text in documents.items():

        doc_chunks = chunk_text(text)

        for chunk in doc_chunks:
            chunks.append(chunk)
            sources.append(doc_name)

    # --- Fix 4: Sanitize chunks — strip whitespace, remove empties ---
    paired = [(c.strip(), s) for c, s in zip(chunks, sources) if c.strip()]
    if paired:
        chunks, sources = zip(*paired)
        chunks = list(chunks)
        sources = list(sources)
    else:
        chunks = []
        sources = []

    # Guard: empty corpus → return safe defaults
    if not chunks:
        logger.warning("No chunks produced from documents — returning empty index")
        model = get_bi_encoder()
        dummy_dim = model.get_sentence_embedding_dimension()
        empty_index = faiss.IndexFlatL2(dummy_dim)
        return (None, empty_index), [], []

    # Tokenize and filter out chunks that produce empty token lists
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    filtered = [
        (c, s, t)
        for c, s, t in zip(chunks, sources, tokenized_chunks)
        if len(t) > 0
    ]

    if filtered:
        chunks, sources, tokenized_chunks = zip(*filtered)
        chunks = list(chunks)
        sources = list(sources)
        tokenized_chunks = list(tokenized_chunks)
    else:
        logger.warning("All chunks produced empty tokens — returning empty index")
        model = get_bi_encoder()
        dim = model.get_sentence_embedding_dimension()
        empty_index = faiss.IndexFlatL2(dim)
        return (None, empty_index), [], []

    logger.debug(f"Chunks: {len(chunks)} | Tokenized: {len(tokenized_chunks)}")

    bm25 = BM25Okapi(tokenized_chunks)

    # Embedding index
    model = get_bi_encoder()
    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    logger.debug(f"Index built: {len(chunks)} chunks from {len(documents)} document(s)")

    return (bm25, index), chunks, sources


# -------------------------------
# Single-Query RRF Retrieval
# -------------------------------

def _rrf_retrieve(bm25, vector_index, chunks, query, initial_k):
    """Run BM25 + FAISS retrieval for a single query and return RRF scored dict."""

    # BM25
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:initial_k]

    # Vector
    model = get_bi_encoder()
    query_embedding = model.encode([query])
    distances, vector_top = vector_index.search(np.array(query_embedding), initial_k)
    vector_top = vector_top[0]

    # RRF fusion
    rrf_scores = {}

    for rank, idx in enumerate(bm25_top):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (config.RRF_K + rank + 1)

    for rank, idx in enumerate(vector_top):
        rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1 / (config.RRF_K + rank + 1)

    return rrf_scores


# -------------------------------
# Hybrid Search + Dual Query + Rerank
# -------------------------------

def search_index(index_bundle, chunks, sources, question, original_query=None, top_k=config.TOP_K):
    """
    Hybrid retrieval with dual-query support.

    Args:
        index_bundle: (bm25, faiss_index) tuple
        chunks: list of text chunks
        sources: list of source names per chunk
        question: rewritten query (primary)
        original_query: original user question (optional, for dual retrieval)
        top_k: number of final results to return
    """

    global _stats_total_chunks, _stats_candidates_after_rrf
    global _stats_candidates_after_dedup, _stats_rerank_time_ms

    t_start = time.perf_counter()

    bm25, vector_index = index_bundle

    # Guard: empty index
    if bm25 is None or not chunks:
        logger.warning("Empty index — returning no results")
        return [], [], []

    _stats_total_chunks = len(chunks)

    # Adaptive retrieval width
    initial_k = min(_adaptive_k(question), len(chunks))

    # -------- Dual Query RRF --------

    rrf_scores = _rrf_retrieve(bm25, vector_index, chunks, question, initial_k)

    if original_query and original_query.strip().lower() != question.strip().lower():
        # Merge RRF scores from original query (equal weight)
        orig_rrf = _rrf_retrieve(bm25, vector_index, chunks, original_query, initial_k)

        for idx, score in orig_rrf.items():
            rrf_scores[idx] = rrf_scores.get(idx, 0) + score

        logger.debug(f"Dual query retrieval: rewritten='{question}' | original='{original_query}'")
    else:
        logger.debug(f"Single query retrieval: '{question}'")

    # -------- RRF Score Threshold Filter --------
    # Sort descending, remove relative outliers, then hard-cap before reranking
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    if ranked:
        top_rrf_score = ranked[0][1]
        rrf_threshold = top_rrf_score * config.RRF_THRESHOLD  # Remove chunks scoring below threshold of top
        ranked = [(idx, score) for idx, score in ranked if score >= rrf_threshold]

    # Hard cap: cross-encoder should never see more than initial_k candidates
    ranked = ranked[:initial_k]

    _stats_candidates_after_rrf = len(ranked)

    if not ranked:
        logger.warning("All candidates below RRF threshold — returning empty for fallback")
        return [], [], []

    # -------- Hash-based Deduplication --------

    seen_hashes = set()
    candidate_indices = []

    for idx, _ in ranked:
        h = _chunk_hash(chunks[idx])
        if h not in seen_hashes:
            seen_hashes.add(h)
            candidate_indices.append(idx)

        if len(candidate_indices) >= initial_k:
            break

    _stats_candidates_after_dedup = len(candidate_indices)

    # -------- Cross-Encoder Reranking --------

    cross_encoder = get_cross_encoder()

    t_rerank_start = time.perf_counter()
    pairs = [(question, chunks[idx]) for idx in candidate_indices]
    ce_scores = cross_encoder.predict(pairs)
    _stats_rerank_time_ms = (time.perf_counter() - t_rerank_start) * 1000

    scored = sorted(
        zip(candidate_indices, ce_scores),
        key=lambda x: x[1],
        reverse=True
    )

    logger.debug("Top reranked chunks:")
    for rank, (idx, score) in enumerate(scored[:top_k]):
        logger.debug(f"  [{rank+1}] score={score:.3f} | source={sources[idx]} | chunk={chunks[idx][:80]}...")

    # -------- Build Final Results --------

    results = []
    result_sources = []
    raw_ce_scores = []

    for idx, ce_score in scored[:top_k]:
        results.append(chunks[idx])
        result_sources.append(sources[idx])
        raw_ce_scores.append(float(ce_score))

    # -------- Confidence Score (Fix 5: per-chunk sigmoid CE scores) --------

    if raw_ce_scores:
        ce_array = np.array(raw_ce_scores)
        sigmoid_scores = 1 / (1 + np.exp(-ce_array))
        confidence_scores = [round(float(s), 4) for s in sigmoid_scores]
    else:
        confidence_scores = []

    logger.debug(f"Confidence scores: {confidence_scores}")

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000
    logger.debug(f"Retrieval latency: {latency_ms:.1f} ms")

    return results, result_sources, confidence_scores


# -------------------------------
# Retrieval Stats
# -------------------------------

def get_retrieval_stats():
    """Return a dict with stats from the most recent search_index call."""
    return {
        "total_chunks": _stats_total_chunks,
        "candidates_after_rrf": _stats_candidates_after_rrf,
        "candidates_after_dedup": _stats_candidates_after_dedup,
        "rerank_time_ms": round(_stats_rerank_time_ms, 2),
    }