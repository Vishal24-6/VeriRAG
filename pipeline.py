"""
pipeline.py — Core RAG Pipeline Orchestrator

This is the central nervous system of VeriRAG. It ties together:
  config.py    → all hyperparameters
  chunking.py  → document splitting
  retriever.py → hybrid search + reranking
  llm.py       → query rewriting + answer generation
  web_search.py → fallback search

The app.py UI layer calls this module instead of wiring modules directly.
Benchmark and evaluation scripts also use this for consistent behavior.
"""

import time
import logging
import numpy as np

import config
from retriever import build_index, search_index, get_retrieval_stats
from llm import query_llm, rewrite_query, is_context_sufficient
from web_search import web_search, clean_query
from models import ModelManager
from cache import QueryCache
from circuit_breaker import llm_breaker, web_breaker

logger = logging.getLogger(__name__)

_models = ModelManager()
_query_cache = QueryCache()


# ──────────────────────────────────────────────────────────
# Summary Detection
# ──────────────────────────────────────────────────────────

_SUMMARY_SIGNALS = (
    "what are the topics", "what topics", "list the topics",
    "summarize", "summary", "overview", "what is the file about",
    "what does the file contain", "what is in the file",
    "what all", "table of contents", "main points",
    "what does the document", "what is this document",
    "what are the main", "all topics", "tell me about the file",
    "what is covered", "what subjects",
)

_DOC_SPECIFIC_SIGNALS = (
    "this pdf", "the file", "the document", "uploaded",
    "whole pdf", "this file", "the pdf", "my document",
    "my pdf", "my file", "study the pdf", "study whole",
    "how many pages", "how long is",
)

_FAILURE_PATTERNS = (
    "i don't know", "i do not know", "cannot answer",
    "no information", "not enough information", "unable to answer",
    "not provided", "not mentioned",
)


def is_summary_query(query):
    """Detect broad/summary questions that need wide sampling."""
    return any(s in query.lower() for s in _SUMMARY_SIGNALS)


def is_doc_specific(query):
    """Detect questions about the document itself (web can't help)."""
    return any(s in query.lower() for s in _DOC_SPECIFIC_SIGNALS)


def is_failure_response(answer):
    """Detect LLM 'I don't know' responses."""
    return any(p in answer.lower() for p in _FAILURE_PATTERNS)


# ──────────────────────────────────────────────────────────
# Confidence Check
# ──────────────────────────────────────────────────────────

def is_low_confidence(scores):
    """Check if retrieval confidence is below thresholds."""
    if not scores:
        return True
    top_scores = scores[:3]
    mean_low = (sum(top_scores) / len(top_scores)) < config.CONFIDENCE_MEAN_THRESHOLD
    max_low = max(scores, default=0) < config.CONFIDENCE_MAX_THRESHOLD
    return mean_low or max_low


# ──────────────────────────────────────────────────────────
# Context Building
# ──────────────────────────────────────────────────────────

def build_document_context(chunks, sources, max_chars=None):
    """Build context string from retrieved chunks, respecting budget."""
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    context = "--- Document Sources ---\n"
    used = len(context)

    chunks_used = 0
    for i, chunk in enumerate(chunks):
        block = f"[{i+1}] ({sources[i]})\n{chunk}\n\n"
        if used + len(block) > max_chars:
            break
        context += block
        used += len(block)
        chunks_used += 1

    return context, used, chunks_used


def build_web_context_block(results, remaining_budget):
    """Build web search context block within remaining budget."""
    if not results or remaining_budget < 200:
        return ""

    web_context = "--- Web Search Results ---\n"
    used = len(web_context)

    for i, r in enumerate(results):
        block = f"[Web {i+1}]\n{r['title']}\n{r['snippet']}\n{r['url']}\n\n"
        if used + len(block) > remaining_budget:
            break
        web_context += block
        used += len(block)

    return web_context


# ──────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────

class QueryResult:
    """Structured result from the RAG pipeline."""

    def __init__(self):
        self.answer = ""
        self.sources = []
        self.confidence = 0.0
        self.num_chunks = 0
        self.used_web = False
        self.elapsed = 0.0
        self.is_error = False
        self.error_message = ""
        self.retrieval_stats = {}


def run_query(
    index_bundle,
    chunks,
    sources,
    question,
    chat_history=None,
):
    """
    Execute the full RAG pipeline for a single question.

    Flow:
      1. Query rewriting (with drift guard)
      2. Summary detection → broad sampling OR hybrid retrieval
      3. Context building (with budget)
      4. Confidence gating
      5. LLM generation OR web fallback
      6. Failure pattern detection → re-route

    Returns:
        QueryResult with answer, sources, confidence, timing, etc.
    """
    result = QueryResult()
    t_start = time.perf_counter()

    # ── Step 1: Query Rewriting ────────────────────────
    try:
        rewritten = rewrite_query(question, chat_history or [])
    except Exception:
        rewritten = question

    if len(rewritten.split()) < 3:
        rewritten = question

    # ── Step 2: Retrieval ─────────────────────────────
    is_summary = is_summary_query(question)

    if is_summary and len(chunks) > 0:
        # Broad sampling across document
        max_samples = min(12, len(chunks))
        stride = max(1, len(chunks) // max_samples)
        sampled = list(range(0, len(chunks), stride))[:max_samples]

        retrieved_chunks = [chunks[i] for i in sampled]
        retrieved_sources = [sources[i] for i in sampled]
        scores = [0.5] * len(retrieved_chunks)
    else:
        retrieved_chunks, retrieved_sources, scores = search_index(
            index_bundle, chunks, sources,
            rewritten, original_query=question
        )
        # Fallback: retry with original if nothing returned
        if not retrieved_chunks:
            retrieved_chunks, retrieved_sources, scores = search_index(
                index_bundle, chunks, sources, question
            )

    # ── Step 3: Context Building ──────────────────────
    doc_context, used_chars, chunks_used = build_document_context(
        retrieved_chunks, retrieved_sources
    )

    result.sources = list(dict.fromkeys(retrieved_sources))
    result.num_chunks = len(retrieved_chunks)
    result.confidence = round(float(max(scores)), 2) if scores else 0.0

    # ── Step 4: Confidence Gating ─────────────────────
    if is_summary:
        sufficient = True
    else:
        # Check retrieval confidence
        if not is_summary and is_low_confidence(scores):
            sufficient = False
        else:
            check_context = "\n".join(retrieved_chunks[:3])
            try:
                sufficient = is_context_sufficient(check_context, question)
            except Exception:
                sufficient = False

    # ── Step 5: Generate or Fallback ──────────────────
    if sufficient:
        try:
            answer = "".join(query_llm(doc_context, question))
            if not answer.strip():
                answer = "I'm sorry, I couldn't generate a response. Please try again."

            # Check for failure patterns
            if is_failure_response(answer):
                if is_doc_specific(question):
                    result.answer = (
                        "This information isn't available in the uploaded document. "
                        "Try asking about specific topics covered in the PDF."
                    )
                    result.elapsed = time.perf_counter() - t_start
                    return result
                else:
                    sufficient = False  # Re-route to web
            else:
                result.answer = answer
                result.elapsed = time.perf_counter() - t_start
                return result
        except Exception:
            sufficient = False

    # ── Step 6: Web Fallback ──────────────────────────
    if is_doc_specific(question):
        result.answer = (
            "This information isn't available in the uploaded document. "
            "Try asking about specific topics covered in the PDF."
        )
        result.elapsed = time.perf_counter() - t_start
        return result

    # Web search
    web_query = clean_query(rewritten)
    try:
        web_results = web_search(web_query)
    except Exception:
        web_results = []

    remaining = config.MAX_CONTEXT_CHARS - used_chars
    web_context = build_web_context_block(web_results, remaining)

    if not web_context and (not retrieved_chunks or is_low_confidence(scores)):
        result.answer = (
            "⚠️ I couldn't find sufficient information in the uploaded documents "
            "or via web search to answer your question. "
            "Please try rephrasing or uploading more relevant documents."
        )
        result.is_error = True
        result.elapsed = time.perf_counter() - t_start
        return result

    # Combine contexts and generate
    combined = doc_context + "\n" + web_context if web_context else doc_context
    result.used_web = bool(web_context)

    try:
        answer = "".join(query_llm(combined, question))
        if not answer.strip():
            answer = "I'm sorry, I couldn't generate a response. Please try again."
        result.answer = answer
    except Exception:
        result.answer = (
            "I'm sorry, I couldn't generate a response right now. "
            "Please try again in a moment."
        )
        result.is_error = True

    # Attach retrieval stats
    try:
        from retriever import get_retrieval_stats
        result.retrieval_stats = get_retrieval_stats()
    except Exception:
        pass

    result.elapsed = time.perf_counter() - t_start
    return result
