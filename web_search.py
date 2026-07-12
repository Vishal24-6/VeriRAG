import os
import re
import logging
import requests
from dotenv import load_dotenv
import config
from circuit_breaker import web_breaker

load_dotenv()

logger = logging.getLogger(__name__)


# ------------------------------------
# Query cleaning
# ------------------------------------

# Stop words that add no search value
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "about", "it", "its", "this", "that", "these", "those",
    "and", "or", "but", "not", "no", "do", "does", "did", "will",
    "can", "could", "would", "should", "may", "might", "shall",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "file", "document", "pdf", "uploaded", "given", "provided",
}


def clean_query(question):
    """Clean a user question into a meaningful web search query."""
    q = question.lower().strip()

    # Remove question/command prefixes
    q = re.sub(
        r"^(what is|what are|explain|describe|tell me about|how does|how do|how to|"
        r"which|who|where|when|why|list|summarize|give me)\s+",
        "",
        q,
    )

    # Remove special characters but keep alphanumeric and spaces
    q = re.sub(r"[^a-z0-9 ]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    # Remove stop words to keep only meaningful keywords
    words = [w for w in q.split() if w not in _STOP_WORDS]

    # If filtering removed everything, fall back to original cleaned text
    if not words:
        q = re.sub(r"[^a-z0-9 ]", " ", question.lower().strip())
        q = re.sub(r"\s+", " ", q).strip()
        # Ultimate fallback: return original question if all cleaning fails
        return q if q else question.strip()

    return " ".join(words)


# ------------------------------------
# Brave Search (primary, optional)
# ------------------------------------

def _brave_search(query, max_results=5):
    """Search using Brave Search API. Requires BRAVE_API_KEY."""

    api_key = os.getenv("BRAVE_API_KEY")

    if not api_key:
        return None  # Signal to use fallback

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": max_results,
    }

    try:
        response = web_breaker.call(lambda: requests.get(url, headers=headers, params=params, timeout=config.WEB_SEARCH_TIMEOUT))
        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", ""),
                "url": item.get("url", ""),
            })

        logger.debug(f"Brave search returned {len(results)} results for: '{query}'")
        return results

    except Exception as e:
        logger.warning(f"Brave search failed: {e}")
        return None  # Fallback to DuckDuckGo


# ------------------------------------
# DuckDuckGo Search (fallback)
# ------------------------------------

def _ddg_search(query, max_results=5):
    """Search using DuckDuckGo. Always available, no API key needed."""

    results = []

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:

            for r in ddgs.text(query, max_results=max_results):

                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                })

        logger.debug(f"DuckDuckGo returned {len(results)} results for: '{query}'")

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed (may be blocked in cloud): {e}")

    return results[:3]


# ------------------------------------
# Unified web search interface
# ------------------------------------

def web_search(query, max_results=config.WEB_SEARCH_MAX_RESULTS):
    """
    Search the web using Brave API (if key is set) with DuckDuckGo fallback.
    Returns: list of {title, snippet, url}
    """

    # Try Brave first
    try:
        results = _brave_search(query, max_results)
    except RuntimeError as e:
        # Circuit breaker is open — skip Brave entirely
        logger.warning(f"Brave circuit breaker open, falling back to DuckDuckGo: {e}")
        results = None

    if results is not None and len(results) > 0:
        return results[:3]

    # Fallback to DuckDuckGo
    return _ddg_search(query, max_results)


# ------------------------------------
# Build context from web results
# ------------------------------------

def build_web_context(results):

    context = ""

    for i, r in enumerate(results):

        context += f"[Web {i+1}]\n"
        context += f"{r['title']}\n"
        context += f"{r['snippet']}\n"
        context += f"{r['url']}\n\n"

    return context