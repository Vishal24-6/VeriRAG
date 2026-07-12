"""
cache.py — Caching Layer

Design Pattern: Decorator / Memoization
  Wraps expensive operations with caching to avoid redundant computation.

Components:
  - QueryCache: LRU cache for query embeddings (avoids re-encoding same query)
  - IndexCache: Hash-based cache for document indices (avoids re-indexing same docs)
"""

import hashlib
import logging
from functools import lru_cache
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class QueryCache:
    """
    LRU cache for query embeddings.

    Why: The same query gets encoded multiple times during dual-query retrieval
    and drift checking. Caching avoids redundant model inference.
    """

    def __init__(self, maxsize: int = 128):
        self._cache = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def get(self, query: str):
        """Get cached embedding for a query."""
        key = self._normalize(query)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, query: str, embedding):
        """Store embedding in cache."""
        key = self._normalize(query)
        # Evict oldest if full
        if len(self._cache) >= self._maxsize:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = embedding

    def _normalize(self, text: str) -> str:
        return text.strip().lower()

    @property
    def stats(self) -> dict:
        """Cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 2) if total > 0 else 0.0,
            "size": len(self._cache),
        }

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


def content_hash(documents: dict) -> str:
    """
    Compute a deterministic hash for a set of documents.
    Used to check if the index needs rebuilding.
    """
    content = "".join(f"{k}:{v}" for k, v in sorted(documents.items()))
    return hashlib.md5(content.encode()).hexdigest()
