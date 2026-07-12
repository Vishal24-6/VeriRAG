"""
models.py — Singleton Model Manager

Design Pattern: Singleton
  Ensures exactly ONE instance of each ML model exists in memory.
  All modules access models through this manager instead of creating their own.

Why:
  - ML models are expensive to load (hundreds of MB)
  - Multiple instances waste RAM and cause OOM errors
  - Centralized access makes testing and swapping models easy
"""

import logging
import threading

import config

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Thread-safe singleton for managing ML model instances.

    Usage:
        manager = ModelManager()
        bi_encoder = manager.bi_encoder
        cross_encoder = manager.cross_encoder
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._bi_encoder = None
        self._cross_encoder = None
        self._initialized = True
        logger.info("ModelManager initialized")

    @property
    def bi_encoder(self):
        """Lazy-load bi-encoder for embedding generation."""
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading bi-encoder: {config.BI_ENCODER_MODEL}")
            self._bi_encoder = SentenceTransformer(config.BI_ENCODER_MODEL)
            logger.info("Bi-encoder loaded successfully")
        return self._bi_encoder

    @property
    def cross_encoder(self):
        """Lazy-load cross-encoder for reranking."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {config.CROSS_ENCODER_MODEL}")
            self._cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)
            logger.info("Cross-encoder loaded successfully")
        return self._cross_encoder

    def preload(self):
        """Preload all models (useful for Docker/startup)."""
        _ = self.bi_encoder
        _ = self.cross_encoder
        logger.info("All models preloaded")

    @classmethod
    def reset(cls):
        """Reset singleton (for testing only)."""
        with cls._lock:
            cls._instance = None
