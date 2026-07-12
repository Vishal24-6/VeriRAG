"""
circuit_breaker.py — Circuit Breaker Pattern

Design Pattern: Circuit Breaker (from Martin Fowler / Michael Nygard)
  Prevents cascading failures when external APIs (LLM, web search) go down.
  Instead of hammering a failing service, it "trips" and fails fast.

States:
  CLOSED  → Normal operation, requests pass through
  OPEN    → Service is down, fail immediately (no API calls)
  HALF_OPEN → Test with one request to see if service recovered

Why:
  - Groq API has rate limits → hammering it causes 429 errors
  - Brave/DuckDuckGo can go down → waiting for timeout wastes time
  - Circuit breaker detects failure patterns and fails fast
"""

import time
import logging
from enum import Enum
from typing import Callable, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests go through
    OPEN = "open"           # Tripped — fail immediately
    HALF_OPEN = "half_open" # Testing — allow one request


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for external API calls.

    Usage:
        breaker = CircuitBreaker(name="groq_api", failure_threshold=3)

        result = breaker.call(lambda: api_request())
        # If API fails 3 times → breaker opens → subsequent calls fail fast
        # After recovery_timeout seconds → tries one request to test
    """

    name: str
    failure_threshold: int = 3       # failures before tripping
    recovery_timeout: float = 30.0   # seconds before trying again
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _success_count: int = field(default=0, init=False)

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute func through the circuit breaker.
        Raises RuntimeError if circuit is open.
        """
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info(f"[{self.name}] Circuit half-open — testing recovery")
                self._state = CircuitState.HALF_OPEN
            else:
                remaining = self.recovery_timeout - (time.time() - self._last_failure_time)
                logger.warning(
                    f"[{self.name}] Circuit OPEN — failing fast "
                    f"(retry in {remaining:.0f}s)"
                )
                raise RuntimeError(
                    f"Circuit breaker [{self.name}] is open. "
                    f"Service unavailable, retrying in {remaining:.0f}s."
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self):
        """Record successful call."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"[{self.name}] Service recovered — circuit CLOSED")
        self._failure_count = 0
        self._success_count += 1
        self._state = CircuitState.CLOSED

    def _on_failure(self, error: Exception):
        """Record failed call. Trip breaker if threshold reached."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                f"[{self.name}] Circuit OPEN after {self._failure_count} failures: {error}"
            )
        else:
            logger.warning(
                f"[{self.name}] Failure {self._failure_count}/{self.failure_threshold}: {error}"
            )

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "failures": self._failure_count,
            "successes": self._success_count,
            "threshold": self.failure_threshold,
        }

    def reset(self):
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0


# ──────────────────────────────────────────────────────────
# Pre-configured breakers for each external service
# ──────────────────────────────────────────────────────────

llm_breaker = CircuitBreaker(name="groq_llm", failure_threshold=3, recovery_timeout=30)
web_breaker = CircuitBreaker(name="web_search", failure_threshold=3, recovery_timeout=60)
