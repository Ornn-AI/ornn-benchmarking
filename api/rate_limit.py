"""Per-API-key rate limiting for the Ornn Benchmarking API.

Uses an in-memory sliding-window counter.  Each API key gets its own
bucket.  When the limit is exceeded the caller receives HTTP 429 with
a ``Retry-After`` header indicating how many seconds to wait.

The limiter is intentionally simple — a single-process in-memory store
is sufficient for a Cloud Run service where each instance handles a
modest request rate.  For high-scale deployments, swap in a Redis or
Firestore-backed implementation behind :class:`RateLimiter`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    """Sliding-window request timestamps for a single key."""

    timestamps: list[float] = field(default_factory=list)


class RateLimiter:
    """In-memory per-key sliding-window rate limiter.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed within *window_seconds*.
    window_seconds:
        Length of the sliding window in seconds.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """Check whether *key* is within the rate limit.

        Returns
        -------
        (allowed, retry_after)
            *allowed* is ``True`` when the request may proceed.
            *retry_after* is ``0`` when allowed, otherwise the number of
            whole seconds the caller should wait before retrying.
        """
        now = time.monotonic()
        window_start = now - self.window_seconds

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket()
                self._buckets[key] = bucket

            # Prune expired timestamps
            bucket.timestamps = [
                ts for ts in bucket.timestamps if ts > window_start
            ]

            if len(bucket.timestamps) >= self.max_requests:
                # Compute retry-after from the oldest timestamp in the window
                oldest = bucket.timestamps[0]
                retry_after = int(oldest + self.window_seconds - now) + 1
                return False, max(retry_after, 1)

            bucket.timestamps.append(now)
            return True, 0

    def reset(self) -> None:
        """Clear all rate-limit state (used by tests)."""
        with self._lock:
            self._buckets.clear()
