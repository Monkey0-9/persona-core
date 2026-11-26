"""Sliding-window rate limiter with Redis support and fail-safe in-memory fallback."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any

from fastapi import HTTPException, Request, status

logger = logging.getLogger("mall_seg.rate_limit")


class SlidingWindowRateLimiter:
    """Rate limiter using Redis ZSET when configured and healthy, otherwise in-memory."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._redis_client: Any | None = None
        self._redis_checked = False

    def _get_redis(self) -> Any | None:
        if self._redis_checked:
            return self._redis_client
        self._redis_checked = True
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            return None
        try:
            import redis

            client = redis.Redis.from_url(redis_url, socket_timeout=0.5, decode_responses=True)
            client.ping()
            self._redis_client = client
            return client
        except Exception as exc:
            logger.warning("Redis unavailable (%s); falling back to local rate limiting.", exc)
            self._redis_client = None
            return None

    def allow_request(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        redis_client = self._get_redis()
        if redis_client is not None:
            try:
                pipe = redis_client.pipeline()
                cutoff = now - window_seconds
                pipe.zremrangebyscore(key, 0, cutoff)
                pipe.zadd(key, {f"{now:.6f}": now})
                pipe.zcard(key)
                pipe.expire(key, window_seconds + 1)
                results = pipe.execute()
                count = int(results[2])
                return count <= max_requests
            except Exception as exc:
                logger.warning("Redis rate-limit check failed (%s); falling back in-memory.", exc)

        with self._lock:
            bucket = self._buckets[key]
            cutoff = now - window_seconds
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= max_requests:
                return False
            bucket.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


rate_limiter = SlidingWindowRateLimiter()


def check_redis_connectivity() -> bool | None:
    """Return True if REDIS_URL responds to ping, False if unreachable, None if unconfigured."""
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        return None
    client = rate_limiter._get_redis()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception:
        return False


def enforce_rate_limit(
    request: Request,
    *,
    scope: str,
    max_requests: int = 60,
    window_seconds: int = 60,
) -> None:
    """Enforce per-client sliding-window rate limits and return 429 cleanly."""
    if os.getenv("MALL_SEG_DISABLE_RATE_LIMIT", "0") == "1":
        return
    override_max = os.getenv(f"MALL_SEG_RL_{scope.upper()}_MAX", "").strip()
    effective_max = int(override_max) if override_max.isdigit() else max_requests
    client_host = request.client.host if request.client else "unknown"
    key = f"rl:{scope}:{client_host}"
    allowed = rate_limiter.allow_request(
        key,
        max_requests=effective_max,
        window_seconds=window_seconds,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry shortly.",
        )
