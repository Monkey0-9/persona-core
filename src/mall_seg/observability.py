"""Structured JSON logging with request_id correlation and Prometheus metrics registry."""

from __future__ import annotations

import contextvars
import json
import logging
import math
import threading
from datetime import UTC, datetime
from typing import Any

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


class JsonLogFormatter(logging.Formatter):
    """Emit structured single-line JSON logs with correlation request_id."""

    _STANDARD_ATTRS = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", request_id_ctx.get()),
            "message": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key not in self._STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_structured_logging(level: int = logging.INFO) -> None:
    """Attach JSON formatter to the mall_seg logger namespace cleanly."""
    root_pkg_logger = logging.getLogger("mall_seg")
    root_pkg_logger.setLevel(level)
    if not any(isinstance(h.formatter, JsonLogFormatter) for h in root_pkg_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        root_pkg_logger.handlers = [handler]
        root_pkg_logger.propagate = False


class MetricsRegistry:
    """Thread-safe in-process Prometheus exposition metrics collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_counts: dict[tuple[str, str, str], int] = {}
        self._duration_buckets: dict[tuple[str, str, float], int] = {}
        self._duration_sum: dict[tuple[str, str], float] = {}
        self._duration_count: dict[tuple[str, str], int] = {}
        self._recent_latencies_ms: list[float] = []
        self._persona_counts: dict[str, int] = {}
        self._drift_alert_counts: dict[str, int] = {}

    def observe_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        normalized_path = path.split("?")[0]
        status_str = str(status_code)
        dur = max(float(duration_seconds), 0.0)
        with self._lock:
            rk = (method, normalized_path, status_str)
            self._request_counts[rk] = self._request_counts.get(rk, 0) + 1

            hk = (method, normalized_path)
            self._duration_sum[hk] = self._duration_sum.get(hk, 0.0) + dur
            self._duration_count[hk] = self._duration_count.get(hk, 0) + 1
            for bucket in LATENCY_BUCKETS:
                if dur <= bucket:
                    bk = (method, normalized_path, bucket)
                    self._duration_buckets[bk] = self._duration_buckets.get(bk, 0) + 1

            self._recent_latencies_ms.append(dur * 1000.0)
            if len(self._recent_latencies_ms) > 500:
                self._recent_latencies_ms = self._recent_latencies_ms[-500:]

    def inc_persona_score(self, persona: str, count: int = 1) -> None:
        with self._lock:
            self._persona_counts[persona] = self._persona_counts.get(persona, 0) + count

    def inc_drift_alert(self, severity: str, count: int = 1) -> None:
        with self._lock:
            key = severity.upper()
            self._drift_alert_counts[key] = self._drift_alert_counts.get(key, 0) + count

    def summary(
        self,
        *,
        db_up: bool,
        redis_up: bool | None,
        model_loaded: bool,
        active_version: str | None,
    ) -> dict[str, Any]:
        with self._lock:
            total_requests = sum(self._request_counts.values())
            latencies = sorted(self._recent_latencies_ms)
            p95 = _percentile(latencies, 0.95)
            p99 = _percentile(latencies, 0.99)
            return {
                "total_requests": total_requests,
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "scoring_by_persona": dict(self._persona_counts),
                "drift_alerts": dict(self._drift_alert_counts),
                "db_up": db_up,
                "redis_up": redis_up,
                "model_loaded": model_loaded,
                "active_model_version": active_version,
            }

    def render_prometheus(
        self,
        *,
        db_up: bool,
        redis_up: bool | None,
        model_loaded: bool,
        active_version: str | None,
    ) -> str:
        lines: list[str] = []
        with self._lock:
            lines.append("# HELP mall_seg_http_requests_total Total HTTP requests handled.")
            lines.append("# TYPE mall_seg_http_requests_total counter")
            for (method, path, code), count in sorted(self._request_counts.items()):
                lines.append(
                    f'mall_seg_http_requests_total{{method="{_esc(method)}",'
                    f'path="{_esc(path)}",status="{_esc(code)}"}} {count}'
                )

            lines.append(
                "# HELP mall_seg_http_request_duration_seconds HTTP request latency histogram."
            )
            lines.append("# TYPE mall_seg_http_request_duration_seconds histogram")
            for (method, path), total_cnt in sorted(self._duration_count.items()):
                for bucket in LATENCY_BUCKETS:
                    b_cnt = self._duration_buckets.get((method, path, bucket), 0)
                    lines.append(
                        f'mall_seg_http_request_duration_seconds_bucket{{method="{_esc(method)}",'
                        f'path="{_esc(path)}",le="{bucket}"}} {b_cnt}'
                    )
                lines.append(
                    f'mall_seg_http_request_duration_seconds_bucket{{method="{_esc(method)}",'
                    f'path="{_esc(path)}",le="+Inf"}} {total_cnt}'
                )
                d_sum = self._duration_sum.get((method, path), 0.0)
                lines.append(
                    f'mall_seg_http_request_duration_seconds_sum{{method="{_esc(method)}",'
                    f'path="{_esc(path)}"}} {d_sum:.6f}'
                )
                lines.append(
                    f'mall_seg_http_request_duration_seconds_count{{method="{_esc(method)}",'
                    f'path="{_esc(path)}"}} {total_cnt}'
                )

            lines.append(
                "# HELP mall_seg_scoring_persona_total Customers scored per assigned persona."
            )
            lines.append("# TYPE mall_seg_scoring_persona_total counter")
            for persona, count in sorted(self._persona_counts.items()):
                lines.append(f'mall_seg_scoring_persona_total{{persona="{_esc(persona)}"}} {count}')

            lines.append(
                "# HELP mall_seg_drift_alerts_total Drift evaluations reaching WARNING/CRITICAL."
            )
            lines.append("# TYPE mall_seg_drift_alerts_total counter")
            for severity, count in sorted(self._drift_alert_counts.items()):
                lines.append(f'mall_seg_drift_alerts_total{{status="{_esc(severity)}"}} {count}')

            lines.append("# HELP mall_seg_db_up Database connectivity gauge (1=up, 0=down).")
            lines.append("# TYPE mall_seg_db_up gauge")
            lines.append(f"mall_seg_db_up {1 if db_up else 0}")

            lines.append(
                "# HELP mall_seg_redis_up Redis connectivity gauge (1=up, 0=down/unconfigured)."
            )
            lines.append("# TYPE mall_seg_redis_up gauge")
            lines.append(f"mall_seg_redis_up {1 if redis_up else 0}")

            lines.append("# HELP mall_seg_model_loaded Model bundle loaded gauge (1=yes, 0=no).")
            lines.append("# TYPE mall_seg_model_loaded gauge")
            lines.append(f"mall_seg_model_loaded {1 if model_loaded else 0}")

            if active_version:
                lines.append("# HELP mall_seg_active_model_info Active model registry version.")
                lines.append("# TYPE mall_seg_active_model_info gauge")
                lines.append(f'mall_seg_active_model_info{{version="{_esc(active_version)}"}} 1')

        return "\n".join(lines) + "\n"


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return float(sorted_vals[idx])


def _esc(val: str) -> str:
    return val.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")


metrics_registry = MetricsRegistry()
