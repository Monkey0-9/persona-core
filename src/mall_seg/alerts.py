"""Safe drift threshold alerting hook with SSRF-validated webhook delivery."""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("mall_seg.alerts")


def is_safe_webhook_url(raw_url: str) -> bool:
    """Validate webhook URL against https/http scheme and non-private/non-loopback hosts."""
    try:
        parsed = urlparse(raw_url.strip())
    except ValueError:
        return False
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost", "127.0.0.1", "::1", "169.254.169.254"}:
        return False
    try:
        ip_obj = ipaddress.ip_address(host)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            return False
        return True
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        for info in infos:
            resolved_ip = ipaddress.ip_address(info[4][0])
            if (
                resolved_ip.is_private
                or resolved_ip.is_loopback
                or resolved_ip.is_link_local
                or resolved_ip.is_reserved
            ):
                return False
    except OSError:
        return False
    return True


def dispatch_drift_alert(
    *,
    alert_status: str,
    max_psi: float,
    max_ks_statistic: float,
    window_label: str,
    sample_size: int,
    details: dict[str, Any] | None = None,
) -> bool:
    """Emit structured drift alert log and optionally POST to SSRF-validated webhook."""
    if alert_status not in {"WARNING", "CRITICAL"}:
        return False

    log_fn = logger.error if alert_status == "CRITICAL" else logger.warning
    log_fn(
        "Drift threshold breached (%s): max_psi=%.4f max_ks=%.4f window=%s n=%d",
        alert_status,
        max_psi,
        max_ks_statistic,
        window_label,
        sample_size,
        extra={
            "alert_status": alert_status,
            "max_psi": max_psi,
            "max_ks_statistic": max_ks_statistic,
            "window_label": window_label,
            "sample_size": sample_size,
        },
    )

    webhook_url = os.getenv("DRIFT_ALERT_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return True

    if not is_safe_webhook_url(webhook_url):
        logger.warning("Blocked unsafe DRIFT_ALERT_WEBHOOK_URL target.")
        return False

    payload = {
        "event": "mall_seg.drift_alert",
        "alert_status": alert_status,
        "max_psi": max_psi,
        "max_ks_statistic": max_ks_statistic,
        "window_label": window_label,
        "sample_size": sample_size,
        "details": details or {},
    }
    try:
        resp = httpx.post(webhook_url, json=payload, timeout=3.0)
        return resp.status_code < 400
    except Exception as exc:
        logger.warning("Drift alert webhook delivery failed: %s", exc)
        return False
