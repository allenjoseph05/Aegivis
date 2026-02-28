"""
Prometheus Metrics — Backend (Phase 3.4)

Exposes /metrics endpoint for Prometheus scraping.

Metrics exported:
  abb_events_ingested_total   -- Events ingested by event_type
  abb_anomalies_detected_total -- Anomalies detected by rule_id and severity
  abb_api_request_duration_ms  -- API request duration by endpoint

Optional dependency: prometheus-client>=0.21.0
Install: pip install 'agentblackbox-backend[observability]'
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_METRICS_AVAILABLE = False
_registry = None


class _Noop:
    def labels(self, **kwargs): return self
    def inc(self, amount=1): pass
    def observe(self, amount): pass
    def set(self, value): pass
    def __call__(self, *args, **kwargs): return self


_noop = _Noop()

events_ingested_total: _Noop = _noop     # type: ignore[assignment]
anomalies_detected_total: _Noop = _noop  # type: ignore[assignment]
api_request_duration_ms: _Noop = _noop   # type: ignore[assignment]


def _init_metrics():
    global _METRICS_AVAILABLE, _registry
    global events_ingested_total, anomalies_detected_total, api_request_duration_ms

    try:
        from prometheus_client import (
            Counter,
            Histogram,
            CollectorRegistry,
        )

        _registry = CollectorRegistry()

        events_ingested_total = Counter(
            "abb_events_ingested_total",
            "Total events ingested by the backend",
            ["event_type"],
            registry=_registry,
        )

        anomalies_detected_total = Counter(
            "abb_anomalies_detected_total",
            "Total anomalies detected",
            ["rule_id", "severity"],
            registry=_registry,
        )

        api_request_duration_ms = Histogram(
            "abb_api_request_duration_ms",
            "API request duration in milliseconds",
            ["endpoint"],
            buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 5000],
            registry=_registry,
        )

        _METRICS_AVAILABLE = True
        logger.info("Prometheus metrics initialized (backend)")

    except ImportError:
        logger.info(
            "prometheus-client not installed -- metrics endpoint disabled. "
            "Install with: pip install 'agentblackbox-backend[observability]'"
        )
    except Exception as exc:
        logger.warning("Failed to initialize Prometheus metrics: %s", exc)


def get_metrics_app():
    """Return Prometheus ASGI app or None if unavailable."""
    if not _METRICS_AVAILABLE or _registry is None:
        return None
    try:
        from prometheus_client import make_asgi_app
        return make_asgi_app(registry=_registry)
    except Exception as exc:
        logger.warning("Failed to create metrics ASGI app: %s", exc)
        return None


_init_metrics()
