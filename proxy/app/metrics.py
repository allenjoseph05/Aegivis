"""
Prometheus Metrics — Phase 3.4

Exposes an ASGI metrics endpoint at /metrics for Prometheus scraping.

Metrics exported:
  aegivis_requests_total          -- LLM proxy requests by provider and status
  aegivis_violations_total        -- Policy/security violations by rule and action
  aegivis_security_scans_total    -- Security scan invocations by scanner name
  aegivis_events_enqueued_total   -- Events enqueued for backend by event_type
  aegivis_llm_latency_ms          -- LLM call latency histogram by provider/model
  aegivis_security_scan_ms        -- Security scan duration histogram by scanner
  aegivis_active_sessions         -- Current active sessions gauge

Optional dependency: prometheus-client>=0.21.0
Install: pip install 'aegivis-proxy[observability]'

If prometheus-client is not installed, all operations are no-ops and
get_metrics_app() returns None.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_METRICS_AVAILABLE = False
_registry = None

# Counter/Histogram/Gauge stubs for graceful degradation
class _Noop:
    """No-op metric stub when prometheus-client is not installed."""
    def labels(self, **kwargs): return self
    def inc(self, amount=1): pass
    def observe(self, amount): pass
    def set(self, value): pass
    def __call__(self, *args, **kwargs): return self


_noop = _Noop()

# Public metric handles -- replaced with real metrics if prometheus-client available
requests_total: _Noop = _noop           # type: ignore[assignment]
violations_total: _Noop = _noop         # type: ignore[assignment]
security_scans_total: _Noop = _noop     # type: ignore[assignment]
events_enqueued_total: _Noop = _noop    # type: ignore[assignment]
llm_latency_ms: _Noop = _noop          # type: ignore[assignment]
security_scan_ms: _Noop = _noop        # type: ignore[assignment]
active_sessions: _Noop = _noop         # type: ignore[assignment]


def _init_metrics():
    global _METRICS_AVAILABLE, _registry
    global requests_total, violations_total, security_scans_total
    global events_enqueued_total, llm_latency_ms, security_scan_ms, active_sessions

    try:
        from prometheus_client import (
            Counter,
            Gauge,
            Histogram,
            CollectorRegistry,
        )

        _registry = CollectorRegistry()

        requests_total = Counter(
            "aegivis_requests_total",
            "Total LLM proxy requests",
            ["provider", "status"],
            registry=_registry,
        )

        violations_total = Counter(
            "aegivis_violations_total",
            "Total policy and security violations",
            ["rule_name", "action"],
            registry=_registry,
        )

        security_scans_total = Counter(
            "aegivis_security_scans_total",
            "Total security scan invocations",
            ["scanner"],
            registry=_registry,
        )

        events_enqueued_total = Counter(
            "aegivis_events_enqueued_total",
            "Total events enqueued for backend",
            ["event_type"],
            registry=_registry,
        )

        llm_latency_ms = Histogram(
            "aegivis_llm_latency_ms",
            "LLM call latency in milliseconds",
            ["provider", "model"],
            buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
            registry=_registry,
        )

        security_scan_ms = Histogram(
            "aegivis_security_scan_duration_ms",
            "Security scan duration in milliseconds",
            ["scanner"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=_registry,
        )

        active_sessions = Gauge(
            "aegivis_active_sessions",
            "Number of currently active agent sessions",
            registry=_registry,
        )

        _METRICS_AVAILABLE = True
        logger.info("Prometheus metrics initialized")

    except ImportError:
        logger.info(
            "prometheus-client not installed -- metrics endpoint disabled. "
            "Install with: pip install 'aegivis-proxy[observability]'"
        )
    except Exception as exc:
        logger.warning("Failed to initialize Prometheus metrics: %s", exc)


def get_metrics_app():
    """
    Return a Prometheus ASGI app to mount at /metrics.
    Returns None if prometheus-client is not available.
    """
    if not _METRICS_AVAILABLE or _registry is None:
        return None
    try:
        from prometheus_client import make_asgi_app
        return make_asgi_app(registry=_registry)
    except Exception as exc:
        logger.warning("Failed to create metrics ASGI app: %s", exc)
        return None


# Initialize on module import
_init_metrics()
