"""
OpenTelemetry Distributed Tracing — Backend (Phase 3.4)

Mirrors proxy/app/tracing.py for the backend service.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_OTEL_AVAILABLE = False


def setup_tracing(
    service_name: str = "agentblackbox-backend",
    endpoint: str = "http://localhost:4317",
) -> bool:
    """Initialize OTel tracer. Returns True if successful."""
    global _OTEL_AVAILABLE

    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry import trace

        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
        })

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _OTEL_AVAILABLE = True
        logger.info(
            "OpenTelemetry tracing initialized: service=%s endpoint=%s",
            service_name, endpoint,
        )
        return True

    except ImportError:
        logger.info(
            "opentelemetry packages not installed -- tracing disabled. "
            "Install with: pip install 'agentblackbox-backend[observability]'"
        )
        return False
    except Exception as exc:
        logger.warning("Failed to initialize OTel tracing: %s", exc)
        return False


def get_tracer(name: str = "agentblackbox"):
    """Return real tracer or no-op tracer."""
    if _OTEL_AVAILABLE:
        try:
            from opentelemetry import trace
            return trace.get_tracer(name)
        except Exception:
            pass
    return _NoOpTracer()


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, key, value): pass
    def set_status(self, status): pass
    def record_exception(self, exc): pass


class _NoOpTracer:
    def start_as_current_span(self, name, *, attributes=None, **kwargs):
        return _NoOpSpan()

    def start_span(self, name, *, attributes=None, **kwargs):
        return _NoOpSpan()
