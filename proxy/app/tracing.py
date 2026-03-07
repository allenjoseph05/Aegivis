"""
OpenTelemetry Distributed Tracing — Phase 3.4

Initializes OTel tracer for the proxy service.

Exports spans to an OTLP endpoint (default: Jaeger on localhost:4317).

Optional dependency: opentelemetry-sdk, opentelemetry-api, opentelemetry-exporter-otlp-proto-grpc
Install: pip install 'aegivis-proxy[observability]'

If opentelemetry packages are not installed, get_tracer() returns a no-op
tracer and all span operations are zero-cost.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_OTEL_AVAILABLE = False
_tracer_provider = None


def setup_tracing(
    service_name: str = "aegivis-proxy",
    endpoint: str = "http://localhost:4317",
) -> bool:
    """
    Initialize OTel tracer with OTLP/gRPC exporter.

    Args:
        service_name: Service name shown in Jaeger/Tempo traces.
        endpoint:     OTLP gRPC endpoint URL.

    Returns:
        True if OTel initialized successfully, False if packages missing.
    """
    global _OTEL_AVAILABLE, _tracer_provider

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

        _tracer_provider = provider
        _OTEL_AVAILABLE = True
        logger.info(
            "OpenTelemetry tracing initialized: service=%s endpoint=%s",
            service_name, endpoint,
        )
        return True

    except ImportError:
        logger.info(
            "opentelemetry packages not installed -- tracing disabled. "
            "Install with: pip install 'aegivis-proxy[observability]'"
        )
        return False
    except Exception as exc:
        logger.warning("Failed to initialize OTel tracing: %s", exc)
        return False


def get_tracer(name: str = "aegivis"):
    """
    Return a tracer instance.

    Returns a real tracer if OTel is configured, otherwise a no-op tracer.
    The no-op tracer's context manager is a no-op so calling code needs
    no conditional guards.
    """
    if _OTEL_AVAILABLE:
        try:
            from opentelemetry import trace
            return trace.get_tracer(name)
        except Exception:
            pass

    # Return a no-op tracer object that supports the context manager protocol
    return _NoOpTracer()


class _NoOpSpan:
    """No-op span that supports the context manager protocol."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_attribute(self, key, value): pass
    def set_status(self, status): pass
    def record_exception(self, exc): pass


class _NoOpTracer:
    """No-op tracer when OTel is not available."""
    def start_as_current_span(self, name, *, attributes=None, **kwargs):
        return _NoOpSpan()

    def start_span(self, name, *, attributes=None, **kwargs):
        return _NoOpSpan()
