"""Aegivis Kubernetes Controller — entry point.

Runs two concurrent processes:
  1. kopf operator (this process) — watches AgentSecurityPolicy CRDs
  2. FastAPI admission webhook server (subprocess) — HTTPS on port 8443

TLS certs are generated at startup and the MutatingWebhookConfiguration is
patched with the CA bundle so the K8s API server trusts the webhook.
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
from pathlib import Path

import kopf
import uvicorn
from kubernetes import config as k8s_config

from .certs import generate_certs, patch_webhook_ca_bundle
from .config import settings

# Import reconciler to register kopf handlers
from . import reconciler  # noqa: F401

logger = logging.getLogger(__name__)


def _run_webhook_server(cert_path: str, key_path: str) -> None:
    """Run the FastAPI webhook server in a subprocess."""
    from .webhook import app

    logging.basicConfig(level=settings.log_level.upper())
    uvicorn.run(
        app,
        host=settings.webhook_host,
        port=settings.webhook_port,
        ssl_certfile=cert_path,
        ssl_keyfile=key_path,
        log_level=settings.log_level.lower(),
    )


def _load_k8s_config() -> None:
    """Load K8s config from in-cluster or local kubeconfig."""
    try:
        k8s_config.load_incluster_config()
        logger.info("Loaded in-cluster K8s config")
    except Exception:
        k8s_config.load_kube_config()
        logger.info("Loaded local kubeconfig")


def main() -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    )

    _load_k8s_config()

    # ── Generate TLS certs ────────────────────────────────────────────────────
    logger.info("Generating TLS certificates in %s", settings.cert_dir)
    cert_path, key_path, ca_der = generate_certs(
        cert_dir=settings.cert_dir,
        service_name=settings.webhook_service_name,
        namespace=settings.webhook_service_namespace,
    )

    # ── Patch MutatingWebhookConfiguration with CA bundle ────────────────────
    patch_webhook_ca_bundle(settings.webhook_config_name, ca_der)

    # ── Start webhook server in a subprocess ─────────────────────────────────
    webhook_proc = multiprocessing.Process(
        target=_run_webhook_server,
        args=(str(cert_path), str(key_path)),
        daemon=True,
        name="aegivis-webhook",
    )
    webhook_proc.start()
    logger.info(
        "Webhook server started (PID=%d) on %s:%d",
        webhook_proc.pid,
        settings.webhook_host,
        settings.webhook_port,
    )

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    def _shutdown(signum, frame):
        logger.info("Received signal %d — shutting down", signum)
        if webhook_proc.is_alive():
            webhook_proc.terminate()
            webhook_proc.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # ── Run kopf operator (blocks until stopped) ──────────────────────────────
    namespace = settings.watch_namespace or None  # None = all namespaces
    kopf.run(
        namespace=namespace,
        clusterwide=(namespace is None),
    )


if __name__ == "__main__":
    main()
