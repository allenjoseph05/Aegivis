"""Controller settings — all configurable via AEGIVIS_CTRL_* env vars."""
from __future__ import annotations

from pydantic_settings import BaseSettings


class ControllerSettings(BaseSettings):
    # Kubernetes namespace to watch (empty = all namespaces / cluster-scoped)
    watch_namespace: str = ""

    # Webhook server
    webhook_host: str = "0.0.0.0"
    webhook_port: int = 8443
    # K8s Service name for this controller (used to build webhook clientConfig URL)
    webhook_service_name: str = "aegivis-controller"
    webhook_service_namespace: str = "aegivis-system"
    # Name of the MutatingWebhookConfiguration to patch with CA bundle
    webhook_config_name: str = "aegivis-sidecar-injector"

    # Proxy sidecar image injected into agent pods
    proxy_image: str = "agentblackbox-proxy:latest"
    proxy_image_pull_policy: str = "IfNotPresent"
    proxy_port: int = 8080

    # Backend URL + API key injected as env vars into the sidecar
    backend_url: str = "http://aegivis-backend.aegivis-system.svc.cluster.local:8000"
    backend_api_key: str = "dev-proxy-key"

    # TLS cert dir (generated at startup, lives in container memory)
    cert_dir: str = "/tmp/aegivis-certs"

    log_level: str = "INFO"

    model_config = {"env_prefix": "AEGIVIS_CTRL_"}


settings = ControllerSettings()
