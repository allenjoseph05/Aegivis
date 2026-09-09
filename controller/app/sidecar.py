"""Build JSON Patch for sidecar injection into a Pod.

Strategy: env-var injection (no privilege escalation required).
  - Adds the aegivis-proxy sidecar container
  - Patches every existing container with OPENAI_BASE_URL + ANTHROPIC_BASE_URL
    pointing to localhost:proxy_port so all major AI SDKs auto-route through
    the proxy without any code changes.
  - Adds an annotation marking the pod as injected (idempotency guard).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .config import settings

logger = logging.getLogger(__name__)

INJECTED_ANNOTATION = "aegivis.io/injected"
POLICY_ANNOTATION   = "aegivis.io/policy"

# Env vars injected into the AI SDK containers.
# Each entry is (ENV_VAR_NAME, value).  The proxy_port is from the policy spec
# (default: 8080).  SDKs that read a well-known env var are redirected
# automatically; others require the user to pass base_url=os.environ[...].
_REDIRECT_ENVS = [
    # ── Core providers ────────────────────────────────────────────────────────
    ("OPENAI_BASE_URL",       f"http://localhost:{settings.proxy_port}/openai"),
    ("ANTHROPIC_BASE_URL",    f"http://localhost:{settings.proxy_port}/anthropic"),
    ("GOOGLE_AI_BASE_URL",    f"http://localhost:{settings.proxy_port}/google"),
    # ── OpenAI-compatible providers (Phase E1) ────────────────────────────────
    # groq-python SDK reads GROQ_BASE_URL
    ("GROQ_BASE_URL",         f"http://localhost:{settings.proxy_port}/groq"),
    # mistralai SDK reads MISTRAL_SERVER_URL
    ("MISTRAL_SERVER_URL",    f"http://localhost:{settings.proxy_port}/mistral"),
    # together SDK / OpenAI SDK reads TOGETHER_BASE_URL
    ("TOGETHER_BASE_URL",     f"http://localhost:{settings.proxy_port}/together"),
    # deepseek via OpenAI SDK reads DEEPSEEK_BASE_URL
    ("DEEPSEEK_BASE_URL",     f"http://localhost:{settings.proxy_port}/deepseek"),
    # fireworks-ai SDK reads FIREWORKS_API_BASE
    ("FIREWORKS_API_BASE",    f"http://localhost:{settings.proxy_port}/fireworks"),
    # openrouter via OpenAI SDK
    ("OPENROUTER_BASE_URL",   f"http://localhost:{settings.proxy_port}/openrouter"),
    # cerebras-cloud-sdk / OpenAI SDK
    ("CEREBRAS_BASE_URL",     f"http://localhost:{settings.proxy_port}/cerebras"),
    # sambanova via OpenAI SDK
    ("SAMBANOVA_URL",         f"http://localhost:{settings.proxy_port}/sambanova"),
    # xAI grok via OpenAI SDK reads XAI_API_BASE
    ("XAI_API_BASE",          f"http://localhost:{settings.proxy_port}/xai"),
    # NVIDIA NIM via OpenAI SDK
    ("NVIDIA_BASE_URL",       f"http://localhost:{settings.proxy_port}/nvidia"),
    # Cohere compatibility endpoint
    ("COHERE_BASE_URL",       f"http://localhost:{settings.proxy_port}/cohere"),
    # ── Google Vertex AI (Phase E3) ───────────────────────────────────────────
    # google-cloud-aiplatform reads GOOGLE_CLOUD_AIPLATFORM_ENDPOINT.
    # google-genai unified SDK reads GOOGLE_GENAI_API_ENDPOINT.
    # Both redirect all Vertex AI REST traffic through the proxy.
    ("GOOGLE_CLOUD_AIPLATFORM_ENDPOINT", f"http://localhost:{settings.proxy_port}/vertex"),
    ("GOOGLE_GENAI_API_ENDPOINT",        f"http://localhost:{settings.proxy_port}/vertex"),
    # ── AWS Bedrock (Phase E2) ────────────────────────────────────────────────
    # boto3 ≥1.28 reads AWS_ENDPOINT_URL_BEDROCK and redirects all bedrock-runtime
    # calls to the proxy without any code changes in the agent.
    ("AWS_ENDPOINT_URL_BEDROCK", f"http://localhost:{settings.proxy_port}/bedrock"),
    # ── Aegivis internal ──────────────────────────────────────────────────────
    ("AEGIVIS_PROXY_URL",     f"http://localhost:{settings.proxy_port}"),
]


def _sidecar_container(proxy_spec: dict) -> dict:
    """Build the aegivis-proxy sidecar container spec."""
    image   = proxy_spec.get("image", settings.proxy_image)
    port    = int(proxy_spec.get("port", settings.proxy_port))
    res_req = proxy_spec.get("resources", {}).get("requests", {"cpu": "100m", "memory": "128Mi"})
    res_lim = proxy_spec.get("resources", {}).get("limits",   {"cpu": "500m", "memory": "256Mi"})

    return {
        "name":            "aegivis-proxy",
        "image":           image,
        "imagePullPolicy": settings.proxy_image_pull_policy,
        "ports": [{"containerPort": port, "name": "proxy", "protocol": "TCP"}],
        "env": [
            {"name": "AEGIVIS_SIDECAR",    "value": "true"},
            {"name": "AEGIVIS_BACKEND_URL", "value": settings.backend_url},
            {"name": "AEGIVIS_BACKEND_API_KEY", "value": settings.backend_api_key},
            # Bind proxy to all interfaces so the other containers can reach it
            {"name": "AEGIVIS_HOST",       "value": "0.0.0.0"},
            {"name": "AEGIVIS_PORT",       "value": str(port)},
        ],
        "resources": {"requests": res_req, "limits": res_lim},
        "readinessProbe": {
            "httpGet": {"path": "/health", "port": port},
            "initialDelaySeconds": 5,
            "periodSeconds": 10,
        },
        "livenessProbe": {
            "httpGet": {"path": "/health", "port": port},
            "initialDelaySeconds": 10,
            "periodSeconds": 30,
        },
    }


def build_patch(
    pod: dict[str, Any],
    policy_name: str,
    proxy_spec: dict,
) -> list[dict]:
    """Return a JSON Patch list to inject the sidecar into the pod."""
    patch: list[dict] = []

    metadata  = pod.get("metadata") or {}
    spec      = pod.get("spec") or {}
    containers = spec.get("containers") or []
    annotations = metadata.get("annotations") or {}

    # ── Ensure annotations dict exists ───────────────────────────────────────
    if not metadata.get("annotations"):
        patch.append({"op": "add", "path": "/metadata/annotations", "value": {}})

    patch.append({
        "op": "add",
        "path": f"/metadata/annotations/{INJECTED_ANNOTATION.replace('/', '~1')}",
        "value": "true",
    })
    patch.append({
        "op": "add",
        "path": f"/metadata/annotations/{POLICY_ANNOTATION.replace('/', '~1')}",
        "value": policy_name,
    })

    # ── Patch existing containers: add redirect env vars ─────────────────────
    for i, container in enumerate(containers):
        # Skip if this is already the sidecar (shouldn't happen, but guard)
        if container.get("name") == "aegivis-proxy":
            continue

        existing_env = container.get("env")
        if existing_env is None:
            # Container has no env array yet — initialise it
            patch.append({
                "op":    "add",
                "path":  f"/spec/containers/{i}/env",
                "value": [],
            })

        # Add each redirect env var (only if not already set by the user)
        existing_names = {e["name"] for e in (existing_env or [])}
        for env_name, env_value in _REDIRECT_ENVS:
            if env_name not in existing_names:
                patch.append({
                    "op":    "add",
                    "path":  f"/spec/containers/{i}/env/-",
                    "value": {"name": env_name, "value": env_value},
                })

    # ── Add sidecar container ─────────────────────────────────────────────────
    sidecar = _sidecar_container(proxy_spec)
    if not containers:
        # No containers yet (shouldn't happen but be safe)
        patch.append({"op": "add", "path": "/spec/containers", "value": [sidecar]})
    else:
        patch.append({"op": "add", "path": "/spec/containers/-", "value": sidecar})

    logger.debug(
        "Built sidecar patch for pod (policy=%s): %d operations",
        policy_name, len(patch),
    )
    return patch


def is_already_injected(pod: dict[str, Any]) -> bool:
    annotations = (pod.get("metadata") or {}).get("annotations") or {}
    return annotations.get(INJECTED_ANNOTATION) == "true"


def labels_match(pod: dict[str, Any], match_labels: dict[str, str]) -> bool:
    """Return True if all match_labels exist in pod labels."""
    pod_labels = (pod.get("metadata") or {}).get("labels") or {}
    return all(pod_labels.get(k) == v for k, v in match_labels.items())
