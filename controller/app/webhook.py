"""FastAPI MutatingAdmissionWebhook for Aegivis sidecar injection.

Receives AdmissionReview requests from the K8s API server and injects
the Aegivis proxy sidecar into matching agent pods.

Endpoint: POST /webhook (HTTPS, port 8443)
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kubernetes import client as k8s_client

from .sidecar import build_patch, is_already_injected, labels_match

logger = logging.getLogger(__name__)

app = FastAPI(title="Aegivis Admission Webhook", docs_url=None, redoc_url=None)


# ── Policy cache (refreshed per request — low-volume webhook) ─────────────────

def _list_policies(namespace: str) -> list[dict[str, Any]]:
    """List all AgentSecurityPolicies in the given namespace."""
    try:
        custom_api = k8s_client.CustomObjectsApi()
        result = custom_api.list_namespaced_custom_object(
            group="aegivis.io",
            version="v1",
            namespace=namespace,
            plural="agentsecuritypolicies",
        )
        return result.get("items") or []
    except Exception as exc:
        logger.warning("Could not list AgentSecurityPolicies in %s: %s", namespace, exc)
        return []


def _find_matching_policy(
    pod: dict[str, Any],
    namespace: str,
) -> tuple[str, dict[str, Any]] | None:
    """Return (policy_name, policy_spec) for the first matching policy, or None."""
    for policy in _list_policies(namespace):
        spec = policy.get("spec") or {}
        selector = spec.get("selector") or {}
        match_labels = selector.get("matchLabels") or {}
        # Empty matchLabels = match all pods in namespace
        if not match_labels or labels_match(pod, match_labels):
            policy_name = (policy.get("metadata") or {}).get("name", "unknown")
            return policy_name, spec
    return None


# ── Admission response builders ───────────────────────────────────────────────

def _allow(uid: str, message: str = "") -> dict:
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": True,
            "status": {"message": message} if message else {},
        },
    }


def _patch(uid: str, patch_ops: list[dict]) -> dict:
    patch_bytes = base64.b64encode(
        json.dumps(patch_ops).encode()
    ).decode()
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": True,
            "patchType": "JSONPatch",
            "patch": patch_bytes,
        },
    }


def _deny(uid: str, message: str) -> dict:
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": False,
            "status": {"message": message},
        },
    }


# ── Main webhook endpoint ─────────────────────────────────────────────────────

@app.post("/webhook")
async def admission_webhook(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception as exc:
        logger.error("Failed to parse AdmissionReview: %s", exc)
        return JSONResponse({"error": "invalid body"}, status_code=400)

    adm_request = body.get("request") or {}
    uid       = adm_request.get("uid", "")
    pod       = adm_request.get("object") or {}
    namespace = adm_request.get("namespace") or (pod.get("metadata") or {}).get("namespace", "default")
    operation = adm_request.get("operation", "")

    # Only handle CREATE operations for Pods
    if operation not in ("CREATE",):
        return JSONResponse(_allow(uid, f"operation {operation} not handled"))

    # Skip system namespaces
    if namespace in ("kube-system", "kube-public"):
        return JSONResponse(_allow(uid, "system namespace skipped"))

    # Idempotency: skip if already injected
    if is_already_injected(pod):
        return JSONResponse(_allow(uid, "already injected"))

    # Find a matching AgentSecurityPolicy
    match = _find_matching_policy(pod, namespace)
    if match is None:
        return JSONResponse(_allow(uid, "no matching AgentSecurityPolicy"))

    policy_name, policy_spec = match
    inject = policy_spec.get("inject", True)

    if not inject:
        return JSONResponse(_allow(uid, f"inject=false in policy {policy_name}"))

    # Build the sidecar patch
    try:
        patch_ops = build_patch(
            pod=pod,
            policy_name=policy_name,
            proxy_spec=policy_spec.get("proxy") or {},
        )
    except Exception as exc:
        logger.error("Failed to build sidecar patch: %s", exc)
        return JSONResponse(_allow(uid, f"patch build failed (non-blocking): {exc}"))

    pod_name = (pod.get("metadata") or {}).get("name") or "(unnamed)"
    logger.info(
        "Injecting sidecar into pod %s/%s (policy=%s, ops=%d)",
        namespace, pod_name, policy_name, len(patch_ops),
    )
    return JSONResponse(_patch(uid, patch_ops))


# ── Health endpoint (also answers liveness probe) ────────────────────────────

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
