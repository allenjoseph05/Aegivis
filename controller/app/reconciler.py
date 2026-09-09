"""kopf handlers for AgentSecurityPolicy CRD.

Handles create / update / delete of AgentSecurityPolicy resources.
For each policy, maintains an owned NetworkPolicy in the same namespace.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import kopf
from kubernetes import client as k8s_client

from .network_policy import build_network_policy, network_policy_name

logger = logging.getLogger(__name__)


def _networking_api() -> k8s_client.NetworkingV1Api:
    return k8s_client.NetworkingV1Api()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Create / Update ───────────────────────────────────────────────────────────

@kopf.on.create("aegivis.io", "v1", "agentsecuritypolicies")
@kopf.on.update("aegivis.io", "v1", "agentsecuritypolicies")
async def reconcile_policy(spec, name, namespace, status, patch, **kwargs):
    logger.info("Reconciling AgentSecurityPolicy %s/%s", namespace, name)

    egress_mode = (spec.get("egress") or {}).get("mode", "audit").lower()
    np_manifest = build_network_policy(name, namespace, spec)

    np_name = network_policy_name(name)

    if not np_manifest:
        # egress.mode = off → delete any existing NetworkPolicy
        try:
            _networking_api().delete_namespaced_network_policy(np_name, namespace)
            logger.info("Deleted NetworkPolicy %s/%s (mode=off)", namespace, np_name)
        except k8s_client.ApiException as exc:
            if exc.status != 404:
                raise
        patch.status["networkPolicyName"]  = ""
        patch.status["lastReconcileTime"]  = _now_iso()
        patch.status["conditions"] = [_ready_condition("NetworkPolicySkipped", "egress.mode=off")]
        return

    # Apply NetworkPolicy (create or replace)
    try:
        existing = _networking_api().read_namespaced_network_policy(np_name, namespace)
        # Update
        np_manifest["metadata"]["resourceVersion"] = existing.metadata.resource_version
        _networking_api().replace_namespaced_network_policy(np_name, namespace, np_manifest)
        logger.info("Updated NetworkPolicy %s/%s (mode=%s)", namespace, np_name, egress_mode)
    except k8s_client.ApiException as exc:
        if exc.status == 404:
            _networking_api().create_namespaced_network_policy(namespace, np_manifest)
            logger.info("Created NetworkPolicy %s/%s (mode=%s)", namespace, np_name, egress_mode)
        else:
            raise

    patch.status["networkPolicyName"] = np_name
    patch.status["lastReconcileTime"] = _now_iso()
    patch.status["conditions"] = [
        _ready_condition(
            "PolicyApplied",
            f"NetworkPolicy {np_name} applied. "
            f"Sidecar injection {'enabled' if spec.get('inject', True) else 'disabled'}. "
            f"Egress mode: {egress_mode}.",
        )
    ]
    logger.info(
        "AgentSecurityPolicy %s/%s reconciled (inject=%s egress=%s)",
        namespace, name, spec.get("inject", True), egress_mode,
    )


# ── Delete ────────────────────────────────────────────────────────────────────

@kopf.on.delete("aegivis.io", "v1", "agentsecuritypolicies")
async def delete_policy(spec, name, namespace, **kwargs):
    logger.info("Deleting AgentSecurityPolicy %s/%s", namespace, name)

    np_name = network_policy_name(name)
    try:
        _networking_api().delete_namespaced_network_policy(np_name, namespace)
        logger.info("Deleted owned NetworkPolicy %s/%s", namespace, np_name)
    except k8s_client.ApiException as exc:
        if exc.status != 404:
            logger.warning(
                "Could not delete NetworkPolicy %s/%s: %s", namespace, np_name, exc
            )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ready_condition(reason: str, message: str) -> dict[str, Any]:
    return {
        "type":               "Ready",
        "status":             "True",
        "reason":             reason,
        "message":            message,
        "lastTransitionTime": _now_iso(),
    }
