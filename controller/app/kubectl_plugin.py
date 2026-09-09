"""kubectl-aegivis — CLI plugin for inspecting the Aegivis controller.

Install as a kubectl plugin:
  cp scripts/kubectl-aegivis /usr/local/bin/
  chmod +x /usr/local/bin/kubectl-aegivis

Then use as:
  kubectl aegivis status
  kubectl aegivis violations --namespace=myns
  kubectl aegivis audit --agent=my-agent
"""
from __future__ import annotations

import json
import sys
from typing import Any

import click
import httpx

from .config import settings


def _backend(path: str, params: dict | None = None) -> Any:
    """Call the Aegivis backend API."""
    url = f"{settings.backend_url.rstrip('/')}/v1/{path.lstrip('/')}"
    headers = {"X-API-Key": settings.backend_api_key}
    try:
        resp = httpx.get(url, headers=headers, params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        click.echo(f"Error {exc.response.status_code}: {exc.response.text}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Connection error: {exc}", err=True)
        sys.exit(1)


def _k8s_policies(namespace: str | None) -> list[dict]:
    """List AgentSecurityPolicy CRDs from K8s."""
    try:
        from kubernetes import client as k8s_client, config as k8s_config

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()

        custom_api = k8s_client.CustomObjectsApi()
        if namespace:
            result = custom_api.list_namespaced_custom_object(
                group="aegivis.io",
                version="v1",
                namespace=namespace,
                plural="agentsecuritypolicies",
            )
        else:
            result = custom_api.list_cluster_custom_object(
                group="aegivis.io",
                version="v1",
                plural="agentsecuritypolicies",
            )
        return result.get("items") or []
    except Exception as exc:
        click.echo(f"K8s error: {exc}", err=True)
        return []


# ── CLI root ──────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """kubectl-aegivis — Aegivis AI Security Controller plugin."""


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace (default: all)")
@click.option("--output", "-o", default="table", type=click.Choice(["table", "json"]))
def status(namespace: str | None, output: str):
    """Show all AgentSecurityPolicies and their status."""
    policies = _k8s_policies(namespace)

    if output == "json":
        click.echo(json.dumps(policies, indent=2, default=str))
        return

    if not policies:
        click.echo("No AgentSecurityPolicies found.")
        return

    # Table header
    header = f"{'NAMESPACE':<20} {'NAME':<30} {'INJECT':<8} {'EGRESS':<12} {'STATUS':<15} {'NP':<30}"
    click.echo(header)
    click.echo("-" * len(header))

    for p in policies:
        meta    = p.get("metadata") or {}
        spec    = p.get("spec") or {}
        status_ = p.get("status") or {}

        ns      = meta.get("namespace", "?")
        name    = meta.get("name", "?")
        inject  = "yes" if spec.get("inject", True) else "no"
        egress  = (spec.get("egress") or {}).get("mode", "audit")
        np      = status_.get("networkPolicyName", "-")

        conditions = status_.get("conditions") or []
        condition  = conditions[0].get("reason", "Unknown") if conditions else "Pending"

        click.echo(f"{ns:<20} {name:<30} {inject:<8} {egress:<12} {condition:<15} {np:<30}")


# ── violations ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace (informational only)")
@click.option("--agent", "-a", default=None, help="Filter by agent ID")
@click.option("--limit", "-l", default=20, help="Number of violations to show")
@click.option("--severity", "-s", default=None, type=click.Choice(["BLOCK", "ALERT"]))
@click.option("--output", "-o", default="table", type=click.Choice(["table", "json"]))
def violations(namespace: str | None, agent: str | None, limit: int, severity: str | None, output: str):
    """Show recent policy violations from the Aegivis backend."""
    params: dict[str, Any] = {"limit": limit}
    if agent:
        params["agent_id"] = agent
    if severity:
        params["action"] = severity

    data = _backend("violations", params)
    items = data.get("violations") or []

    if output == "json":
        click.echo(json.dumps(items, indent=2, default=str))
        return

    if not items:
        click.echo("No violations found.")
        return

    header = f"{'TIMESTAMP':<26} {'AGENT':<20} {'ACTION':<8} {'RULE':<35} {'SESSION':<20}"
    click.echo(header)
    click.echo("-" * len(header))

    for v in items:
        ts      = (v.get("timestamp") or "")[:25]
        agent_  = (v.get("agent_id") or "-")[:19]
        action  = v.get("action") or "-"
        rule    = (v.get("rule_id") or "-")[:34]
        session = (v.get("session_id") or "-")[:19]

        # Colour-code action
        if action == "BLOCK":
            action_str = click.style(f"{action:<8}", fg="red", bold=True)
        elif action == "ALERT":
            action_str = click.style(f"{action:<8}", fg="yellow")
        else:
            action_str = f"{action:<8}"

        click.echo(f"{ts:<26} {agent_:<20} {action_str} {rule:<35} {session:<20}")


# ── audit ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--agent", "-a", default=None, help="Filter report to a specific agent ID")
@click.option("--framework", "-f", default="soc2",
              type=click.Choice(["soc2", "owasp_asi_2026", "hipaa", "eu_ai_act", "gdpr"]),
              help="Compliance framework")
@click.option("--from-date", default="2020-01-01", help="Start date (YYYY-MM-DD)")
@click.option("--to-date",   default=None,         help="End date (YYYY-MM-DD, default: today)")
@click.option("--output", "-o", default="summary", type=click.Choice(["summary", "json"]))
def audit(agent: str | None, framework: str, from_date: str, to_date: str | None, output: str):
    """Generate a compliance audit report from the Aegivis backend."""
    from datetime import date

    params: dict[str, Any] = {
        "framework": framework,
        "from_date": from_date,
        "to_date":   to_date or date.today().isoformat(),
    }
    if agent:
        params["agent_id"] = agent

    data = _backend("export/audit-report", params)

    if output == "json":
        click.echo(json.dumps(data, indent=2, default=str))
        return

    # Summary output
    summary = data.get("summary") or {}
    click.echo(f"\n{'='*60}")
    click.echo(f"  Aegivis Compliance Audit — {framework.upper()}")
    click.echo(f"{'='*60}")
    click.echo(f"  Org:       {data.get('org_id', '?')}")
    click.echo(f"  Generated: {data.get('generated_at', '?')}")
    click.echo(f"  Period:    {data.get('from_date', '?')} → {data.get('to_date', '?')}")

    overall = summary.get("overall_status", "unknown").upper()
    colour  = "green" if overall == "PASS" else "red"
    click.echo(f"  Status:    {click.style(overall, fg=colour, bold=True)}")
    click.echo("")

    controls = summary.get("controls") or []
    if controls:
        click.echo(f"  {'CONTROL':<40} {'STATUS':<8} EVIDENCE")
        click.echo(f"  {'-'*80}")
        for ctrl in controls:
            cid   = f"[{ctrl.get('id', '?')}] {ctrl.get('name', '?')}"[:39]
            cstat = ctrl.get("status", "?").upper()
            cevid = (ctrl.get("evidence") or "")[:60]
            ccolour = "green" if cstat == "PASS" else ("yellow" if cstat == "PARTIAL" else "red")
            click.echo(f"  {cid:<40} {click.style(cstat, fg=ccolour):<8} {cevid}")

    click.echo("")


# ── inject ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("pod_name")
@click.option("--namespace", "-n", default="default")
@click.option("--policy",    "-p", default=None, help="AgentSecurityPolicy name (auto-detect if omitted)")
def inject(pod_name: str, namespace: str, policy: str | None):
    """Manually trigger sidecar injection check for an existing pod (dry-run info)."""
    click.echo(
        click.style("Note: ", fg="yellow", bold=True) +
        "Sidecar injection only applies at Pod CREATE time via MutatingAdmissionWebhook.\n"
        "Existing pods cannot be injected without restart."
    )
    click.echo(f"\nTo inject {pod_name!r}, delete and recreate the pod:")
    click.echo(f"  kubectl rollout restart deployment/<deployment-name> -n {namespace}")
    click.echo(
        f"\nEnsure an AgentSecurityPolicy with inject=true exists in namespace '{namespace}'."
    )
    if policy:
        click.echo(f"  Target policy: {policy}")


if __name__ == "__main__":
    cli()
