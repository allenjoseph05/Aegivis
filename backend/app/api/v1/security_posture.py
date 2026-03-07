"""
Security Posture API — Phase 12 Dashboard Support.

Provides the data endpoints that power the Security Posture dashboard page:
observed tool behavior, anomalies, network destinations, and live compliance.
Also provides the Security Package ZIP download.

GET /v1/security-posture/{agent_id}           — observed behavior summary
GET /v1/security-posture/{agent_id}/tools     — per-tool observations
GET /v1/security-posture/{agent_id}/anomalies — flagged anomaly events
GET /v1/security-posture/{agent_id}/network   — observed network destinations
GET /v1/manifests/{agent_id}/package          — download security package ZIP
"""
from __future__ import annotations

import io
import json
import logging
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.connection import get_session
from ...middleware.auth import OrgContext, require_api_key
from ...config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# GET /v1/security-posture/{agent_id}
# ---------------------------------------------------------------------------

@router.get("/security-posture/{agent_id}", summary="Security posture observed behavior summary")
async def get_posture_summary(
    agent_id: str,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return high-level observed behavior summary for the security posture page."""
    org_id = org_ctx.org_id

    result = await db.execute(text("""
        SELECT
            COUNT(DISTINCT session_id)                                      AS total_sessions,
            COUNT(*) FILTER (WHERE event_type = 'TOOL_CALL_START')         AS tool_calls,
            COUNT(DISTINCT event_data->>'tool_name')
                FILTER (WHERE event_type = 'TOOL_CALL_START')              AS unique_tools,
            COUNT(DISTINCT session_id)
                FILTER (WHERE event_type = 'TOOL_CALL_START'
                    AND event_data->>'tool_name' IS NOT NULL
                    AND (security->>'ssrf_detected')::boolean = true)       AS sessions_with_network_calls
        FROM audit_events
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND timestamp > NOW() - INTERVAL '1 day' * :days
    """), {"org_id": org_id, "agent_id": agent_id, "days": days})
    row = result.fetchone()

    # Anomaly count (from violations table)
    anom_result = await db.execute(text("""
        SELECT COUNT(*) AS anomaly_count
        FROM policy_violations
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND created_at > NOW() - INTERVAL '1 day' * :days
    """), {"org_id": org_id, "agent_id": agent_id, "days": days})
    anom_row = anom_result.fetchone()

    # Check if active manifest exists
    manifest_result = await db.execute(text("""
        SELECT manifest_id, issued_at, expires_at
        FROM capability_manifests
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND revoked_at IS NULL
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY issued_at DESC
        LIMIT 1
    """), {"org_id": org_id, "agent_id": agent_id})
    manifest_row = manifest_result.fetchone()

    return {
        "agent_id": agent_id,
        "observation_days": days,
        "total_sessions": row[0] if row else 0,
        "tool_calls": row[1] if row else 0,
        "unique_tools": row[2] if row else 0,
        "sessions_with_network_calls": row[3] if row else 0,
        "anomaly_count": anom_row[0] if anom_row else 0,
        "has_active_manifest": manifest_row is not None,
        "active_manifest_id": manifest_row[0] if manifest_row else None,
        "active_manifest_issued": manifest_row[1].isoformat() if manifest_row and manifest_row[1] else None,
        "active_manifest_expires": manifest_row[2].isoformat() if manifest_row and manifest_row[2] else None,
    }


# ---------------------------------------------------------------------------
# GET /v1/security-posture/{agent_id}/tools
# ---------------------------------------------------------------------------

@router.get("/security-posture/{agent_id}/tools", summary="Observed tool behavior")
async def get_observed_tools(
    agent_id: str,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return per-tool observation statistics for review before manifest generation."""
    result = await db.execute(text("""
        SELECT
            ae.event_data->>'tool_name'       AS tool_name,
            COUNT(*)                           AS call_count,
            COUNT(DISTINCT session_id)         AS session_count,
            MIN(timestamp)                     AS first_seen,
            MAX(timestamp)                     AS last_seen,
            jsonb_agg(DISTINCT jsonb_object_keys(ae.event_data->'tool_arguments'))
                FILTER (WHERE ae.event_data->'tool_arguments' IS NOT NULL)
                                               AS observed_arg_keys
        FROM audit_events ae
        WHERE ae.org_id = :org_id
          AND ae.agent_id = :agent_id
          AND ae.event_type = 'TOOL_CALL_START'
          AND ae.timestamp > NOW() - INTERVAL '1 day' * :days
          AND ae.event_data->>'tool_name' IS NOT NULL
        GROUP BY ae.event_data->>'tool_name'
        ORDER BY call_count DESC
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id, "days": days})
    rows = result.mappings().fetchall()

    tools = []
    for r in rows:
        tools.append({
            "tool_name": r["tool_name"],
            "call_count": r["call_count"],
            "session_count": r["session_count"],
            "first_seen": r["first_seen"].isoformat() if r["first_seen"] else None,
            "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None,
            "observed_arg_keys": r["observed_arg_keys"] or [],
        })
    return {"tools": tools, "count": len(tools)}


# ---------------------------------------------------------------------------
# GET /v1/security-posture/{agent_id}/anomalies
# ---------------------------------------------------------------------------

@router.get("/security-posture/{agent_id}/anomalies", summary="Flagged anomaly events")
async def get_anomalies(
    agent_id: str,
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return recent policy violations and anomalies for review."""
    result = await db.execute(text("""
        SELECT
            id, rule_name, action, reason, session_id, created_at,
            event_data
        FROM policy_violations
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND created_at > NOW() - INTERVAL '1 day' * :days
        ORDER BY created_at DESC
        LIMIT :limit
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id, "days": days, "limit": limit})
    rows = result.mappings().fetchall()

    anomalies = []
    for r in rows:
        row = dict(r)
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                row[k] = v.isoformat()
        anomalies.append(row)
    return {"anomalies": anomalies, "count": len(anomalies)}


# ---------------------------------------------------------------------------
# GET /v1/security-posture/{agent_id}/network
# ---------------------------------------------------------------------------

@router.get("/security-posture/{agent_id}/network", summary="Observed network destinations")
async def get_observed_network(
    agent_id: str,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """Return observed outbound network destinations for egress allowlist review."""
    result = await db.execute(text("""
        SELECT
            security->>'ssrf_destination'  AS destination,
            COUNT(*)                         AS seen_count,
            MAX(timestamp)                   AS last_seen
        FROM audit_events
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND timestamp > NOW() - INTERVAL '1 day' * :days
          AND security->>'ssrf_destination' IS NOT NULL
        GROUP BY security->>'ssrf_destination'
        ORDER BY seen_count DESC
    """), {"org_id": org_ctx.org_id, "agent_id": agent_id, "days": days})
    rows = result.mappings().fetchall()

    dests = []
    for r in rows:
        dests.append({
            "destination": r["destination"],
            "seen_count": r["seen_count"],
            "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None,
        })
    return {"destinations": dests, "count": len(dests)}


# ---------------------------------------------------------------------------
# GET /v1/manifests/{agent_id}/package
# ---------------------------------------------------------------------------

@router.get("/manifests/{agent_id}/package", summary="Download security package ZIP")
async def download_security_package(
    agent_id: str,
    db: AsyncSession = Depends(get_session),
    org_ctx: OrgContext = Depends(require_api_key),
):
    """
    Generate and return a ZIP containing the security package:
      - docker-compose.hardened.yml
      - aegivis-seccomp.json
      - aegivis-network-policy.conf
      - manifest.json
      - SETUP.md
    """
    org_id = org_ctx.org_id

    # Fetch active manifest
    manifest_row = await db.execute(text("""
        SELECT manifest_id, manifest_json, permitted_network_destinations
        FROM capability_manifests
        WHERE org_id = :org_id
          AND agent_id = :agent_id
          AND revoked_at IS NULL
          AND (expires_at IS NULL OR expires_at > NOW())
        ORDER BY issued_at DESC
        LIMIT 1
    """), {"org_id": org_id, "agent_id": agent_id})
    manifest = manifest_row.mappings().fetchone()

    manifest_id = manifest["manifest_id"] if manifest else "no-manifest"
    manifest_json_str = manifest["manifest_json"] if manifest else "{}"
    network_dests = manifest["permitted_network_destinations"] if manifest else []
    if isinstance(network_dests, str):
        network_dests = json.loads(network_dests)

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("docker-compose.hardened.yml", _gen_docker_compose(agent_id, manifest_id))
        zf.writestr("aegivis-seccomp.json", _gen_seccomp())
        zf.writestr("aegivis-network-policy.conf", _gen_network_policy(agent_id, manifest_id, network_dests))
        zf.writestr("manifest.json", manifest_json_str)
        zf.writestr("SETUP.md", _gen_setup_md(agent_id, manifest_id, network_dests))
    buf.seek(0)

    filename = f"aegivis-security-package-{agent_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Security package file generators
# ---------------------------------------------------------------------------

def _gen_docker_compose(agent_id: str, manifest_id: str) -> str:
    return f"""# Aegivis Security Package — {agent_id}
# Generated: {datetime.now(timezone.utc).isoformat()}
# Manifest: {manifest_id}
#
# Instructions: Replace your existing docker-compose.yml with this file,
# or merge the security_opt, cap_drop, read_only, and networks sections
# into your existing service definition.

version: "3.9"

services:
  agent:
    image: your-agent-image:latest    # ← replace with your agent image
    security_opt:
      - seccomp:./aegivis-seccomp.json
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:size=100m,noexec
    volumes:
      - ./workspace:/workspace:rw    # only writable path
    environment:
      - AEGIVIS_MANIFEST_ID={manifest_id}
      - AEGIVIS_PROXY_URL=http://aegivis-proxy:8080
      # ← add your agent's existing env vars below this line
    networks:
      - aegivis-restricted
    depends_on:
      - aegivis-proxy

  aegivis-proxy:
    image: aegivis/proxy:latest
    ports:
      - "8080:8080"
    environment:
      - AEGIVIS_BACKEND_URL=http://aegivis-backend:8000
      - AEGIVIS_MANIFEST_ENFORCE=true
      - AEGIVIS_MANIFEST_SIGNING_KEY=${{AEGIVIS_MANIFEST_SIGNING_KEY}}
      - AEGIVIS_SECURITY_EGRESS_MODE=allowlist
      - AEGIVIS_SECURITY_EGRESS_ALLOWLIST=${{AEGIVIS_SECURITY_EGRESS_ALLOWLIST}}
      - AEGIVIS_SECURITY_IFC_ENABLED=true
      - AEGIVIS_SECURITY_IFC_BLOCK=true
    networks:
      - aegivis-restricted
      - aegivis-external

  aegivis-backend:
    image: aegivis/backend:latest
    environment:
      - DATABASE_URL=${{DATABASE_URL}}
      - AEGIVIS_MANIFEST_SIGNING_KEY=${{AEGIVIS_MANIFEST_SIGNING_KEY}}
    networks:
      - aegivis-internal

networks:
  aegivis-restricted:
    driver: bridge
  aegivis-external:
    driver: bridge
  aegivis-internal:
    driver: bridge
    internal: true
"""


def _gen_seccomp() -> str:
    profile = {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
        "syscalls": [{
            "names": [
                "read", "write", "open", "openat", "close", "stat", "fstat",
                "lstat", "poll", "lseek", "mmap", "mprotect", "munmap", "brk",
                "rt_sigaction", "rt_sigprocmask", "ioctl", "pread64", "pwrite64",
                "readv", "writev", "access", "pipe", "select", "sched_yield",
                "mremap", "msync", "mincore", "madvise", "dup", "dup2",
                "nanosleep", "getpid", "sendfile", "socket", "connect",
                "accept", "sendto", "recvfrom", "sendmsg", "recvmsg", "shutdown",
                "bind", "listen", "getsockname", "getpeername", "socketpair",
                "setsockopt", "getsockopt", "clone", "fork", "execve",
                "exit", "wait4", "kill", "uname", "fcntl", "flock", "fsync",
                "fdatasync", "truncate", "ftruncate", "getdents", "getcwd",
                "chdir", "rename", "mkdir", "rmdir", "link", "unlink",
                "symlink", "readlink", "chmod", "chown", "umask", "gettimeofday",
                "getrlimit", "getrusage", "sysinfo", "times",
                "getuid", "getgid", "geteuid", "getegid",
                "getgroups", "futex", "sched_getaffinity",
                "arch_prctl", "set_tid_address",
                "clock_gettime", "clock_getres", "clock_nanosleep",
                "exit_group", "epoll_wait", "epoll_ctl", "epoll_create",
                "epoll_create1", "getdents64", "set_robust_list",
                "get_robust_list", "splice", "tee", "ppoll", "pselect6",
                "openat2", "statx", "getrandom",
            ],
            "action": "SCMP_ACT_ALLOW",
        }],
    }
    return json.dumps(profile, indent=2)


def _gen_network_policy(agent_id: str, manifest_id: str, destinations: list) -> str:
    dest_lines = "\n".join(f"ALLOW {d}" for d in destinations) if destinations else "# No specific destinations — add manually"
    return f"""# Aegivis Network Egress Policy — {agent_id}
# Generated: {datetime.now(timezone.utc).isoformat()} | Manifest: {manifest_id}
#
# Configure your network proxy or firewall with these rules.
# This file is for documentation — apply rules to your infrastructure.
#
# Permitted outbound destinations:
{dest_lines}
#
# All other destinations: DENY
DEFAULT DENY

# To add more destinations:
# 1. Update the manifest via the dashboard (Generate Security Package)
# 2. Or add entries to AEGIVIS_SECURITY_EGRESS_ALLOWLIST env var on the proxy
"""


def _gen_setup_md(agent_id: str, manifest_id: str, destinations: list) -> str:
    dest_list = ", ".join(destinations[:5]) + ("..." if len(destinations) > 5 else "")
    return f"""# Aegivis Security Package Setup Guide
## Agent: {agent_id} | Manifest: {manifest_id}
Generated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

---

## What Is This?

This package configures your agent deployment with Aegivis's containment
architecture. Once applied, your agent can be fully "injected" by an attacker
and still cannot cause damage — because every action is checked against a
cryptographic manifest and OS-level rules it cannot override.

## Files in This Package

| File | Purpose |
|------|---------|
| `docker-compose.hardened.yml` | Drop-in Docker Compose with all security settings |
| `aegivis-seccomp.json` | OS-level syscall allowlist for the agent process |
| `aegivis-network-policy.conf` | Network egress allowlist documentation |
| `manifest.json` | Signed capability manifest (do not edit) |
| `SETUP.md` | This guide |

## Quick Setup (3 steps)

### Step 1: Set environment variables

```bash
# Generate a signing key (same value must be set on proxy AND backend)
export AEGIVIS_MANIFEST_SIGNING_KEY=$(openssl rand -hex 32)

# Set permitted network destinations (from your manifest)
export AEGIVIS_SECURITY_EGRESS_ALLOWLIST="{dest_list}"

# Database URL for the backend
export DATABASE_URL="postgresql://aegivis:password@postgres:5432/aegivis"
```

### Step 2: Replace your docker-compose.yml

```bash
cp docker-compose.hardened.yml docker-compose.yml
# Edit the 'agent' service to use your actual image name and env vars
```

### Step 3: Start the stack

```bash
docker compose up -d
docker exec aegivis-backend alembic upgrade head
```

## Verifying Protection

Once running, open the Aegivis dashboard at http://localhost:5173.

Navigate to **Security Posture** → your agent. You should see:
- Status: Protected (manifest active)
- IFC enforcement: enabled
- Egress enforcement: allowlist mode

Test with a simulated injection (send an email to an unknown address) —
the dashboard should show a BLOCK event under "Blocked Calls".

## Maintenance

The manifest expires in 30 days. Renew it by clicking **Renew** in the
Security Posture page, or run:

```bash
curl -X POST http://localhost:8000/v1/manifests/{agent_id}/generate \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{{"validity_days": 30}}'
```

## Support

- Documentation: https://docs.aegivis.io
- Issues: https://github.com/aegivis/aegivis/issues
"""
