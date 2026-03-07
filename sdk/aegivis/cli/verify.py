"""
CLI command: aegivis verify --session <session_id>

Queries the backend to verify hash chain integrity for a session.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

BACKEND_URL = os.environ.get("AEGIVIS_BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("AEGIVIS_BACKEND_API_KEY", "dev-dashboard-key")


def verify(session_id: str, org_id: str = "default-org") -> int:
    """
    Verify a session's hash chain. Returns 0 if valid, 1 if tampered, 2 if error.
    """
    url = f"{BACKEND_URL}/v1/sessions/{session_id}/verify"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                url,
                params={"org_id": org_id},
                headers={"X-API-Key": API_KEY},
            )
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to backend at {BACKEND_URL}", file=sys.stderr)
        print("Is the backend running? Try: docker compose up -d backend", file=sys.stderr)
        return 2

    if resp.status_code == 404:
        print(f"ERROR: Session '{session_id}' not found", file=sys.stderr)
        return 2

    if resp.status_code != 200:
        print(f"ERROR: Backend returned HTTP {resp.status_code}", file=sys.stderr)
        return 2

    result = resp.json()

    if result["valid"]:
        print(f"✓ CHAIN INTACT — {result['total_events']} events verified")
        print(f"  Session: {session_id}")
        print(f"  Checked at: {result['checked_at']}")
        return 0
    else:
        print(f"✗ CHAIN INTEGRITY VIOLATION")
        print(f"  Session: {session_id}")
        print(f"  First tampered event at sequence #{result.get('first_failed_sequence')}")
        print(f"  Error: {result.get('error_message')}")
        print(f"  Checked at: {result['checked_at']}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Aegivis — verify audit trail integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aegivis verify --session sess_abc123def456
  aegivis verify --session sess_abc123def456 --org my-company
  AEGIVIS_BACKEND_URL=https://abb.mycompany.com aegivis verify --session sess_xxx
        """,
    )
    parser.add_argument("--session", required=True, help="Session ID to verify")
    parser.add_argument("--org", default="default-org", help="Organization ID (default: default-org)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.json:
        url = f"{BACKEND_URL}/v1/sessions/{args.session}/verify"
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, params={"org_id": args.org}, headers={"X-API-Key": API_KEY})
        print(json.dumps(resp.json(), indent=2))
        return 0

    return verify(args.session, args.org)


if __name__ == "__main__":
    sys.exit(main())
