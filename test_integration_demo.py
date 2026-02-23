"""
AgentBlackBox Integration Demo
===============================
Run this script to verify AgentBlackBox is capturing your agent's activity.

This script:
  1. Makes several LLM calls through the proxy (simulating a tool-using agent)
  2. Queries the backend to show what was captured
  3. Verifies the hash chain integrity
  4. Checks for policy violations

Prerequisites:
  - AgentBlackBox proxy running at localhost:8080
  - AgentBlackBox backend running at localhost:8000  (or docker compose up -d)
  - OpenAI API key in OPENAI_API_KEY env var

Usage:
  pip install openai requests
  export OPENAI_API_KEY="sk-..."
  python test_integration_demo.py

No OpenAI key? Use Ollama (free, local):
  ollama pull llama3.2
  export OPENAI_BASE_URL="http://localhost:8080/ollama/v1"
  python test_integration_demo.py --model llama3.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("Run: pip install requests")
    sys.exit(1)

try:
    import openai
except ImportError:
    print("Run: pip install openai")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────

PROXY_URL    = os.environ.get("ABB_PROXY_URL",   "http://localhost:8080")
BACKEND_URL  = os.environ.get("ABB_BACKEND_URL", "http://localhost:8000")
BACKEND_KEY  = os.environ.get("ABB_API_KEY",     "dev-dashboard-key")
OPENAI_KEY   = os.environ.get("OPENAI_API_KEY",  "sk-placeholder")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(msg: str): print(f"  \033[32m[OK]\033[0m {msg}")
def _fail(msg: str): print(f"  \033[31m[FAIL]\033[0m {msg}")
def _info(msg: str): print(f"  \033[90m[..]\033[0m {msg}")
def _section(title: str): print(f"\n\033[1m{'=' * 55}\033[0m\n\033[1m {title}\033[0m\n{'=' * 55}")


def backend_get(path: str) -> dict | None:
    try:
        resp = requests.get(
            f"{BACKEND_URL}{path}",
            headers={"X-API-Key": BACKEND_KEY},
            timeout=10,
        )
        return resp.json() if resp.ok else None
    except Exception as e:
        return None


def backend_post(path: str, body: dict) -> dict | None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}{path}",
            json=body,
            headers={"X-API-Key": BACKEND_KEY},
            timeout=10,
        )
        return resp.json() if resp.ok else None
    except Exception as e:
        return None


# ── Step 1: Health checks ─────────────────────────────────────────────────────

def check_services():
    _section("Step 1: Checking AgentBlackBox services")

    # Proxy
    try:
        resp = requests.get(f"{PROXY_URL}/health", timeout=5)
        if resp.ok:
            data = resp.json()
            _ok(f"Proxy running — {data.get('policy_rules', 0)} policy rules active")
        else:
            _fail(f"Proxy returned {resp.status_code}")
            return False
    except Exception:
        _fail(f"Proxy not reachable at {PROXY_URL}")
        print("    Start it: cd proxy && uvicorn app.main:app --port 8080")
        return False

    # Backend
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.ok:
            _ok(f"Backend running at {BACKEND_URL}")
        else:
            _fail(f"Backend returned {resp.status_code}")
            return False
    except Exception:
        _fail(f"Backend not reachable at {BACKEND_URL}")
        print("    Start it: cd backend && uvicorn app.main:app --port 8000")
        return False

    return True


# ── Step 2: Make LLM calls through proxy ─────────────────────────────────────

def run_demo_agent(model: str) -> str | None:
    """
    Simulate a simple research agent with 2-3 LLM calls.
    Returns the session_id captured by the proxy.
    """
    _section("Step 2: Running demo agent through proxy")

    # Create OpenAI client pointed at our proxy
    client = openai.OpenAI(
        api_key=OPENAI_KEY,
        base_url=f"{PROXY_URL}/openai/v1",
        default_headers={
            "X-ABB-Agent-ID": "demo-research-agent",   # optional: label this agent
        },
    )

    # ── Call 1: Initial research question ────────────────────────────────────
    _info("LLM call 1/2: Initial question...")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Be concise."},
            {"role": "user",   "content": "What are the three most important safety concerns with autonomous AI agents? Answer in exactly 3 bullet points."},
        ]
        resp1 = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=200,
        )
        answer1 = resp1.choices[0].message.content
        _ok(f"Got response ({len(answer1)} chars)")
        print(f"\n    Agent said:\n    {answer1[:200].replace(chr(10), chr(10)+'    ')}\n")
    except openai.AuthenticationError:
        _fail("OpenAI API key invalid. Set OPENAI_API_KEY or use --model with Ollama.")
        return None
    except openai.APIConnectionError as e:
        _fail(f"Cannot connect to proxy: {e}")
        return None
    except Exception as e:
        _fail(f"LLM call failed: {e}")
        return None

    # ── Call 2: Follow-up ─────────────────────────────────────────────────────
    _info("LLM call 2/2: Follow-up question (same session)...")
    try:
        messages.append({"role": "assistant", "content": answer1})
        messages.append({"role": "user",      "content": "Which of those three is the most urgent right now in 2026? One sentence."})

        resp2 = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=100,
        )
        answer2 = resp2.choices[0].message.content
        _ok(f"Got response ({len(answer2)} chars)")
        print(f"\n    Agent said:\n    {answer2}\n")
    except Exception as e:
        _fail(f"LLM call 2 failed: {e}")

    # Give the proxy's async transport a moment to flush events to backend
    _info("Waiting 3s for proxy to flush events to backend...")
    time.sleep(3)

    return None  # session_id is discovered from backend in next step


# ── Step 3: Show what was captured ───────────────────────────────────────────

def show_captured_data():
    _section("Step 3: What AgentBlackBox captured")

    sessions_data = backend_get("/v1/sessions")
    if not sessions_data:
        _fail("Could not reach backend or no sessions found yet.")
        _info("The proxy sends events asynchronously — wait a moment and re-run.")
        return None

    sessions = sessions_data.get("sessions", [])
    if not sessions:
        _fail("No sessions in backend yet. Events may still be in transit.")
        _info("Try: curl http://localhost:8000/v1/sessions -H 'X-API-Key: dev-dashboard-key'")
        return None

    # Show the most recent session
    latest = sessions[0]
    session_id = latest.get("session_id") or latest.get("id")
    _ok(f"Found {len(sessions)} session(s) in backend")
    print()
    print(f"    Most recent session:")
    print(f"      session_id    : {session_id}")
    print(f"      agent_id      : {latest.get('agent_id', 'unknown')}")
    print(f"      event_count   : {latest.get('event_count', '?')}")
    print(f"      llm_calls     : {latest.get('llm_call_count', '?')}")
    print(f"      tool_calls    : {latest.get('tool_call_count', '?')}")
    print(f"      provider      : {latest.get('provider', '?')}")
    print(f"      model         : {latest.get('model', '?')}")

    return session_id


# ── Step 4: Show events ───────────────────────────────────────────────────────

def show_events(session_id: str):
    _section("Step 4: Event timeline")

    events_data = backend_get(f"/v1/sessions/{session_id}/events")
    if not events_data:
        _fail("Could not fetch events")
        return

    events = events_data.get("events", [])
    _ok(f"{len(events)} events in this session")
    print()

    event_type_colors = {
        "LLM_CALL_START": "\033[34m",   # blue
        "LLM_CALL_END":   "\033[36m",   # cyan
        "TOOL_CALL_START": "\033[33m",  # yellow
        "TOOL_CALL_END":  "\033[33m",   # yellow
        "AGENT_FINISH":   "\033[32m",   # green
        "SYSTEM_ERROR":   "\033[31m",   # red
        "CHECKPOINT":     "\033[90m",   # gray
    }
    reset = "\033[0m"

    for ev in events:
        et = ev.get("event_type", "?")
        color = event_type_colors.get(et, "")
        seq = ev.get("sequence_number", "?")
        ts = ev.get("timestamp_ns", 0)
        hash_short = (ev.get("current_hash") or "")[:12]

        # Show pii detected
        pii = ev.get("pii_detected", [])
        pii_str = f"  \033[31mPII:{','.join(pii)}\033[0m" if pii else ""

        print(f"    [{seq:>3}] {color}{et:<18}{reset}  hash:{hash_short}...{pii_str}")

        # For LLM_CALL_END show a snippet of the response
        if et == "LLM_CALL_END":
            payload = ev.get("payload", {})
            text = payload.get("response_text", "")
            if text:
                snippet = text[:80].replace("\n", " ")
                print(f"          response: \"{snippet}...\"")


# ── Step 5: Verify hash chain ─────────────────────────────────────────────────

def verify_chain(session_id: str):
    _section("Step 5: Hash chain verification")

    result = backend_get(f"/v1/sessions/{session_id}/verify")
    if not result:
        _fail("Could not verify chain")
        return

    if result.get("valid"):
        _ok(f"Chain VALID — {result.get('events_checked', '?')} events verified")
        print(f"    Every event's hash matches. No tampering detected.")
    else:
        _fail(f"Chain INVALID — tamper detected at event #{result.get('tampered_at_sequence', '?')}")

    print(f"    Verified at: {result.get('verified_at', '?')}")


# ── Step 6: Show policy violations ────────────────────────────────────────────

def show_violations():
    _section("Step 6: Policy violations")

    data = backend_get("/v1/violations/summary")
    if not data:
        _fail("Could not fetch violations")
        return

    summary = data.get("summary", [])
    if not summary:
        _ok("No policy violations — your agent behaved within policy limits")
    else:
        print(f"    Policy violations detected:")
        for item in summary:
            action = item.get("action", "?")
            rule   = item.get("rule_name", "?")
            count  = item.get("count", "?")
            color  = "\033[31m" if action == "BLOCK" else "\033[33m"
            print(f"      {color}[{action}]{reset}  {rule}  (x{count})")


# ── Step 7: Generate compliance report ───────────────────────────────────────

def show_compliance(session_id: str):
    _section("Step 7: Compliance report (EU AI Act)")

    result = backend_post("/v1/reports/generate", {
        "session_id": session_id,
        "regulation": "eu_ai_act",
    })

    if not result:
        _info("Compliance report endpoint returned no data (normal if session has no events yet)")
        return

    if "error" in result:
        _info(f"Report: {result['error']}")
        return

    _ok("EU AI Act compliance report generated")
    print(f"    regulation    : {result.get('regulation', '?')}")
    print(f"    compliant     : {result.get('compliant', '?')}")

    findings = result.get("findings", [])
    if findings:
        print(f"    findings ({len(findings)}):")
        for f in findings[:3]:
            print(f"      - {f.get('article','?')}: {f.get('status','?')}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AgentBlackBox integration demo")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use (default: gpt-4o-mini)")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls, just query existing data")
    args = parser.parse_args()

    print("\n\033[1mAgentBlackBox Integration Demo\033[0m")
    print("Testing that your AI agent's activity is being captured and governed.\n")

    # Step 1: Health
    if not check_services():
        print("\n\033[31mFix the above errors and re-run.\033[0m\n")
        sys.exit(1)

    # Step 2: Run agent
    if not args.skip_llm:
        run_demo_agent(args.model)

    # Step 3-7: Show results
    session_id = show_captured_data()

    if session_id:
        show_events(session_id)
        verify_chain(session_id)

    show_violations()

    if session_id:
        show_compliance(session_id)

    print(f"\n\033[1mDone.\033[0m Open http://localhost:5173 to see the full dashboard.\n")


if __name__ == "__main__":
    main()
