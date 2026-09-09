"""
Aegivis Demo Data Seeder
===============================
Directly posts realistic synthetic events and violations to the backend.
No LLM or API key required. Populates the dashboard for demo/review purposes.

Generates:
  - 5 agents (research-bot, data-analyst, code-assistant, email-processor, scheduler-agent)
  - ~29 sessions with LLM calls, tool calls, violations, and completions
  - Mix of normal activity, PII alerts, credential alerts, and injection violations
  - 7-day time spread so time-series charts show real trends, not a single spike

Usage:
    cd Aegivis
    pip install httpx
    python demo/seed.py

After running, open: http://localhost:5173

Reset all data:
    docker exec aegivis-postgres psql -U abb -d aegivis \\
      -c "TRUNCATE audit_events, violations CASCADE;"

Idempotent: re-running adds more sessions (new UUIDs each time).
"""
import hashlib
import json
import random
import sys
import time
import uuid

import httpx

BACKEND_URL = "http://localhost:8000"
API_KEY     = "dev-dashboard-key"
ORG_ID      = "default-org"

HEADERS = {
    "X-API-Key":    API_KEY,
    "Content-Type": "application/json",
}

# Realistic varied queries so tool args look different across sessions
_QUERIES = [
    "enterprise LLM security best practices 2026",
    "EU AI Act compliance requirements",
    "autonomous agent risk OWASP ASI",
    "prompt injection defense techniques",
    "NIST AI risk management framework",
    "RAG pipeline security vulnerabilities",
    "multi-agent system coordination patterns",
    "LLM output validation strategies",
    "zero-trust architecture for AI systems",
    "red team exercises for language models",
]


# ─── Hash chain helpers ───────────────────────────────────────────────────────

def _hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def genesis_hash(session_id: str) -> str:
    return _hash(f"genesis:{session_id}")


def event_hash(prev_hash: str, event_id: str, ts_ns: int, payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return _hash(f"{prev_hash}{event_id}{ts_ns}{payload_str}")


# ─── Event builder ────────────────────────────────────────────────────────────

def make_event(
    event_type: str,
    session_id: str,
    agent_id: str,
    sequence: int,
    prev_hash: str,
    ts_ns: int,
    payload: dict,
    provider: str = "anthropic",
    model: str = "claude-3-5-haiku-20241022",
    run_id: str | None = None,
    security: dict | None = None,
    blocked: bool = False,
) -> tuple[dict, str]:
    """Returns (event_dict, current_hash)."""
    eid = str(uuid.uuid4())
    rid = run_id or str(uuid.uuid4())
    curr = event_hash(prev_hash, eid, ts_ns, payload)

    ev: dict = {
        "event_id":        eid,
        "schema_version":  "1.0",
        "org_id":          ORG_ID,
        "session_id":      session_id,
        "agent_id":        agent_id,
        "provider":        provider,
        "model":           model,
        "run_id":          rid,
        "event_type":      event_type,
        "payload":         payload,
        "timestamp_ns":    ts_ns,
        "sequence_number": sequence,
        "previous_hash":   prev_hash,
        "current_hash":    curr,
        "blocked":         blocked,
    }
    if security:
        ev["security"] = security
    return ev, curr


# ─── Timestamp helper ─────────────────────────────────────────────────────────

def _ts(base_ns: int, offset_ms: int) -> int:
    return base_ns + offset_ms * 1_000_000


def _base_ns() -> int:
    """Random timestamp spread across the last 7 days (30 min–7 days ago)."""
    return (int(time.time()) - random.randint(30 * 60, 7 * 24 * 3600)) * 1_000_000_000


# ─── Session builders ─────────────────────────────────────────────────────────

def build_normal_session(
    agent_id: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-haiku-20241022",
) -> tuple[list[dict], list[dict]]:
    """A clean research session: 2 LLM calls with tool use, no violations."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    base_ns    = _base_ns()
    prev_hash  = genesis_hash(session_id)
    seq        = 0
    events     = []
    run_id     = str(uuid.uuid4())
    query      = random.choice(_QUERIES)

    tools_used = random.choice([
        ["web_search"],
        ["web_search", "write_report"],
        ["read_file", "web_search"],
    ])

    # LLM_CALL_START
    seq += 1
    ev, prev_hash = make_event(
        "LLM_CALL_START", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 0), {
            "model": model,
            "provider": provider,
            "messages_count": 1,
            "has_tools": True,
            "session_duration_ms": 0,
        }, provider=provider, model=model, run_id=run_id,
        security={"injection_score": round(random.uniform(0.0, 0.08), 3)},
    )
    events.append(ev)

    # TOOL_CALL_START + TOOL_CALL_END per tool
    for tool_name in tools_used:
        seq += 1
        tool_payload = {
            "tool_name": tool_name,
            "tool_args": {"query": query} if tool_name == "web_search"
                         else {"path": "reports/q4.md"},
            "run_id": run_id,
        }
        ev, prev_hash = make_event(
            "TOOL_CALL_START", session_id, agent_id, seq, prev_hash,
            _ts(base_ns, 250), tool_payload, provider=provider, model=model, run_id=run_id,
        )
        events.append(ev)

        seq += 1
        ev, prev_hash = make_event(
            "TOOL_CALL_END", session_id, agent_id, seq, prev_hash,
            _ts(base_ns, 800), {
                "tool_name": tool_name,
                "latency_ms": random.randint(120, 600),
                "run_id": run_id,
            }, provider=provider, model=model, run_id=run_id,
        )
        events.append(ev)

    # LLM_CALL_END
    seq += 1
    ev, prev_hash = make_event(
        "LLM_CALL_END", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 1500), {
            "model": model,
            "provider": provider,
            "latency_ms": random.randint(800, 3000),
            "token_usage": {
                "input_tokens": random.randint(200, 800),
                "output_tokens": random.randint(100, 400),
            },
            "finish_reason": "tool_use",
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    # Second LLM call (agent processes tool results)
    seq += 1
    run_id2 = str(uuid.uuid4())
    ev, prev_hash = make_event(
        "LLM_CALL_START", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 1600), {
            "model": model,
            "provider": provider,
            "messages_count": 3,
            "has_tools": True,
            "session_duration_ms": 1600,
        }, provider=provider, model=model, run_id=run_id2,
        security={"injection_score": round(random.uniform(0.0, 0.06), 3)},
    )
    events.append(ev)

    seq += 1
    ev, prev_hash = make_event(
        "LLM_CALL_END", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 4500), {
            "model": model,
            "provider": provider,
            "latency_ms": random.randint(1200, 4000),
            "token_usage": {
                "input_tokens": random.randint(400, 1200),
                "output_tokens": random.randint(200, 600),
            },
            "finish_reason": "end_turn",
        }, provider=provider, model=model, run_id=run_id2,
    )
    events.append(ev)

    # AGENT_FINISH
    seq += 1
    ev, _ = make_event(
        "AGENT_FINISH", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 4600), {
            "total_llm_calls": 2,
            "total_tool_calls": len(tools_used),
            "session_duration_ms": 4600,
            "finish_reason": "task_complete",
        }, provider=provider, model=model, run_id=run_id2,
    )
    events.append(ev)

    return events, []


def build_violation_session(
    agent_id: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-haiku-20241022",
) -> tuple[list[dict], list[dict]]:
    """A session containing an injection attempt that fires an ALERT."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    base_ns    = _base_ns()
    prev_hash  = genesis_hash(session_id)
    seq        = 0
    events     = []
    run_id     = str(uuid.uuid4())

    # LLM_CALL_START with elevated injection score (ALERT range 0.50–0.79)
    seq += 1
    inj_score = round(random.uniform(0.52, 0.75), 3)
    ev, prev_hash = make_event(
        "LLM_CALL_START", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 0), {
            "model": model,
            "provider": provider,
            "messages_count": 1,
            "has_tools": False,
            "session_duration_ms": 0,
        }, provider=provider, model=model, run_id=run_id,
        security={
            "injection_score": inj_score,
            "phrase_score": round(inj_score * 0.8, 3),
            "token_score": round(inj_score * 0.2, 3),
        },
    )
    events.append(ev)

    # LLM_CALL_END
    seq += 1
    ev, prev_hash = make_event(
        "LLM_CALL_END", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 2000), {
            "model": model,
            "provider": provider,
            "latency_ms": random.randint(800, 2500),
            "token_usage": {"input_tokens": 320, "output_tokens": 180},
            "finish_reason": "end_turn",
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    # AGENT_FINISH
    seq += 1
    ev, _ = make_event(
        "AGENT_FINISH", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 2100), {
            "total_llm_calls": 1,
            "total_tool_calls": 0,
            "session_duration_ms": 2100,
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    violation = {
        "rule_name": "prompt-injection-alert",
        "action": "ALERT",
        "reason": f"Injection score {inj_score:.2f} exceeds threshold (0.50)",
        "event_type": "LLM_CALL_START",
        "session_id": session_id,
        "agent_id": agent_id,
        "org_id": ORG_ID,
        "timestamp_ns": _ts(base_ns, 0),
    }
    return events, [violation]


def build_blocked_session(
    agent_id: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-haiku-20241022",
) -> tuple[list[dict], list[dict]]:
    """A session where the request was blocked (injection_score >= 0.80)."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    base_ns    = _base_ns()
    prev_hash  = genesis_hash(session_id)
    seq        = 0
    run_id     = str(uuid.uuid4())

    seq += 1
    inj_score = round(random.uniform(0.81, 0.97), 3)
    ev, _ = make_event(
        "LLM_CALL_START", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 0), {
            "model": model,
            "provider": provider,
            "messages_count": 1,
            "has_tools": False,
            "session_duration_ms": 0,
        }, provider=provider, model=model, run_id=run_id,
        security={
            "injection_score": inj_score,
            "phrase_score":    round(random.uniform(0.70, 0.90), 3),
            "token_score":     round(random.uniform(0.30, 0.60), 3),
        },
        blocked=True,
    )

    violation = {
        "rule_name": "prompt-injection-block",
        "action": "BLOCK",
        "reason": f"Injection score {inj_score:.2f} exceeds block threshold (0.80)",
        "event_type": "LLM_CALL_START",
        "session_id": session_id,
        "agent_id": agent_id,
        "org_id": ORG_ID,
        "timestamp_ns": _ts(base_ns, 0),
    }
    return [ev], [violation]


def build_pii_session(
    agent_id: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-haiku-20241022",
) -> tuple[list[dict], list[dict]]:
    """A session that triggers a PII alert (email/phone detected in LLM request)."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    base_ns    = _base_ns()
    prev_hash  = genesis_hash(session_id)
    seq        = 0
    events     = []
    run_id     = str(uuid.uuid4())

    seq += 1
    inj_score = round(random.uniform(0.05, 0.15), 3)
    ev, prev_hash = make_event(
        "LLM_CALL_START", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 0), {
            "model": model,
            "provider": provider,
            "messages_count": 1,
            "has_tools": False,
            "session_duration_ms": 0,
            "pii_detected": ["email_address", "phone_number"],
        }, provider=provider, model=model, run_id=run_id,
        security={"injection_score": inj_score},
    )
    events.append(ev)

    seq += 1
    ev, prev_hash = make_event(
        "LLM_CALL_END", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 1800), {
            "model": model,
            "provider": provider,
            "latency_ms": random.randint(600, 2000),
            "token_usage": {"input_tokens": 280, "output_tokens": 150},
            "finish_reason": "end_turn",
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    seq += 1
    ev, _ = make_event(
        "AGENT_FINISH", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 1900), {
            "total_llm_calls": 1,
            "total_tool_calls": 0,
            "session_duration_ms": 1900,
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    violation = {
        "rule_name": "pii-in-llm-request",
        "action": "ALERT",
        "reason": "PII detected in LLM request: email_address, phone_number",
        "event_type": "LLM_CALL_START",
        "session_id": session_id,
        "agent_id": agent_id,
        "org_id": ORG_ID,
        "timestamp_ns": _ts(base_ns, 0),
    }
    return events, [violation]


def build_credential_session(
    agent_id: str,
    provider: str = "anthropic",
    model: str = "claude-3-5-haiku-20241022",
) -> tuple[list[dict], list[dict]]:
    """A session that triggers a credential-leak alert (API key detected)."""
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    base_ns    = _base_ns()
    prev_hash  = genesis_hash(session_id)
    seq        = 0
    events     = []
    run_id     = str(uuid.uuid4())

    seq += 1
    inj_score   = round(random.uniform(0.08, 0.20), 3)
    token_score = round(random.uniform(0.60, 0.85), 3)
    ev, prev_hash = make_event(
        "LLM_CALL_START", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 0), {
            "model": model,
            "provider": provider,
            "messages_count": 1,
            "has_tools": False,
            "session_duration_ms": 0,
            "pii_detected": ["api_key"],
        }, provider=provider, model=model, run_id=run_id,
        security={
            "injection_score": inj_score,
            "token_score": token_score,
        },
    )
    events.append(ev)

    seq += 1
    ev, prev_hash = make_event(
        "LLM_CALL_END", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 1500), {
            "model": model,
            "provider": provider,
            "latency_ms": random.randint(500, 1800),
            "token_usage": {"input_tokens": 240, "output_tokens": 120},
            "finish_reason": "end_turn",
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    seq += 1
    ev, _ = make_event(
        "AGENT_FINISH", session_id, agent_id, seq, prev_hash,
        _ts(base_ns, 1600), {
            "total_llm_calls": 1,
            "total_tool_calls": 0,
            "session_duration_ms": 1600,
        }, provider=provider, model=model, run_id=run_id,
    )
    events.append(ev)

    violation = {
        "rule_name": "credential-leak-alert",
        "action": "ALERT",
        "reason": f"Credential detected in LLM request: api_key (token_score={token_score:.2f})",
        "event_type": "LLM_CALL_START",
        "session_id": session_id,
        "agent_id": agent_id,
        "org_id": ORG_ID,
        "timestamp_ns": _ts(base_ns, 0),
    }
    return events, [violation]


# ─── Network helpers ──────────────────────────────────────────────────────────

def post_batch(events: list[dict]) -> dict:
    batch = {
        "events":     events,
        "batch_id":   str(uuid.uuid4()),
        "sent_at_ns": time.time_ns(),
    }
    with httpx.Client(timeout=20) as client:
        r = client.post(f"{BACKEND_URL}/v1/events", headers=HEADERS, json=batch)
        r.raise_for_status()
        return r.json()


def post_violations(violations: list[dict]) -> dict:
    batch = {"violations": violations, "sent_at_ns": time.time_ns()}
    with httpx.Client(timeout=20) as client:
        r = client.post(f"{BACKEND_URL}/v1/violations", headers=HEADERS, json=batch)
        r.raise_for_status()
        return r.json()


def register_agent(agent_id: str, name: str, declared_purpose: str) -> None:
    with httpx.Client(timeout=10) as client:
        try:
            client.post(f"{BACKEND_URL}/v1/agents", headers=HEADERS, json={
                "agent_id": agent_id,
                "name": name,
                "declared_purpose": declared_purpose,
            })
        except Exception:
            pass  # ignore 409 if already exists


def check_backend() -> bool:
    try:
        r = httpx.get(f"{BACKEND_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ─── Agent definitions ────────────────────────────────────────────────────────

AGENTS = [
    ("research-bot",    "anthropic", "claude-3-5-haiku-20241022"),
    ("data-analyst",    "anthropic", "claude-3-5-sonnet-20241022"),
    ("code-assistant",  "anthropic", "claude-3-5-haiku-20241022"),
    ("email-processor", "openai",    "gpt-4o-mini"),
    ("scheduler-agent", "anthropic", "claude-3-5-haiku-20241022"),
]

# Session schedule per agent (29 sessions total, ~250 events, ~15 violations)
# Format: (agent_id, provider, model, [builder_fn, ...])
PER_AGENT_SCHEDULE = [
    # research-bot: 4 normal + 2 alert + 1 block + 1 pii = 8 sessions
    ("research-bot", "anthropic", "claude-3-5-haiku-20241022", [
        build_normal_session, build_normal_session,
        build_normal_session, build_normal_session,
        build_violation_session, build_violation_session,
        build_blocked_session,
        build_pii_session,
    ]),
    # data-analyst: 3 normal + 1 alert + 1 block + 1 cred = 6 sessions
    ("data-analyst", "anthropic", "claude-3-5-sonnet-20241022", [
        build_normal_session, build_normal_session, build_normal_session,
        build_violation_session,
        build_blocked_session,
        build_credential_session,
    ]),
    # code-assistant: 3 normal + 2 alert + 1 block = 6 sessions
    ("code-assistant", "anthropic", "claude-3-5-haiku-20241022", [
        build_normal_session, build_normal_session, build_normal_session,
        build_violation_session, build_violation_session,
        build_blocked_session,
    ]),
    # email-processor: 3 normal + 1 alert + 1 pii = 5 sessions
    ("email-processor", "openai", "gpt-4o-mini", [
        build_normal_session, build_normal_session, build_normal_session,
        build_violation_session,
        build_pii_session,
    ]),
    # scheduler-agent: 2 normal + 1 block + 1 cred = 4 sessions
    ("scheduler-agent", "anthropic", "claude-3-5-haiku-20241022", [
        build_normal_session, build_normal_session,
        build_blocked_session,
        build_credential_session,
    ]),
]


# ─── Main seeder ──────────────────────────────────────────────────────────────

def main():
    print("\nAegivis Demo Data Seeder")
    print("=" * 52)

    if not check_backend():
        print("ERROR: Backend not reachable at", BACKEND_URL)
        print("  Run: docker compose up -d")
        sys.exit(1)

    print(f"Backend : {BACKEND_URL}")
    print(f"Org     : {ORG_ID}")
    print()

    # 1. Register all agents
    print("Registering agents...")
    for agent_id, provider, model in AGENTS:
        register_agent(
            agent_id,
            name=agent_id.replace("-", " ").title(),
            declared_purpose=f"Demo {agent_id} using {model} via {provider}",
        )
        print(f"  {agent_id} ({provider}/{model})")
    print()

    # 2. Build and post all sessions, collecting violations along the way
    all_violations: list[dict] = []
    total_events   = 0
    total_sessions = 0

    for agent_id, provider, model, schedule in PER_AGENT_SCHEDULE:
        print(f"Agent: {agent_id}  ({model})")
        for builder_fn in schedule:
            events, viols = builder_fn(agent_id, provider, model)
            result = post_batch(events)
            acc = result.get("accepted", 0)
            total_events   += acc
            total_sessions += 1
            all_violations.extend(viols)

            if viols:
                label = "BLOCK" if any(v["action"] == "BLOCK" for v in viols) else "ALERT"
            else:
                label = "clean"
            print(f"  {builder_fn.__name__:32s} — {acc:3d} events  [{label}]")

        time.sleep(0.2)

    # 3. Post all violations in a single batch
    print()
    if all_violations:
        try:
            result = post_violations(all_violations)
            accepted_viols = result.get("accepted", len(all_violations))
            print(f"Violations: {accepted_viols}/{len(all_violations)} accepted")
        except Exception as e:
            print(f"Violations: Warning — could not post: {e}")
            print("  (Violations are also captured live by the proxy.)")
    else:
        print("No violations to post.")

    print()
    print("=" * 52)
    print(f"Seeded  : {total_sessions} sessions, ~{total_events} events, "
          f"{len(all_violations)} violations across {len(AGENTS)} agents")
    print()
    print("Open the dashboard:")
    print("  Overview  : http://localhost:5173/")
    print("  Sessions  : http://localhost:5173/sessions")
    print("  Security  : http://localhost:5173/security")
    print("  Metrics   : http://localhost:5173/metrics")
    print("  Topology  : http://localhost:5173/topology")
    print("  Agents    : http://localhost:5173/agents")
    print("  Policy    : http://localhost:5173/policy  (click Generate Suggestions)")
    print()
    print("Reset all data:")
    print('  docker exec aegivis-postgres psql -U abb -d aegivis \\')
    print('    -c "TRUNCATE audit_events, violations CASCADE;"')


if __name__ == "__main__":
    main()
