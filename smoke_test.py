"""
AgentBlackBox smoke test — no real LLM key needed.

Sends fake OpenAI-format requests through the proxy to generate a full
audit trail: LLM calls, tool calls, annotations, and a session finish.

Run:
    python smoke_test.py

Watch the dashboard at http://localhost:5173 while this runs.
"""
import json
import time
import uuid
import httpx

PROXY_URL  = "http://localhost:8080/openai/v1"  # correct path: /openai/v1/chat/completions
BACKEND_URL = "http://localhost:8000"
API_KEY    = "dev-proxy-key"
SESSION_ID = f"smoke_{uuid.uuid4().hex[:10]}"
AGENT_ID   = "smoke-test-agent"

print(f"\nSmoke test session: {SESSION_ID}")
print(f"Dashboard: http://localhost:5173\n")

headers_proxy = {
    "Content-Type": "application/json",
    "Authorization": "Bearer fake-key-not-used",   # proxy doesn't validate this
    "X-ABB-Session-Id": SESSION_ID,
    "X-ABB-Agent-Id": AGENT_ID,
}
headers_backend = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
}


def send_event(event: dict):
    """Push a single event directly to the backend (simulates what the proxy does)."""
    batch = {
        "events": [event],
        "batch_id": str(uuid.uuid4()),
        "sent_at_ns": time.time_ns(),
    }
    r = httpx.post(f"{BACKEND_URL}/v1/events", json=batch, headers=headers_backend, timeout=10)
    r.raise_for_status()
    return r.json()


def make_event(event_type: str, payload: dict, seq: int, prev_hash: str) -> dict:
    import hashlib
    ev = {
        "event_id":       f"evt_{uuid.uuid4().hex[:20]}",
        "schema_version": "1.0",
        "org_id":         "default-org",
        "session_id":     SESSION_ID,
        "agent_id":       AGENT_ID,
        "provider":       "openai",
        "model":          "gpt-4o",
        "interception_layer": "proxy",
        "run_id":         str(uuid.uuid4()),
        "parent_run_id":  None,
        "event_type":     event_type,
        "payload":        payload,
        "payload_hash":   None,
        "pii_detected":   [],
        "timestamp_ns":   time.time_ns(),
        "sequence_number": seq,
        "previous_hash":  prev_hash,
    }
    ev["current_hash"] = hashlib.sha256(
        json.dumps(ev, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return ev


# ── Build a synthetic 8-event session ────────────────────────────────────────

prev_hash = f"GENESIS_{SESSION_ID}"
seq = 0
events_sent = 0


def push(event_type, payload):
    global prev_hash, seq, events_sent
    seq += 1
    ev = make_event(event_type, payload, seq, prev_hash)
    prev_hash = ev["current_hash"]
    result = send_event(ev)
    events_sent += 1
    print(f"  [{seq}] {event_type:20s} -> accepted={result['accepted']}")
    time.sleep(0.1)


print("Sending events...")

push("LLM_CALL_START", {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Search the web for AgentBlackBox"},
    ],
    "tools_available": [{"name": "web_search", "description": "Search the web"}],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": False,
})

push("LLM_CALL_END", {
    "response_text": None,
    "finish_reason": "tool_calls",
    "tool_calls": [{"id": "tc_1", "name": "web_search", "arguments": '{"query": "AgentBlackBox"}'}],
    "token_usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    "latency_ms": 842,
    "http_status": 200,
})

push("TOOL_CALL_START", {
    "tool_name": "web_search",
    "tool_description": "Search the web",
    "tool_input": {"query": "AgentBlackBox"},
    "tool_call_id": "tc_1",
})

push("TOOL_CALL_END", {
    "tool_name": "web_search",
    "tool_call_id": "tc_1",
    "tool_output_masked": "[search results redacted]",
    "tool_output_hash": "abc123",
    "pii_fields_detected": [],
    "success": True,
})

push("LLM_CALL_START", {
    "messages": [
        {"role": "system",  "content": "You are a helpful assistant."},
        {"role": "user",    "content": "Search the web for AgentBlackBox"},
        {"role": "tool",    "content": "Found: agentblackbox.io — forensic audit for AI agents"},
    ],
    "tools_available": [],
    "temperature": 0.7,
    "max_tokens": 1024,
    "stream": False,
})

push("LLM_CALL_END", {
    "response_text": "AgentBlackBox is a forensic audit platform for AI agents.",
    "finish_reason": "stop",
    "tool_calls": [],
    "token_usage": {"prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
    "latency_ms": 1240,
    "http_status": 200,
})

push("AGENT_THOUGHT", {
    "message": "Task completed successfully",
    "level": "info",
    "metadata": {"source": "smoke_test"},
})

push("AGENT_FINISH", {
    "final_output": "AgentBlackBox is a forensic audit platform for AI agents.",
    "total_llm_calls": 2,
    "total_tool_calls": 1,
    "session_duration_ms": 2500,
})

# ── Verify ────────────────────────────────────────────────────────────────────

print(f"\nSent {events_sent} events. Verifying hash chain...")
r = httpx.get(
    f"{BACKEND_URL}/v1/sessions/{SESSION_ID}/verify",
    params={"org_id": "default-org"},
    headers=headers_backend,
    timeout=10,
)
result = r.json()
valid = result.get("valid", False)
print(f"  Chain valid: {valid}  (total events: {result.get('total_events')})")

print(f"\nSession in dashboard: http://localhost:5173")
print(f"Direct API:           {BACKEND_URL}/v1/sessions/{SESSION_ID}/events")
print(f"\nDone! Check the dashboard to see your audit trail.")
