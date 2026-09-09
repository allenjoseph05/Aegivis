"""
Aegivis Demo Agent (Ollama)
==================================
An agentic loop using Ollama (local LLM, no API key needed), routed through
the Aegivis proxy. Demonstrates full session tracking, tool call
interception, and security scanning in the dashboard.

Requirements:
    pip install openai httpx
    Ollama running: https://ollama.com  (ollama serve)
    Model pulled:   ollama pull llama3.2

Usage:
    cd Aegivis
    python demo/agent.py

What you'll see in the dashboard (http://localhost:5173):
  - Sessions page: new sessions with timeline of LLM + tool call events
  - Security page: any violations (PII, injections) surfaced as alerts
  - Metrics page: updated call counts, model usage
  - Policy Builder: observed tools populated after a few runs
"""
import json
import sys
import time

try:
    from openai import OpenAI, APIStatusError
except ImportError:
    print("ERROR: openai package not installed.")
    print("  Run: pip install openai")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("ERROR: httpx package not installed.")
    print("  Run: pip install httpx")
    sys.exit(1)

# ─── Proxy + model configuration ─────────────────────────────────────────────

PROXY_BASE_URL = "http://localhost:8080/ollama/v1"
OLLAMA_URL     = "http://localhost:11434"
MODEL          = "llama3.2:latest"
MAX_TURNS      = 8

# ─── Tool definitions (OpenAI / Ollama format) ────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information on a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file by path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body":    {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_report",
            "description": "Write a research report to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Output filename"},
                    "content":  {"type": "string", "description": "Report content"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]


# ─── Simulated tool executor ──────────────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    if name == "web_search":
        q = args.get("query", "")
        return (
            f"Search results for '{q}':\n"
            "1. Recent study shows 40% improvement in model alignment techniques (2025)\n"
            "2. New benchmark evaluates AI safety properties across 12 dimensions\n"
            "3. Open-source toolkit released for adversarial testing of LLM agents\n"
            "4. Industry report: 78% of enterprise AI deployments lack runtime monitoring\n"
        )
    elif name == "read_file":
        path = args.get("path", "")
        return (
            f"Contents of {path}:\n"
            "Q4 2024 AI Safety Report\n"
            "Key findings:\n"
            "- 23% of agent deployments had at least one security incident\n"
            "- Prompt injection remains the #1 attack vector (61% of incidents)\n"
            "- Average detection time: 4.2 hours without runtime monitoring\n"
            "- Organizations with runtime safety layers: 0 incidents in the period\n"
        )
    elif name == "send_email":
        to      = args.get("to", "")
        subject = args.get("subject", "")
        return f"Email sent successfully to {to} | Subject: '{subject}'"
    elif name == "write_report":
        filename = args.get("filename", "report.md")
        length   = len(args.get("content", ""))
        return f"Report written to {filename} ({length} chars)"
    return f"Tool '{name}' executed: {json.dumps(args)[:100]}"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(MODEL.split(":")[0] in m for m in models):
            print(f"WARNING: Model '{MODEL}' not found in Ollama.")
            print(f"  Available: {', '.join(models) or 'none'}")
            print(f"  Pull it:   ollama pull {MODEL}")
            return False
        return True
    except Exception:
        return False


# ─── Agentic loop ─────────────────────────────────────────────────────────────

def run_task(
    task: str,
    agent_id: str,
    session_id: str | None = None,
    system: str | None = None,
) -> str:
    """Run a single agentic task through the proxy. Returns final text response."""
    client = OpenAI(
        api_key="ollama",           # Ollama ignores the key
        base_url=PROXY_BASE_URL,
    )

    extra_headers = {"x-aegivis-agent-id": agent_id}
    if session_id:
        extra_headers["x-aegivis-session-id"] = session_id

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": task})

    print(f"\n{'-'*60}")
    print(f"Agent : {agent_id}")
    print(f"Task  : {task[:100]}")
    print(f"{'-'*60}")

    final_text = ""

    for turn in range(MAX_TURNS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                extra_headers=extra_headers,
            )
        except APIStatusError as e:
            if e.status_code in (400, 403):
                print(f"\n[BLOCKED by proxy] Turn {turn+1}")
                try:
                    body = json.loads(e.response.text)
                    print(f"  Rule  : {body.get('rule', '?')}")
                    print(f"  Reason: {body.get('reason', str(e))[:150]}")
                except Exception:
                    print(f"  Detail: {str(e)[:200]}")
                return "[BLOCKED]"
            raise

        choice = response.choices[0]
        msg    = choice.message

        # Accumulate text
        if msg.content:
            final_text = msg.content

        # No tool calls — we're done
        if not msg.tool_calls or choice.finish_reason == "stop":
            print(f"\nTurn {turn+1} — done")
            if final_text:
                print(f"Response: {final_text[:300]}")
            break

        # Execute tool calls
        print(f"\nTurn {turn+1} — {len(msg.tool_calls)} tool call(s):")
        messages.append(msg.model_dump(exclude_unset=True))

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            print(f"  -> {tc.function.name}({json.dumps(args)[:80]})")
            result = execute_tool(tc.function.name, args)
            print(f"     {result[:100]}")
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    return final_text


# ─── Demo scenarios ───────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "label":    "Research agent — normal operation",
        "agent_id": "research-bot",
        "system":   "You are a research assistant. Use the web_search tool to find information, then write a brief summary. Be concise.",
        "task":     "Search for recent AI safety research and summarize the key findings.",
    },
    {
        "label":    "Data analyst — file processing",
        "agent_id": "data-analyst",
        "system":   "You are a data analyst. Use the read_file tool, then report the key statistics you find. Be concise.",
        "task":     "Read the Q4 AI safety report at path 'reports/q4_safety.md' and list the key statistics.",
    },
    {
        "label":    "Multi-step — research + email",
        "agent_id": "research-bot",
        "system":   "You are a research assistant. Search for information and then send a brief email summary. Be concise.",
        "task":     "Search for AI agent security risks, then email a one-paragraph summary to security-team@example.com.",
    },
    {
        "label":    "Code assistant — no tools needed",
        "agent_id": "code-assistant",
        "system":   "You are a coding assistant. Answer technical questions directly and concisely.",
        "task":     "What are the top 3 best practices for securing API keys in Python applications?",
    },
]


def main():
    print("\nAegivis Demo Agent (Ollama)")
    print("=" * 60)
    print(f"Proxy  : {PROXY_BASE_URL}")
    print(f"Model  : {MODEL}")
    print(f"Dashboard: http://localhost:5173")
    print("=" * 60)

    if not check_ollama():
        print("\nERROR: Ollama is not running or model not available.")
        print("  Start Ollama: ollama serve")
        print(f"  Pull model:   ollama pull {MODEL}")
        sys.exit(1)

    print()
    print("Running 4 scenarios. Watch the dashboard for live events.")
    print("Sessions appear under: Dashboard -> Sessions")
    print()

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n[{i}/{len(SCENARIOS)}] {scenario['label']}")
        run_task(
            task=scenario["task"],
            agent_id=scenario["agent_id"],
            system=scenario.get("system"),
        )
        time.sleep(1)

    print("\n" + "=" * 60)
    print("All scenarios complete.")
    print()
    print("Check the dashboard:")
    print("  Sessions   : http://localhost:5173/sessions")
    print("  Security   : http://localhost:5173/security")
    print("  Metrics    : http://localhost:5173/metrics")
    print("  Policy     : http://localhost:5173/policy  (then Generate Suggestions)")


if __name__ == "__main__":
    main()
