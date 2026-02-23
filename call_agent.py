"""
Real OpenAI call through AgentBlackBox proxy.

This is identical to a normal OpenAI script — the ONLY difference
is base_url points to localhost:8080 instead of api.openai.com.

Run:
    python call_agent.py
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Choose your backend ───────────────────────────────────────────────────────
# Option A: OpenAI (requires credits at platform.openai.com/settings/billing)
# client = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"],
#     base_url="http://localhost:8080/openai/v1",
# )
# model = "gpt-4o-mini"

# Option B: Ollama (free, local — install from https://ollama.com then: ollama pull llama3.2)
client = OpenAI(
    api_key="ollama",                           # any non-empty string works
    base_url="http://localhost:8080/ollama/v1", # proxy routes to local Ollama
)
model = "llama3.2"
# ─────────────────────────────────────────────────────────────────────────────

print(f"Sending request through AgentBlackBox proxy (model: {model})...")

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What is 2 + 2? Answer in one sentence."},
    ],
)

print(f"\nResponse: {response.choices[0].message.content}")
print(f"\nCheck dashboard: http://localhost:5173")
print("Your LLM call was captured automatically — no SDK needed.")
