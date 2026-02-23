"""
Tool call demo through AgentBlackBox proxy.

Shows the full agent loop:
  LLM_CALL_START -> LLM_CALL_END (tool_calls) ->
  TOOL_CALL_START -> TOOL_CALL_END ->
  LLM_CALL_START -> LLM_CALL_END (final answer)

All events captured automatically. Check http://localhost:5173 while this runs.
"""
import json
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:8080/ollama/v1",
)

# ── Define tools the agent can call ──────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. London",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a math calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '15 * 4 + 7'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


def run_tool(name: str, args: dict) -> str:
    """Fake tool executor — returns hardcoded results for demo purposes."""
    if name == "get_weather":
        city = args.get("city", "unknown")
        return json.dumps({
            "city": city,
            "temperature": "22°C",
            "condition": "Partly cloudy",
            "humidity": "65%",
        })
    elif name == "calculator":
        expr = args.get("expression", "0")
        try:
            result = eval(expr, {"__builtins__": {}})  # safe: no builtins
            return json.dumps({"expression": expr, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})
    return json.dumps({"error": f"Unknown tool: {name}"})


# ── Agent loop ────────────────────────────────────────────────────────────────

messages = [
    {
        "role": "user",
        "content": "What is the weather in London? Also calculate 15 * 4 + 7.",
    }
]

print("Starting agent loop through AgentBlackBox proxy...")
print("Dashboard: http://localhost:5173\n")

step = 0
while True:
    step += 1
    print(f"Step {step}: Calling LLM...")

    response = client.chat.completions.create(
        model="llama3.2",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    msg = response.choices[0].message
    finish_reason = response.choices[0].finish_reason
    print(f"         finish_reason={finish_reason}")

    # Add assistant response to history
    messages.append(msg)

    # If no tool calls, we have the final answer
    if not msg.tool_calls:
        print(f"\nFinal answer:\n{msg.content}")
        break

    # Execute each tool the LLM requested
    for tool_call in msg.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        print(f"         Tool call: {name}({args})")

        result = run_tool(name, args)
        print(f"         Tool result: {result}")

        # Add tool result back into conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # Safety: stop after 5 steps to avoid infinite loops
    if step >= 5:
        print("\nMax steps reached.")
        break

print("\nDone! Open the dashboard and click the session to see the full event timeline.")
print("You should see: LLM_CALL_START -> LLM_CALL_END -> LLM_CALL_START -> LLM_CALL_END")
print("The proxy captures tool_calls in the LLM_CALL_END payload automatically.")
