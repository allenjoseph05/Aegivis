"""
AWS Bedrock provider — request/response parser.

Supports two Bedrock APIs:
  1. Converse API  — unified format across Claude, Llama, Mistral, Titan, etc.
  2. InvokeModel   — raw per-model JSON (Claude, Llama, Mistral supported)

The proxy uses its own boto3 client (via bedrock_gateway.py) to forward
requests.  Incoming SigV4 auth headers are ignored — the proxy authenticates
with its own AWS credentials.

Route: POST /bedrock/model/{modelId}/converse
       POST /bedrock/model/{modelId}/converse-stream   (buffered, returned as JSON)
       POST /bedrock/model/{modelId}/invoke
       POST /bedrock/model/{modelId}/invoke-with-response-stream (buffered)

Agent setup (only change required):
    AWS_ENDPOINT_URL_BEDROCK=http://localhost:8080/bedrock
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER_NAME = "bedrock"

# ── Model family detection ────────────────────────────────────────────────────

_MODEL_PREFIXES: list[tuple[str, str]] = [
    ("anthropic.",  "anthropic"),
    ("meta.",       "meta"),
    ("amazon.",     "amazon"),
    ("mistral.",    "mistral"),
    ("cohere.",     "cohere"),
    ("ai21.",       "ai21"),
    ("stability.",  "stability"),
]


def detect_model_family(model_id: str) -> str:
    """Return the model provider family from a Bedrock model ID.

    Examples:
        "anthropic.claude-3-5-sonnet-20241022-v2:0" → "anthropic"
        "meta.llama3-8b-instruct-v1:0"              → "meta"
        "amazon.titan-text-express-v1"               → "amazon"
        "mistral.mixtral-8x7b-instruct-v0:1"         → "mistral"
    """
    lowered = model_id.lower()
    # Cross-region inference profiles look like "us.anthropic.claude-..." — strip prefix
    for part in lowered.split("."):
        for prefix, family in _MODEL_PREFIXES:
            if (part + ".") == prefix or lowered.startswith(prefix):
                return family
    return "unknown"


# ── Content block helpers ─────────────────────────────────────────────────────

def _flatten_content_blocks(content: Any) -> str:
    """Flatten a Bedrock content block list into a plain string for scanning."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif "text" in block:
            parts.append(block["text"])
        elif "toolResult" in block:
            tr = block["toolResult"]
            for c in tr.get("content") or []:
                if isinstance(c, dict) and "text" in c:
                    parts.append(c["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            parts.append(f"[tool:{tu.get('name','')} {json.dumps(tu.get('input',{}))}]")
        elif "image" in block:
            parts.append("[image]")
        elif "document" in block:
            parts.append("[document]")
    return " ".join(parts)


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Normalize Bedrock Converse messages → canonical {role, content: str} list."""
    out: list[dict] = []
    for msg in messages:
        role    = msg.get("role", "user")
        content = _flatten_content_blocks(msg.get("content", ""))
        out.append({"role": role, "content": content})
    return out


# ── Converse API ──────────────────────────────────────────────────────────────

def extract_converse_request(model_id: str, body: dict) -> dict:
    """Translate a Bedrock Converse API request body → canonical Aegivis format."""
    messages = _normalize_messages(body.get("messages") or [])

    # Inject system prompt as first message (mirrors Anthropic handling)
    system_blocks = body.get("system") or []
    if isinstance(system_blocks, str):
        system_text = system_blocks
    else:
        system_text = " ".join(
            b.get("text", "") for b in system_blocks if isinstance(b, dict)
        )
    if system_text:
        messages = [{"role": "system", "content": system_text}] + messages

    # Normalize tools from toolConfig
    tools: list[dict] = []
    tool_config = body.get("toolConfig") or {}
    for tool in tool_config.get("tools") or []:
        spec = tool.get("toolSpec") or tool
        tools.append({
            "name":        spec.get("name", ""),
            "description": spec.get("description"),
            "parameters":  (spec.get("inputSchema") or {}).get("json"),
        })

    inference = body.get("inferenceConfig") or {}
    return {
        "model":       model_id,
        "messages":    messages,
        "tools":       tools,
        "temperature": inference.get("temperature"),
        "max_tokens":  inference.get("maxTokens"),
        "stream":      False,
        "extra_params": None,
    }


def parse_converse_response(body: dict) -> dict:
    """Translate a Bedrock Converse API response → canonical Aegivis format."""
    output   = body.get("output") or {}
    message  = output.get("message") or {}
    content  = message.get("content") or []

    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        if "text" in block:
            text_parts.append(block["text"])
        elif "toolUse" in block:
            tu = block["toolUse"]
            tool_calls.append({
                "id":        tu.get("toolUseId", ""),
                "name":      tu.get("name", ""),
                "arguments": json.dumps(tu.get("input") or {}),
            })

    usage = body.get("usage") or {}
    token_usage = None
    if usage:
        token_usage = {
            "input_tokens":  usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens":  usage.get("totalTokens", 0),
        }

    return {
        "response_text": " ".join(text_parts) if text_parts else None,
        "finish_reason": body.get("stopReason", "end_turn"),
        "tool_calls":    tool_calls,
        "token_usage":   token_usage,
    }


def assemble_converse_stream(event_stream) -> dict:
    """
    Consume a boto3 converse_stream EventStream and return a Converse-API-shaped
    response dict (same shape as converse() response).

    EventStream events of interest:
      contentBlockStart    — toolUse block starts (capture toolUseId + name)
      contentBlockDelta    — text delta or toolInput delta
      messageStop          — stopReason
      metadata             — usage stats
    """
    text_parts: list[str]  = []
    stop_reason: str        = "end_turn"
    usage: dict             = {}

    # tool_use accumulation: index → {toolUseId, name, input_json_parts}
    tool_use_map: dict[int, dict] = {}
    current_tool_index: int | None = None

    try:
        for event in event_stream:
            if "contentBlockStart" in event:
                start = event["contentBlockStart"]
                idx   = start.get("contentBlockIndex", 0)
                if "toolUse" in (start.get("start") or {}):
                    tu = start["start"]["toolUse"]
                    tool_use_map[idx] = {
                        "toolUseId": tu.get("toolUseId", ""),
                        "name":      tu.get("name", ""),
                        "parts":     [],
                    }
                    current_tool_index = idx

            elif "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta") or {}
                idx   = event["contentBlockDelta"].get("contentBlockIndex", 0)
                if "text" in delta:
                    text_parts.append(delta["text"])
                elif "toolUse" in delta:
                    if idx in tool_use_map:
                        tool_use_map[idx]["parts"].append(
                            delta["toolUse"].get("input", "")
                        )

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", "end_turn")

            elif "metadata" in event:
                usage = event["metadata"].get("usage") or {}

    except Exception as exc:
        logger.warning("Error consuming Bedrock EventStream: %s", exc)

    tool_calls = []
    for entry in tool_use_map.values():
        raw_input = "".join(entry["parts"])
        try:
            parsed_input = json.loads(raw_input) if raw_input else {}
        except json.JSONDecodeError:
            parsed_input = {}
        tool_calls.append({
            "id":        entry["toolUseId"],
            "name":      entry["name"],
            "arguments": json.dumps(parsed_input),
        })

    token_usage = None
    if usage:
        token_usage = {
            "input_tokens":  usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "total_tokens":  usage.get("totalTokens", 0),
        }

    return {
        "response_text": "".join(text_parts) or None,
        "finish_reason": stop_reason,
        "tool_calls":    tool_calls,
        "token_usage":   token_usage,
    }


def rebuild_converse_body(canonical: dict, original_body: dict) -> dict:
    """
    Translate canonical format back to a Bedrock Converse API request dict.
    Used when the proxy modifies messages (canary injection / spotlighting).
    Preserves non-message fields from the original body.
    """
    messages: list[dict] = []
    system_text: str = ""

    for msg in canonical.get("messages") or []:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
        else:
            messages.append({
                "role":    role,
                "content": [{"text": content}],
            })

    result = dict(original_body)
    result["messages"] = messages
    if system_text:
        result["system"] = [{"text": system_text}]

    return result


# ── InvokeModel API ───────────────────────────────────────────────────────────

def parse_invoke_request(model_id: str, body: dict) -> dict:
    """Parse an InvokeModel request body → canonical format.

    Each model family uses a different request format.
    """
    family = detect_model_family(model_id)

    if family == "anthropic":
        # Anthropic Messages API format (identical to direct Anthropic API)
        from .anthropic import extract_request_params as anthropic_extract
        params = anthropic_extract(body)
        params["model"] = model_id
        return params

    if family == "meta":
        # Llama chat format: {"prompt": "<|...|>...", "max_gen_len": N, "temperature": F}
        prompt = body.get("prompt", "")
        messages = _parse_llama_prompt(prompt) or [{"role": "user", "content": prompt}]
        return {
            "model":       model_id,
            "messages":    messages,
            "tools":       [],
            "temperature": body.get("temperature"),
            "max_tokens":  body.get("max_gen_len"),
            "stream":      False,
            "extra_params": None,
        }

    if family == "mistral":
        # Mistral Bedrock format: {"prompt": "<s>[INST]...[/INST]", "max_tokens": N}
        prompt = body.get("prompt", "")
        messages = _parse_mistral_prompt(prompt) or [{"role": "user", "content": prompt}]
        return {
            "model":       model_id,
            "messages":    messages,
            "tools":       [],
            "temperature": body.get("temperature"),
            "max_tokens":  body.get("max_tokens"),
            "stream":      False,
            "extra_params": None,
        }

    if family == "amazon":
        # Titan / Nova: {"inputText": "...", "textGenerationConfig": {...}}
        text = body.get("inputText", "")
        cfg  = body.get("textGenerationConfig") or {}
        return {
            "model":       model_id,
            "messages":    [{"role": "user", "content": text}],
            "tools":       [],
            "temperature": cfg.get("temperature"),
            "max_tokens":  cfg.get("maxTokenCount"),
            "stream":      False,
            "extra_params": None,
        }

    # Unknown family — capture as raw for audit
    return {
        "model":       model_id,
        "messages":    [{"role": "user", "content": str(body)[:1000]}],
        "tools":       [],
        "temperature": None,
        "max_tokens":  None,
        "stream":      False,
        "extra_params": {"raw_body": body},
    }


def parse_invoke_response(model_id: str, body: dict) -> dict:
    """Parse an InvokeModel response body → canonical format."""
    family = detect_model_family(model_id)

    if family == "anthropic":
        from .anthropic import parse_response as anthropic_parse
        return anthropic_parse(body)

    if family == "meta":
        # {"generation": "...", "prompt_token_count": N, "generation_token_count": N}
        token_usage = None
        if "prompt_token_count" in body:
            in_tok  = body.get("prompt_token_count", 0)
            out_tok = body.get("generation_token_count", 0)
            token_usage = {
                "input_tokens":  in_tok,
                "output_tokens": out_tok,
                "total_tokens":  in_tok + out_tok,
            }
        return {
            "response_text": body.get("generation"),
            "finish_reason": body.get("stop_reason", "stop"),
            "tool_calls":    [],
            "token_usage":   token_usage,
        }

    if family == "mistral":
        # {"outputs": [{"text": "...", "stop_reason": "stop"}]}
        outputs = body.get("outputs") or []
        text    = outputs[0].get("text") if outputs else None
        reason  = outputs[0].get("stop_reason", "stop") if outputs else "stop"
        return {
            "response_text": text,
            "finish_reason": reason,
            "tool_calls":    [],
            "token_usage":   None,
        }

    if family == "amazon":
        # Titan: {"results": [{"outputText": "...", "completionReason": "..."}]}
        results = body.get("results") or []
        text    = results[0].get("outputText") if results else None
        reason  = results[0].get("completionReason", "stop") if results else "stop"
        return {
            "response_text": text,
            "finish_reason": reason,
            "tool_calls":    [],
            "token_usage":   None,
        }

    return {
        "response_text": str(body)[:500],
        "finish_reason": "stop",
        "tool_calls":    [],
        "token_usage":   None,
    }


# ── Prompt format parsers ─────────────────────────────────────────────────────

_LLAMA3_TURN = re.compile(
    r"<\|start_header_id\|>(user|assistant|system)<\|end_header_id\|>\s*(.*?)<\|eot_id\|>",
    re.DOTALL,
)

def _parse_llama_prompt(prompt: str) -> list[dict]:
    """Best-effort parse of Llama 3 chat prompt → messages list."""
    messages = []
    for m in _LLAMA3_TURN.finditer(prompt):
        role, content = m.group(1), m.group(2).strip()
        messages.append({"role": role, "content": content})
    return messages


_MISTRAL_TURN = re.compile(r"\[INST\](.*?)\[/INST\](.*?)(?=\[INST\]|$)", re.DOTALL)

def _parse_mistral_prompt(prompt: str) -> list[dict]:
    """Best-effort parse of Mistral [INST] prompt → messages list."""
    messages = []
    for m in _MISTRAL_TURN.finditer(prompt):
        user_text  = m.group(1).strip()
        asst_text  = m.group(2).strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
        if asst_text:
            messages.append({"role": "assistant", "content": asst_text})
    return messages


# ── BedrockProvider class ─────────────────────────────────────────────────────

class BedrockProvider:
    """Provider class shim — used by the route handler for event typing.

    The actual forwarding is done via bedrock_gateway.py (boto3).
    extract_request_params / parse_response are called with model_id injected.
    """
    name = PROVIDER_NAME

    @staticmethod
    def extract_request_params(body: dict, model_id: str = "unknown") -> dict:
        return extract_converse_request(model_id, body)

    @staticmethod
    def parse_response(body: dict, model_id: str = "unknown") -> dict:
        return parse_converse_response(body)
