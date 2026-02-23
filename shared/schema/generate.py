#!/usr/bin/env python3
"""
Code generator: shared/schema/events.json → Pydantic models + TypeScript types.

Usage:
    python shared/schema/generate.py

Outputs:
    proxy/app/models.py          (Pydantic v2)
    backend/app/models_gen.py    (Pydantic v2)
    dashboard/src/types/events.ts (TypeScript)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

SCHEMA_PATH = Path(__file__).parent / "events.json"

PYDANTIC_HEADER = '''\
# AUTO-GENERATED — do not edit manually.
# Source: shared/schema/events.json
# Regenerate: python shared/schema/generate.py
from __future__ import annotations
from typing import Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
'''

TS_HEADER = """\
// AUTO-GENERATED — do not edit manually.
// Source: shared/schema/events.json
// Regenerate: python shared/schema/generate.py
"""


def generate_pydantic() -> str:
    lines = [PYDANTIC_HEADER]

    lines.append("""
class EventType(str, Enum):
    LLM_CALL_START = "LLM_CALL_START"
    LLM_CALL_END = "LLM_CALL_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_END = "TOOL_CALL_END"
    AGENT_FINISH = "AGENT_FINISH"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CHECKPOINT = "CHECKPOINT"


class InterceptionLayer(str, Enum):
    PROXY = "proxy"
    OTEL = "otel"
    SDK = "sdk"
    EBPF = "ebpf"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[dict] = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Union[str, dict]


class TokenUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMCallStartPayload(BaseModel):
    messages: List[Message]
    tools_available: List[ToolDefinition] = Field(default_factory=list)
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None
    extra_params: Optional[dict] = None


class LLMCallEndPayload(BaseModel):
    response_text: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    token_usage: Optional[TokenUsage] = None
    latency_ms: Optional[float] = None
    http_status: Optional[int] = None


class ToolCallStartPayload(BaseModel):
    tool_name: str
    tool_description: Optional[str] = None
    tool_input: Optional[Union[dict, str]] = None
    tool_call_id: str


class ToolCallEndPayload(BaseModel):
    tool_name: str
    tool_call_id: str
    tool_output_masked: Optional[str] = None
    tool_output_hash: Optional[str] = None
    pii_fields_detected: List[str] = Field(default_factory=list)
    success: bool = True


class AgentFinishPayload(BaseModel):
    final_output: Optional[str] = None
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    session_duration_ms: Optional[float] = None


class SystemErrorPayload(BaseModel):
    error_code: Optional[Union[str, int]] = None
    error_message: str
    provider: str = "unknown"
    http_status: Optional[int] = None


class CheckpointPayload(BaseModel):
    merkle_root: str
    events_covered: int
    from_sequence: int
    to_sequence: int


class AuditEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    event_id: str = Field(description="ULID (lexicographically sortable by time)")
    schema_version: str = Field(default="1.0")
    org_id: str
    session_id: str
    agent_id: str
    provider: Provider
    model: str
    interception_layer: InterceptionLayer = InterceptionLayer.PROXY
    run_id: str
    parent_run_id: Optional[str] = None
    event_type: EventType
    payload: dict
    payload_hash: Optional[str] = None
    pii_detected: List[str] = Field(default_factory=list)
    timestamp_ns: int
    sequence_number: int
    previous_hash: str
    current_hash: str


class EventBatch(BaseModel):
    events: List[AuditEvent]
    batch_id: str
    sent_at_ns: int
""")

    return "\n".join(lines)


def generate_typescript() -> str:
    return TS_HEADER + """
export type EventType =
  | "LLM_CALL_START"
  | "LLM_CALL_END"
  | "TOOL_CALL_START"
  | "TOOL_CALL_END"
  | "AGENT_FINISH"
  | "SYSTEM_ERROR"
  | "CHECKPOINT";

export type InterceptionLayer = "proxy" | "otel" | "sdk" | "ebpf";

export type Provider = "openai" | "anthropic" | "google" | "azure" | "ollama" | "unknown";

export interface Message {
  role: "system" | "user" | "assistant" | "tool" | "function";
  content?: string | null;
  tool_call_id?: string | null;
  name?: string | null;
}

export interface ToolDefinition {
  name: string;
  description?: string | null;
  parameters?: Record<string, unknown> | null;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: string | Record<string, unknown>;
}

export interface TokenUsage {
  prompt_tokens?: number | null;
  completion_tokens?: number | null;
  total_tokens?: number | null;
}

export interface LLMCallStartPayload {
  messages: Message[];
  tools_available: ToolDefinition[];
  temperature?: number | null;
  max_tokens?: number | null;
  stream?: boolean | null;
  extra_params?: Record<string, unknown> | null;
}

export interface LLMCallEndPayload {
  response_text?: string | null;
  finish_reason?: string | null;
  tool_calls: ToolCall[];
  token_usage?: TokenUsage | null;
  latency_ms?: number | null;
  http_status?: number | null;
}

export interface ToolCallStartPayload {
  tool_name: string;
  tool_description?: string | null;
  tool_input?: Record<string, unknown> | string | null;
  tool_call_id: string;
}

export interface ToolCallEndPayload {
  tool_name: string;
  tool_call_id: string;
  tool_output_masked?: string | null;
  tool_output_hash?: string | null;
  pii_fields_detected: string[];
  success: boolean;
}

export interface AgentFinishPayload {
  final_output?: string | null;
  total_llm_calls: number;
  total_tool_calls: number;
  session_duration_ms?: number | null;
}

export interface SystemErrorPayload {
  error_code?: string | number | null;
  error_message: string;
  provider: string;
  http_status?: number | null;
}

export interface CheckpointPayload {
  merkle_root: string;
  events_covered: number;
  from_sequence: number;
  to_sequence: number;
}

export type EventPayload =
  | LLMCallStartPayload
  | LLMCallEndPayload
  | ToolCallStartPayload
  | ToolCallEndPayload
  | AgentFinishPayload
  | SystemErrorPayload
  | CheckpointPayload
  | Record<string, unknown>;

export interface AuditEvent {
  event_id: string;
  schema_version: string;
  org_id: string;
  session_id: string;
  agent_id: string;
  provider: Provider;
  model: string;
  interception_layer: InterceptionLayer;
  run_id: string;
  parent_run_id?: string | null;
  event_type: EventType;
  payload: EventPayload;
  payload_hash?: string | null;
  pii_detected: string[];
  timestamp_ns: number;
  sequence_number: number;
  previous_hash: string;
  current_hash: string;
}

export interface EventBatch {
  events: AuditEvent[];
  batch_id: string;
  sent_at_ns: number;
}

export interface Session {
  session_id: string;
  org_id: string;
  agent_id: string;
  provider: Provider;
  model: string;
  started_at: string;
  last_event_at: string;
  event_count: number;
  llm_call_count: number;
  tool_call_count: number;
  total_tokens?: number | null;
  chain_valid: boolean;
  has_errors: boolean;
}
"""


def main():
    schema = json.loads(SCHEMA_PATH.read_text())
    root = SCHEMA_PATH.parent.parent.parent  # agentblackbox root

    pydantic_code = generate_pydantic()
    ts_code = generate_typescript()

    # Write to proxy
    proxy_models = root / "proxy" / "app" / "models.py"
    proxy_models.write_text(pydantic_code, encoding="utf-8")
    print(f"Written: {proxy_models}")

    # Write to backend
    backend_models = root / "backend" / "app" / "models_gen.py"
    backend_models.write_text(pydantic_code, encoding="utf-8")
    print(f"Written: {backend_models}")

    # Write TypeScript types
    ts_types = root / "dashboard" / "src" / "types" / "events.ts"
    ts_types.write_text(ts_code, encoding="utf-8")
    print(f"Written: {ts_types}")

    print("\nCode generation complete.")


if __name__ == "__main__":
    main()
