# AUTO-GENERATED — do not edit manually.
# Source: shared/schema/events.json
# Regenerate: python shared/schema/generate.py
from __future__ import annotations
from typing import Any
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


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
    content: str | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str | dict


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class LLMCallStartPayload(BaseModel):
    messages: list[Message]
    tools_available: list[ToolDefinition] = Field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = None
    extra_params: dict | None = None


class LLMCallEndPayload(BaseModel):
    response_text: str | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    token_usage: TokenUsage | None = None
    latency_ms: float | None = None
    http_status: int | None = None


class ToolCallStartPayload(BaseModel):
    tool_name: str
    tool_description: str | None = None
    tool_input: dict | str | None = None
    tool_call_id: str


class ToolCallEndPayload(BaseModel):
    tool_name: str
    tool_call_id: str
    tool_output_masked: str | None = None
    tool_output_hash: str | None = None
    pii_fields_detected: list[str] = Field(default_factory=list)
    success: bool = True


class AgentFinishPayload(BaseModel):
    final_output: str | None = None
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    session_duration_ms: float | None = None


class SystemErrorPayload(BaseModel):
    error_code: str | int | None = None
    error_message: str
    provider: str = "unknown"
    http_status: int | None = None


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
    parent_run_id: str | None = None
    event_type: EventType
    payload: dict
    payload_hash: str | None = None
    pii_detected: list[str] = Field(default_factory=list)
    timestamp_ns: int
    sequence_number: int
    previous_hash: str
    current_hash: str


class EventBatch(BaseModel):
    events: list[AuditEvent]
    batch_id: str
    sent_at_ns: int
