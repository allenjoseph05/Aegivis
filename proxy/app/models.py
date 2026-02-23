# AUTO-GENERATED — do not edit manually.
# Source: shared/schema/events.json
# Regenerate: python shared/schema/generate.py
from __future__ import annotations
from typing import Any, List, Optional, Union
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
