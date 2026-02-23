// AUTO-GENERATED — do not edit manually.
// Source: shared/schema/events.json
// Regenerate: python shared/schema/generate.py

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

export interface Anomaly {
  id: number;
  session_id: string;
  agent_id: string;
  org_id: string;
  rule_id: string;
  severity: "critical" | "high" | "medium" | "low";
  description: string;
  event_id?: string | null;
  sequence_number?: number | null;
  metadata: Record<string, unknown>;
  detected_at: string;
}
