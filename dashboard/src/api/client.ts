/**
 * Typed API client for AgentBlackBox backend.
 * All requests include the API key header.
 */
import axios from "axios";
import type { Anomaly, AgentMetrics, ModelMetrics, MetricsOverview, AuditEvent, Session } from "../types/events";

const BASE_URL = import.meta.env.VITE_BACKEND_URL || "";
const API_KEY = import.meta.env.VITE_API_KEY || "dev-dashboard-key";

const http = axios.create({
  baseURL: BASE_URL,
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  },
});

// ─── Sessions ────────────────────────────────────────────────────────────────

export interface ListSessionsParams {
  agent_id?: string;
  provider?: string;
  limit?: number;
  offset?: number;
}

export interface ListSessionsResponse {
  sessions: Session[];
  count: number;
  offset: number;
  limit: number;
}

export async function listSessions(
  params: ListSessionsParams = {}
): Promise<ListSessionsResponse> {
  const res = await http.get("/v1/sessions", { params });
  return res.data;
}

export async function getSession(sessionId: string): Promise<Session> {
  const res = await http.get(`/v1/sessions/${sessionId}`);
  return res.data;
}

// ─── Events ──────────────────────────────────────────────────────────────────

export interface GetEventsParams {
  limit?: number;
  offset?: number;
  event_type?: string;
}

export interface GetEventsResponse {
  session_id: string;
  events: AuditEvent[];
  count: number;
  offset: number;
  limit: number;
}

export async function getSessionEvents(
  sessionId: string,
  params: GetEventsParams = {}
): Promise<GetEventsResponse> {
  const res = await http.get(`/v1/sessions/${sessionId}/events`, { params });
  return res.data;
}

// ─── Verification ─────────────────────────────────────────────────────────────

export interface VerifyResult {
  session_id: string;
  valid: boolean;
  total_events: number;
  first_failed_sequence: number | null;
  error_message: string | null;
  checked_at: string;
}

export async function verifySession(sessionId: string): Promise<VerifyResult> {
  const res = await http.get(`/v1/sessions/${sessionId}/verify`);
  return res.data;
}

// ─── Forensic Replay ──────────────────────────────────────────────────────────

export interface ForensicReplayResponse {
  session_id: string;
  chain_valid: boolean;
  chain_check: VerifyResult;
  report: {
    org_id: string;
    agent_id: string;
    provider: string;
    model: string;
    started_at_ns: number;
    ended_at_ns: number;
    duration_ms: number;
    total_events: number;
    llm_call_count: number;
    tool_call_count: number;
    error_count: number;
    pii_exposure_count: number;
    total_tokens: number;
    timeline: TimelineEntry[];
    data_access_map: DataAccessEntry[];
    risk_flags: RiskFlag[];
  };
  anomalies: AnomalyFlag[];
}

export interface TimelineEntry {
  sequence_number: number;
  event_id: string;
  event_type: string;
  label: string;
  severity: string;
  timestamp_ns: number;
  summary: string;
  details: Record<string, unknown>;
  pii_detected: string[];
  run_id: string;
  parent_run_id: string | null;
}

export interface DataAccessEntry {
  tool_name: string;
  call_count: number;
  first_seen_ns: number;
  last_seen_ns: number;
}

export interface RiskFlag {
  severity: string;
  flag_type: string;
  description: string;
  sequence_number: number | null;
  event_id: string | null;
}

export interface AnomalyFlag {
  rule_id: string;
  severity: string;
  description: string;
  event_id: string | null;
  sequence_number: number | null;
  metadata: Record<string, unknown>;
}

export async function getForensicReplay(sessionId: string): Promise<ForensicReplayResponse> {
  const res = await http.get(`/v1/sessions/${sessionId}/replay`);
  return res.data;
}

// ─── Compliance ───────────────────────────────────────────────────────────────

export type Regulation = "eu_ai_act" | "gdpr" | "hipaa" | "soc2";

export interface ComplianceReportResponse {
  regulation: string;
  session_id: string;
  org_id: string;
  generated_at: string;
  compliant: boolean;
  evidence: Record<string, unknown>;
  gaps: string[];
  recommendations: string[];
}

export async function generateComplianceReport(
  sessionId: string,
  regulation: Regulation
): Promise<ComplianceReportResponse> {
  const res = await http.post("/v1/reports/generate", {
    session_id: sessionId,
    regulation,
  });
  return res.data;
}

// ─── Anomalies ────────────────────────────────────────────────────────────────

export interface ListAnomaliesParams {
  session_id?: string;
  agent_id?: string;
  severity?: string;
  rule_id?: string;
  limit?: number;
  offset?: number;
}

export interface ListAnomaliesResponse {
  anomalies: Anomaly[];
  total: number;
  limit: number;
  offset: number;
}

export async function listAnomalies(
  params: ListAnomaliesParams = {}
): Promise<ListAnomaliesResponse> {
  const res = await http.get("/v1/anomalies", { params });
  return res.data;
}

// ─── Violations ───────────────────────────────────────────────────────────────

export interface Violation {
  id: number;
  rule_name: string;
  action: "BLOCK" | "ALERT" | "LOG";
  reason: string;
  event_type: string;
  session_id: string;
  agent_id: string;
  org_id: string;
  timestamp_ns: number;
  received_at: string | null;
}

export interface ListViolationsParams {
  session_id?: string;
  agent_id?: string;
  rule_name?: string;
  action?: string;
  limit?: number;
  offset?: number;
}

export interface ListViolationsResponse {
  violations: Violation[];
  total: number;
  limit: number;
  offset: number;
}

export async function listViolations(
  params: ListViolationsParams = {}
): Promise<ListViolationsResponse> {
  const res = await http.get("/v1/violations", { params });
  return res.data;
}

export interface ViolationSummaryEntry {
  rule_name: string;
  action: string;
  count: number;
  last_fired_ns: number;
}

export async function getViolationsSummary(): Promise<{
  summary: ViolationSummaryEntry[];
}> {
  const res = await http.get("/v1/violations/summary");
  return res.data;
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

export async function getMetricsOverview(): Promise<MetricsOverview> {
  const res = await http.get("/v1/metrics/overview");
  return res.data;
}

export interface ListAgentMetricsParams {
  agent_id?: string;
  limit?: number;
  offset?: number;
}

export interface ListAgentMetricsResponse {
  agents: AgentMetrics[];
  total: number;
  limit: number;
  offset: number;
}

export async function listAgentMetrics(
  params: ListAgentMetricsParams = {}
): Promise<ListAgentMetricsResponse> {
  const res = await http.get("/v1/metrics/agents", { params });
  return res.data;
}

export interface ListModelMetricsResponse {
  models: ModelMetrics[];
  total: number;
}

export async function listModelMetrics(): Promise<ListModelMetricsResponse> {
  const res = await http.get("/v1/metrics/models");
  return res.data;
}

// ─── Topology ─────────────────────────────────────────────────────────────────

import type { TopologyGraph, TopologyFilters } from "../pages/Topology/topology.types";
import type { SplunkPushRequest, ElasticPushRequest, PushResult } from "../pages/Export/export.types";

export async function getTopology(
  filters: TopologyFilters = { include_isolated: true, min_edge_calls: 1 }
): Promise<TopologyGraph> {
  const res = await http.get("/v1/topology", { params: filters });
  return res.data;
}

// ─── SIEM Export ───────────────────────────────────────────────────────────────

export async function pushToSplunk(req: SplunkPushRequest): Promise<PushResult> {
  const res = await http.post("/v1/export/splunk", req);
  return res.data;
}

export async function pushToElasticsearch(req: ElasticPushRequest): Promise<PushResult> {
  const res = await http.post("/v1/export/elasticsearch", req);
  return res.data;
}

// ─── Proxy — Tool Permissions ─────────────────────────────────────────────────

const PROXY_URL =
  import.meta.env.VITE_PROXY_URL || "http://localhost:8080";

const proxyHttp = axios.create({
  baseURL: PROXY_URL,
  headers: { "Content-Type": "application/json" },
});

export interface ToolPermissionRule {
  name: string;
  tools: string[];
  agents: string[];
  except_tools: string[];
  action: "BLOCK" | "ALERT" | "LOG";
  reason: string;
  enabled: boolean;
  arg_conditions: Array<{
    field: string;
    op: string;
    value?: unknown;
  }>;
}

export interface ToolPermissionsResponse {
  rules: ToolPermissionRule[];
  total: number;
  enabled: number;
}

export async function listToolPermissions(): Promise<ToolPermissionsResponse> {
  const res = await proxyHttp.get("/tool-permissions");
  return res.data;
}

export async function reloadToolPermissions(
  rules?: ToolPermissionRule[]
): Promise<{ status: string; rule_count: number; enabled_rule_count: number }> {
  const body = rules ? { rules } : {};
  const res = await proxyHttp.post("/tool-permissions/reload", body);
  return res.data;
}

// ─── Proxy — Security Benchmark ───────────────────────────────────────────────

export { PROXY_URL };

export interface CaseResult {
  text: string;
  category: string;
  expected: "attack" | "clean";
  score: number;
  detected: boolean;
  blocked: boolean;
  layer_scores: Record<string, number>;
  threats: string[];
  latency_ms: number;
}

export interface CategorySummary {
  name: string;
  total: number;
  detected: number;
  blocked: number;
  detection_rate: number;
  block_rate: number;
}

export interface BenchmarkReport {
  run_at: string;
  duration_s: number;
  total_attacks: number;
  total_clean: number;
  true_positives: number;
  false_negatives: number;
  true_negatives: number;
  false_positives: number;
  tpr: number;
  fpr: number;
  precision: number;
  f1: number;
  accuracy: number;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  latency_mean_ms: number;
  categories: CategorySummary[];
  active_layers: string[];
  layer_coverage: Record<string, number>;
  samples: CaseResult[];
  // Benchmark rigor fields
  contamination_pct?: number;
  paraphrase_tpr?: number;
  paraphrase_count?: number;
  paraphrase_detected?: number;
}

export async function runBenchmark(): Promise<BenchmarkReport> {
  const res = await proxyHttp.post("/benchmark/run");
  return res.data;
}

export async function fetchLastBenchmark(): Promise<BenchmarkReport | null> {
  const res = await proxyHttp.get("/benchmark/last");
  return res.data ?? null;  // endpoint returns JSON null when no report exists yet
}

export async function runExternalBenchmark(): Promise<BenchmarkReport> {
  const res = await proxyHttp.post("/benchmark/external/run");
  return res.data;
}

export async function fetchLastExternalBenchmark(): Promise<BenchmarkReport | null> {
  const res = await proxyHttp.get("/benchmark/external/last");
  return res.data ?? null;  // endpoint returns JSON null when no report exists yet
}

// ─── Policy Builder ────────────────────────────────────────────────────────────

export interface PolicyCondition {
  field: string;
  op: string;
  value?: unknown;
}

export interface PolicyRule {
  name: string;
  event_types: string[];
  conditions: PolicyCondition[];
  action: "BLOCK" | "ALERT" | "RATE_LIMIT" | "LOG" | "ALLOW";
  reason: string;
  enabled: boolean;
}

export interface PolicyRulesResponse {
  rules: PolicyRule[];
}

export interface ObservedTool {
  agent_id: string;
  tool_name: string;
  call_count: number;
  violation_count: number;
  last_seen: string | null;
}

export interface ObservedToolsResponse {
  tools: ObservedTool[];
  window_days: number;
  org_id: string;
}

/** Fetch active policy rules from the proxy */
export async function listPolicyRules(): Promise<PolicyRulesResponse> {
  const res = await proxyHttp.get("/policies");
  return res.data;
}

/** Save updated policy rules to the proxy (hot-reload) */
export async function savePolicyRules(rules: PolicyRule[]): Promise<{ status: string; rule_count: number }> {
  const res = await proxyHttp.post("/policies/reload", { rules });
  return res.data;
}

/** Observed tool usage per agent (backend) */
export async function listObservedTools(days = 30): Promise<ObservedToolsResponse> {
  const res = await http.get("/v1/policy/observed-tools", { params: { days } });
  return res.data;
}

// ─── Policy Suggestions ────────────────────────────────────────────────────────

export interface PolicySuggestion {
  id: string;
  title: string;
  description: string;
  evidence: string;
  rule: PolicyRule;
}

export interface PolicySuggestionsResponse {
  suggestions: PolicySuggestion[];
  window_days: number;
  total_events_analyzed: number;
  insufficient_data: boolean;
}

/** Auto-generated policy rule suggestions derived from observed traffic */
export async function getPolicySuggestions(days = 30): Promise<PolicySuggestionsResponse> {
  const res = await http.get("/v1/policy/suggestions", { params: { days } });
  return res.data;
}

// ─── Agent Registry ────────────────────────────────────────────────────────────

export interface Agent {
  agent_id: string;
  name: string;
  declared_purpose: string;
  allowed_tools: string[];
  owner: string | null;
  registered_at: string;
  updated_at: string;
}

export interface ListAgentsResponse {
  agents: Agent[];
  count: number;
}

export async function listAgents(): Promise<ListAgentsResponse> {
  const res = await http.get("/v1/agents");
  return res.data;
}

export async function registerAgent(body: {
  agent_id: string;
  name: string;
  declared_purpose?: string;
  allowed_tools?: string[];
  owner?: string;
}): Promise<{ status: string; agent_id: string }> {
  const res = await http.post("/v1/agents", body);
  return res.data;
}

export async function updateAgent(
  agentId: string,
  body: { name?: string; declared_purpose?: string; allowed_tools?: string[]; owner?: string }
): Promise<{ status: string; agent_id: string }> {
  const res = await http.patch(`/v1/agents/${agentId}`, body);
  return res.data;
}

export async function deleteAgent(agentId: string): Promise<void> {
  await http.delete(`/v1/agents/${agentId}`);
}

// ─── Compliance Audit Report ────────────────────────────────────────────────

export type ComplianceFramework = "owasp_asi_2026" | "eu_ai_act" | "hipaa" | "soc2";

export interface AuditControl {
  id: string;
  name: string;
  status: "pass" | "partial" | "fail";
  evidence: string;
  violation_detail: Array<{ rule: string; action: string; count: number }>;
}

export interface AuditReportSummary {
  total_sessions: number;
  llm_calls: number;
  tool_calls: number;
  memory_blocked: number;
  pii_events: number;
  error_count: number;
  agent_count: number;
  avg_latency_ms: number | null;
  blocked_count: number;
  alert_count: number;
  total_violations: number;
  anomalies: number;
  chain_verified_sessions: number;
  chain_valid_pct: number;
  overall_status: "pass" | "partial" | "fail";
}

export interface AuditReportViolation {
  id: number;
  rule_name: string;
  action: string;
  reason: string;
  event_type: string;
  session_id: string;
  agent_id: string;
  org_id: string;
  timestamp_ns: number;
}

export interface AuditReportAgent {
  agent_id: string;
  name: string;
  declared_purpose: string;
  allowed_tools: string[];
  owner: string | null;
  registered_at: string | null;
}

export interface AuditReport {
  report_id: string;
  generated_at: string;
  org_id: string;
  framework: ComplianceFramework;
  period: {
    from_iso: string;
    to_iso: string;
    from_ts_ns: number;
    to_ts_ns: number;
  };
  summary: AuditReportSummary;
  controls: AuditControl[];
  violations: AuditReportViolation[];
  agents: AuditReportAgent[];
}

export interface AuditReportParams {
  from_date?: string;
  to_date?: string;
  framework?: ComplianceFramework;
  agent_id?: string;
}

export async function getAuditReport(params: AuditReportParams = {}): Promise<AuditReport> {
  const res = await http.get("/v1/export/audit-report", { params });
  return res.data;
}

// ─── Tool Baselines ───────────────────────────────────────────────────────────

export interface BaselineSummary {
  agent_id: string;
  pending: number;
  approved: number;
  denied: number;
  total: number;
  last_seen_at: string | null;
  enforcement_active: boolean;
}

export interface AgentTool {
  tool_name: string;
  tool_schema: Record<string, unknown>;
  status: "pending" | "approved" | "denied";
  call_count: number;
  first_seen_at: string | null;
  last_seen_at: string | null;
  approved_at: string | null;
}

export interface AgentToolsResponse {
  agent_id: string;
  org_id: string;
  tools: AgentTool[];
  counts: { pending: number; approved: number; denied: number; total: number };
  enforcement_active: boolean;
}

/** Overview of all agents with baseline counts. */
export async function listBaselines(): Promise<{ baselines: BaselineSummary[]; count: number }> {
  const res = await http.get("/v1/baselines");
  return res.data;
}

/** Full tool list for one agent with status, schema, and call counts. */
export async function listAgentTools(agentId: string): Promise<AgentToolsResponse> {
  const res = await http.get(`/v1/baselines/${agentId}/tools`);
  return res.data;
}

/** Approve a single tool for an agent. */
export async function approveTool(agentId: string, toolName: string): Promise<void> {
  await http.post(`/v1/baselines/${agentId}/tools/${toolName}/approve`);
}

/** Deny a single tool for an agent. */
export async function denyTool(agentId: string, toolName: string): Promise<void> {
  await http.post(`/v1/baselines/${agentId}/tools/${toolName}/deny`);
}

/** Approve every pending tool for an agent in one click. */
export async function approveAllTools(
  agentId: string
): Promise<{ approved_count: number; approved_tools: string[] }> {
  const res = await http.post(`/v1/baselines/${agentId}/tools/approve-all`);
  return res.data;
}
