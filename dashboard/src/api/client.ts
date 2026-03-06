/**
 * Typed API client for Aegivis backend.
 * All requests include the API key header.
 */
import axios from "axios";
import type { Anomaly, AgentMetrics, ModelMetrics, MetricsOverview, AuditEvent, Session } from "../types/events";

export const BASE_URL = import.meta.env.VITE_BACKEND_URL || "";
export const API_KEY = import.meta.env.VITE_API_KEY || "dev-dashboard-key";

const http = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  },
});

// Global error handler — surfaces auth failures and server errors as console
// warnings so every page gets consistent behaviour without duplicating logic.
http.interceptors.response.use(
  (response) => response,
  (error) => {
    const status = error.response?.status;
    if (status === 401 || status === 403) {
      console.warn("[Aegivis] Auth error — check your API key in Settings.");
    } else if (status >= 500) {
      console.warn(`[Aegivis] Backend error ${status}: ${error.response?.data?.detail ?? "unknown"}`);
    }
    return Promise.reject(error);
  }
);

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

export type Regulation = "eu_ai_act" | "gdpr" | "hipaa" | "soc2" | "owasp_asi_2026";

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
  const today = new Date().toISOString().slice(0, 10);
  const report = await getAuditReport({
    session_id: sessionId,
    framework: regulation as ComplianceFramework,
    from_date: "2020-01-01",
    to_date: today,
  });

  const gaps = report.controls
    .filter((c) => c.status !== "pass")
    .map((c) => `[${c.id}] ${c.name}: ${c.evidence}`);

  const recommendations: string[] = [];
  const gapsText = gaps.join(" ").toLowerCase();
  if (gapsText.includes("inject") || gapsText.includes("prompt")) {
    recommendations.push("Enable ML-based injection detection and lower alert thresholds.");
  }
  if (gapsText.includes("pii")) {
    recommendations.push("Review PII masking coverage; consider enabling PII redaction in proxy config.");
  }
  if (gapsText.includes("chain") || gapsText.includes("hash")) {
    recommendations.push("Investigate chain integrity failures via Session Detail → Verify Chain.");
  }
  if (gapsText.includes("tool") || gapsText.includes("ssrf") || gapsText.includes("rce")) {
    recommendations.push("Tighten tool allowlist; enable RCE/SSRF blocking rules.");
  }
  if (recommendations.length === 0 && gaps.length > 0) {
    recommendations.push("Review flagged controls and address partial compliance items.");
  }

  return {
    regulation,
    session_id: sessionId,
    org_id: report.org_id,
    generated_at: report.generated_at,
    compliant: report.summary.overall_status === "pass",
    evidence: report.summary as unknown as Record<string, unknown>,
    gaps,
    recommendations,
  };
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

export interface ViolationsStatBucket {
  bucket: string;
  action: string;
  count: number;
}

export interface ViolationsStatsResponse {
  interval: string;
  days: number;
  buckets: ViolationsStatBucket[];
}

export async function getViolationsStats(
  interval: "hour" | "day" = "day",
  days = 7
): Promise<ViolationsStatsResponse> {
  const res = await http.get("/v1/violations/stats", { params: { interval, days } });
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

// ─── SIEM Export ───────────────────────────────────────────────────────────────

import type { SplunkPushRequest, ElasticPushRequest, PushResult } from "../pages/Export/export.types";

export async function pushToSplunk(req: SplunkPushRequest): Promise<PushResult> {
  const res = await http.post("/v1/export/splunk", req);
  return res.data;
}

export async function pushToElasticsearch(req: ElasticPushRequest): Promise<PushResult> {
  const res = await http.post("/v1/export/elasticsearch", req);
  return res.data;
}

// ─── Proxy — Tool Permissions ─────────────────────────────────────────────────

// Proxy management calls go through nginx (/proxy/ → proxy:8080) to avoid
// direct browser→localhost:8080 calls that break on IPv6/WSL2.
// VITE_PROXY_URL can override for non-Docker setups (e.g. bare-metal dev).
const PROXY_URL =
  import.meta.env.VITE_PROXY_URL || "http://localhost:8080";

const proxyHttp = axios.create({
  baseURL: import.meta.env.VITE_PROXY_URL || "/proxy",
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
  // Long-context evasion probe (held-out — requires WalledGuard 8k context to catch)
  long_context_tpr?: number;
  long_context_count?: number;
  long_context_detected?: number;
}

// ─── Benchmark ────────────────────────────────────────────────────────────────

export interface BenchmarkRunOptions {
  useClassifier?: boolean;
}

export async function runBenchmark(opts?: BenchmarkRunOptions): Promise<BenchmarkReport> {
  const res = await proxyHttp.post("/benchmark/run", {
    use_classifier: opts?.useClassifier ?? false,
  });
  return res.data;
}

export async function fetchLastBenchmark(): Promise<BenchmarkReport | null> {
  const res = await proxyHttp.get("/benchmark/last");
  return res.data ?? null;
}

export async function runExternalBenchmark(opts?: BenchmarkRunOptions): Promise<BenchmarkReport> {
  const res = await proxyHttp.post("/benchmark/external/run", {
    use_classifier: opts?.useClassifier ?? false,
  });
  return res.data;
}

export async function fetchLastExternalBenchmark(): Promise<BenchmarkReport | null> {
  const res = await proxyHttp.get("/benchmark/external/last");
  return res.data ?? null;
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

export type ComplianceFramework = "owasp_asi_2026" | "eu_ai_act" | "hipaa" | "soc2" | "gdpr";

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
  session_id?: string;
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

// ─── Security Playground ──────────────────────────────────────────────────────

export type PlaygroundMode = "prompt" | "tool_output" | "tool_call";

export interface PlaygroundLayerResult {
  id: string;
  name: string;
  score: number;
  label: string;
  triggered: boolean;
  details: Record<string, unknown>;
}

export interface PlaygroundScanResult {
  overall_score: number;
  overall_label: "safe" | "suspicious" | "malicious";
  would_block: boolean;
  would_alert: boolean;
  layers: PlaygroundLayerResult[];
  latency_ms: number;
  thresholds: { block: number; alert: number };
}

export interface PlaygroundScanRequest {
  mode: PlaygroundMode;
  text?: string;
  tool_name?: string;
  tool_args?: Record<string, unknown>;
}

export async function scanPlayground(
  req: PlaygroundScanRequest
): Promise<PlaygroundScanResult> {
  const res = await proxyHttp.post("/playground/scan", req);
  return res.data;
}

// ─── Violations — extended ────────────────────────────────────────────────────

export interface ViolationContext {
  violation: Violation & { is_false_positive: boolean; marked_fp_at: string | null };
  session_violations: Array<{
    id: number;
    rule_name: string;
    action: string;
    reason: string;
    event_type: string;
    timestamp_ns: number;
  }>;
  session_violation_count: number;
}

export async function getViolationContext(id: number): Promise<ViolationContext> {
  const res = await http.get(`/v1/violations/${id}/context`);
  return res.data;
}

export async function markFalsePositive(id: number): Promise<void> {
  await http.post(`/v1/violations/${id}/mark-fp`);
}

export async function unmarkFalsePositive(id: number): Promise<void> {
  await http.post(`/v1/violations/${id}/unmark-fp`);
}

export async function countViolations(params: ListViolationsParams = {}): Promise<{ total: number }> {
  const res = await http.get("/v1/violations/count", { params });
  return res.data;
}

// ─── Admin (danger zone) ──────────────────────────────────────────────────────

export async function purgeViolations(): Promise<{ deleted: number }> {
  const res = await http.post("/v1/admin/purge/violations");
  return res.data;
}

export async function purgeSessions(): Promise<{ sessions_deleted: number; events_deleted: number }> {
  const res = await http.post("/v1/admin/purge/sessions");
  return res.data;
}

export async function purgeBaselines(): Promise<{ deleted: number }> {
  const res = await http.post("/v1/admin/purge/baselines");
  return res.data;
}

// ─── Analytics ────────────────────────────────────────────────────────────────

export interface TrendPoint {
  date: string;
  posture_score: number;
  blocked: number;
  alerted: number;
  sessions: number;
}

export interface TrendsResponse {
  days: number;
  points: TrendPoint[];
}

export interface AgentComparison {
  agent_id: string;
  session_count: number;
  llm_call_count: number;
  tool_call_count: number;
  error_count: number;
  total_tokens: number;
  anomaly_count: number;
  pii_event_count: number;
  avg_latency_ms: number | null;
  last_seen: string | null;
  block_count: number;
  alert_count: number;
  total_violations: number;
}

export async function getTrends(days = 30): Promise<TrendsResponse> {
  const res = await http.get("/v1/analytics/trends", { params: { days } });
  return res.data;
}

export async function getAgentComparison(): Promise<{ agents: AgentComparison[]; total: number }> {
  const res = await http.get("/v1/analytics/agents");
  return res.data;
}

// ─── Org Settings ─────────────────────────────────────────────────────────────

export interface OrgSettings {
  slack_webhook_url?: string;
  smtp_host?: string;
  smtp_port?: string;
  smtp_user?: string;
  smtp_password?: string;
  smtp_from?: string;
  smtp_to?: string;
  pagerduty_routing_key?: string;
  alert_webhook_url?: string;
  alert_webhook_secret?: string;
  alert_min_severity?: string;
  alert_cooldown_s?: string;
  splunk_hec_url?: string;
  splunk_hec_token?: string;
  elastic_url?: string;
  elastic_api_key?: string;
}

export async function getOrgSettings(): Promise<{ settings: OrgSettings }> {
  const res = await http.get<{ settings: OrgSettings }>("/v1/settings");
  return res.data;
}

export async function updateOrgSettings(
  updates: Partial<OrgSettings>
): Promise<{ status: string; updated_count: number }> {
  const res = await http.patch<{ status: string; updated_count: number }>(
    "/v1/settings",
    { updates }
  );
  return res.data;
}

export async function testNotification(
  channel: string
): Promise<{ ok: boolean; message: string }> {
  const res = await http.post<{ ok: boolean; message: string }>(
    "/v1/settings/test",
    { channel }
  );
  return res.data;
}

// ─── HITL Approvals ───────────────────────────────────────────────────────────

export interface Approval {
  id: string;
  org_id: string;
  session_id: string;
  agent_id: string;
  tool_name: string;
  tool_args: Record<string, unknown>;
  trigger: string;
  status: "pending" | "approved" | "denied" | "expired";
  decided_by?: string;
  decision_note?: string;
  created_at: string;
  decided_at?: string;
  expires_at: string;
}

export interface ListApprovalsParams {
  status?: string;
  agent_id?: string;
  limit?: number;
}

export async function listApprovals(
  params: ListApprovalsParams = {}
): Promise<{ approvals: Approval[]; count: number }> {
  const res = await http.get<{ approvals: Approval[]; count: number }>(
    "/v1/approvals",
    { params }
  );
  return res.data;
}

export async function approveRequest(
  id: string,
  note?: string
): Promise<{ status: string; id: string }> {
  const res = await http.post(`/v1/approvals/${id}/approve`, { decision_note: note });
  return res.data;
}

export async function denyRequest(
  id: string,
  note?: string
): Promise<{ status: string; id: string }> {
  const res = await http.post(`/v1/approvals/${id}/deny`, { decision_note: note });
  return res.data;
}

// ─── Security Posture (Phase 12) ──────────────────────────────────────────────

export interface PostureSummary {
  agent_id: string;
  observation_days: number;
  total_sessions: number;
  tool_calls: number;
  unique_tools: number;
  sessions_with_network_calls: number;
  anomaly_count: number;
  has_active_manifest: boolean;
  active_manifest_id: string | null;
  active_manifest_issued: string | null;
  active_manifest_expires: string | null;
}

export interface ObservedToolPosture {
  tool_name: string;
  call_count: number;
  session_count: number;
  first_seen: string | null;
  last_seen: string | null;
  observed_arg_keys: string[];
}

export interface NetworkDestination {
  destination: string;
  seen_count: number;
  last_seen: string | null;
}

export interface PostureAnomaly {
  id: string;
  rule_name: string;
  action: string;
  reason: string;
  session_id: string;
  created_at: string;
}

export interface ManifestComplianceStats {
  agent_id: string;
  window_hours: number;
  allowed_tool_calls: number;
  blocked_tool_calls: number;
}

export async function getPostureSummary(agentId: string, days = 30): Promise<PostureSummary> {
  const res = await http.get(`/v1/security-posture/${agentId}`, { params: { days } });
  return res.data;
}

export async function getObservedToolsPosture(
  agentId: string,
  days = 30
): Promise<{ tools: ObservedToolPosture[]; count: number }> {
  const res = await http.get(`/v1/security-posture/${agentId}/tools`, { params: { days } });
  return res.data;
}

export async function getPostureAnomalies(
  agentId: string,
  days = 30
): Promise<{ anomalies: PostureAnomaly[]; count: number }> {
  const res = await http.get(`/v1/security-posture/${agentId}/anomalies`, { params: { days } });
  return res.data;
}

export async function getObservedNetwork(
  agentId: string,
  days = 30
): Promise<{ destinations: NetworkDestination[]; count: number }> {
  const res = await http.get(`/v1/security-posture/${agentId}/network`, { params: { days } });
  return res.data;
}

export async function generateManifest(
  agentId: string,
  opts: {
    validity_days?: number;
    notes?: string;
    permitted_tools?: Record<string, unknown>[] | null;
    permitted_network_destinations?: string[] | null;
  } = {}
): Promise<Record<string, unknown>> {
  const body: Record<string, unknown> = {
    validity_days: opts.validity_days ?? 30,
    notes: opts.notes ?? "",
  };
  if (opts.permitted_tools !== undefined) body.permitted_tools = opts.permitted_tools;
  if (opts.permitted_network_destinations !== undefined)
    body.permitted_network_destinations = opts.permitted_network_destinations;
  const res = await http.post(`/v1/manifests/${agentId}/generate`, body);
  return res.data;
}

export async function getActiveManifest(agentId: string): Promise<Record<string, unknown> | null> {
  try {
    const res = await http.get(`/v1/manifests/${agentId}/active`);
    return res.data;
  } catch (e: unknown) {
    if ((e as { response?: { status?: number } })?.response?.status === 404) return null;
    throw e;
  }
}

export async function revokeManifest(agentId: string): Promise<{ revoked: string[]; count: number }> {
  const res = await http.delete(`/v1/manifests/${agentId}`);
  return res.data;
}

export async function getManifestCompliance(
  agentId: string,
  hours = 24
): Promise<ManifestComplianceStats> {
  const res = await http.get(`/v1/manifests/${agentId}/compliance`, { params: { hours } });
  return res.data;
}

export function getSecurityPackageUrl(agentId: string): string {
  return `${BASE_URL}/v1/manifests/${agentId}/package?api_key=${API_KEY}`;
}

// ---------------------------------------------------------------------------
// Security Features
// ---------------------------------------------------------------------------

export interface SecurityFeature {
  key: string;
  group: string;
  name: string;
  description: string;
  why?: string;
  type: "boolean" | "enum" | "number";
  default: string;
  value: string;
  is_overridden: boolean;
  is_agent_override?: boolean;
  is_org_override?: boolean;
  options?: string[];
  unit?: string;
  requires?: string;
}

export async function getSecurityFeatures(): Promise<{ features: SecurityFeature[]; org_id: string }> {
  const res = await http.get("/v1/security-features");
  return res.data;
}

export async function updateSecurityFeatures(
  updates: Record<string, string>,
): Promise<{ updated: string[]; count: number }> {
  const res = await http.patch("/v1/security-features", { updates });
  return res.data;
}

export async function getAgentSecurityFeatures(
  agentId: string,
): Promise<{ features: SecurityFeature[]; agent_id: string; org_id: string }> {
  const res = await http.get(`/v1/agents/${agentId}/security-features`);
  return res.data;
}

export async function updateAgentSecurityFeatures(
  agentId: string,
  updates: Record<string, string>,
): Promise<{ updated: string[]; count: number }> {
  const res = await http.patch(`/v1/agents/${agentId}/security-features`, { updates });
  return res.data;
}

// ---------------------------------------------------------------------------
// Egress Rules
// ---------------------------------------------------------------------------

export interface EgressRule {
  destination: string;
  notes: string;
  created_at: string | null;
  created_by: string;
}

export async function listEgressRules(): Promise<{ rules: EgressRule[]; count: number }> {
  const res = await http.get("/v1/egress-rules");
  return res.data;
}

export async function addEgressRule(
  destination: string,
  notes = "",
): Promise<{ destination: string; added: boolean }> {
  const res = await http.post("/v1/egress-rules", { destination, notes });
  return res.data;
}

export async function deleteEgressRule(
  destination: string,
): Promise<{ destination: string; deleted: boolean }> {
  const res = await http.delete(`/v1/egress-rules/${encodeURIComponent(destination)}`);
  return res.data;
}

// ─── Data Flow Graph (PDG) ────────────────────────────────────────────────

export interface PDGEdge {
  tool_name: string;
  source_tool: string | null;
  source_fragment: string;
  sink_tool: string;
  sink_arg: string | null;
  hop_count: number;
}

export interface SessionPDGResponse {
  session_id: string;
  total_edges: number;
  total_untrusted_sources: number;
  untrusted_sources: string[];
  tools_with_violations: string[];
  edges: PDGEdge[];
}

export async function getSessionPDG(sessionId: string): Promise<SessionPDGResponse> {
  const res = await http.get(`/v1/sessions/${sessionId}/pdg`);
  return res.data;
}

// ─── Reasoning Traces ──────────────────────────────────────────────────────

export interface ThinkingBlock {
  thinking: string;
  signature?: string;
}

export interface ReasoningEntry {
  event_id: string;
  run_id: string;
  model: string;
  // Nanosecond timestamps (~1.77×10¹⁸) exceed Number.MAX_SAFE_INTEGER (2^53).
  // Precision loss ≈ 256 ns — acceptable for display only.
  // Use sequence_number or event_id for ordering guarantees.
  timestamp_ns: number;
  thinking_blocks: ThinkingBlock[];
  block_count: number;
}

export interface SessionReasoningResponse {
  session_id: string;
  reasoning: ReasoningEntry[];
  total_calls_with_reasoning: number;
}

export async function getSessionReasoning(
  sessionId: string,
): Promise<SessionReasoningResponse> {
  const res = await http.get(`/v1/sessions/${sessionId}/reasoning`);
  return res.data;
}
