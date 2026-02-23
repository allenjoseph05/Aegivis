/**
 * Typed API client for AgentBlackBox backend.
 * All requests include the API key header.
 */
import axios from "axios";
import type { Anomaly, AuditEvent, Session } from "../types/events";

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
  org_id?: string;
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

export async function getSession(
  sessionId: string,
  orgId = "default-org"
): Promise<Session> {
  const res = await http.get(`/v1/sessions/${sessionId}`, {
    params: { org_id: orgId },
  });
  return res.data;
}

// ─── Events ──────────────────────────────────────────────────────────────────

export interface GetEventsParams {
  org_id?: string;
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

export async function verifySession(
  sessionId: string,
  orgId = "default-org"
): Promise<VerifyResult> {
  const res = await http.get(`/v1/sessions/${sessionId}/verify`, {
    params: { org_id: orgId },
  });
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

export async function getForensicReplay(
  sessionId: string,
  orgId = "default-org"
): Promise<ForensicReplayResponse> {
  const res = await http.get(`/v1/sessions/${sessionId}/replay`, {
    params: { org_id: orgId },
  });
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
  regulation: Regulation,
  orgId = "default-org"
): Promise<ComplianceReportResponse> {
  const res = await http.post("/v1/reports/generate", {
    session_id: sessionId,
    org_id: orgId,
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
