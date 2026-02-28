/**
 * Type definitions for the SIEM Export feature.
 *
 * Mirrors the Pydantic models in backend/app/api/v1/export.py.
 */

/** Active export target/destination. */
export type ExportTarget = "jsonlines" | "splunk" | "elasticsearch" | "audit";

/** Common filter fields shared by all export targets. */
export interface ExportFilters {
  session_id?: string;
  agent_id?: string;
  from_ts_ns?: number;
  to_ts_ns?: number;
  limit?: number;
}

/** Request body for the Splunk HEC push endpoint. */
export interface SplunkPushRequest extends ExportFilters {
  hec_url: string;
  hec_token: string;
  index?: string;
  source?: string;
}

/** Request body for the Elasticsearch Bulk API push endpoint. */
export interface ElasticPushRequest extends ExportFilters {
  es_url: string;
  api_key?: string;
  index?: string;
}

/** Summary returned after a push operation. */
export interface PushResult {
  sent: number;
  batches: number;
  errors: string[];
}

/** UI state for a single export panel. */
export interface ExportPanelState {
  isLoading: boolean;
  result: PushResult | null;
  error: string | null;
}
