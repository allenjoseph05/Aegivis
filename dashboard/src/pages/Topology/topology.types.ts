/**
 * Type definitions for the Agent Topology feature.
 *
 * These types mirror the backend service dataclasses in
 * backend/app/services/topology.py.
 */

/** Risk classification level based on composite security score. */
export type RiskLevel = "low" | "medium" | "high" | "critical";

/**
 * A single agent node in the topology graph.
 * Aggregated stats from audit_events, policy_violations, and agent_anomalies.
 */
export interface TopologyNode {
  agent_id: string;
  session_count: number;
  llm_call_count: number;
  tool_call_count: number;
  error_count: number;
  violation_count: number;
  anomaly_count: number;
  pii_event_count: number;
  /** Maximum injection score seen across all LLM calls. Range [0, 1]. */
  injection_score_max: number;
  /** Composite risk score. Range [0, 1]. */
  risk_score: number;
  risk_level: RiskLevel;
  first_seen: string | null;
  last_seen: string | null;
  providers: string[];
  models: string[];
}

/**
 * A directed edge from one agent to another.
 * Source → Target means Source spawned a call that ended up in Target.
 */
export interface TopologyEdge {
  source: string;
  target: string;
  call_count: number;
  avg_latency_ms: number | null;
  first_seen: string | null;
  last_seen: string | null;
}

/** The full computed topology graph returned by GET /v1/topology. */
export interface TopologyGraph {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  total_nodes: number;
  total_edges: number;
  computed_at: string;
}

/** Query parameters for the topology endpoint. */
export interface TopologyFilters {
  include_isolated: boolean;
  min_edge_calls: number;
}
