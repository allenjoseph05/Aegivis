/**
 * Detail panel shown when a node is selected in the topology graph.
 *
 * Displays all aggregated metrics and security stats for a single agent.
 */
import type { TopologyNode } from "./topology.types";
import styles from "./AgentCard.module.css";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n: number, decimals = 0): string {
  return n.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

const RISK_BADGE_STYLE: Record<
  string,
  { background: string; color: string }
> = {
  low:      { background: "#dcfce7", color: "#15803d" },
  medium:   { background: "#fef9c3", color: "#854d0e" },
  high:     { background: "#ffedd5", color: "#9a3412" },
  critical: { background: "#fee2e2", color: "#991b1b" },
};

// ─── Component ────────────────────────────────────────────────────────────────

interface AgentCardProps {
  node: TopologyNode;
  onClose: () => void;
}

export function AgentCard({ node, onClose }: AgentCardProps) {
  const badgeStyle = RISK_BADGE_STYLE[node.risk_level] ?? {
    background: "#f1f5f9",
    color: "#475569",
  };

  return (
    <div className={styles.card} role="complementary" aria-label={`Agent details: ${node.agent_id}`}>
      {/* Header */}
      <div className={styles.cardHeader}>
        <div className={styles.agentId}>{node.agent_id}</div>
        <button
          className={styles.closeBtn}
          onClick={onClose}
          aria-label="Close agent details"
        >
          &times;
        </button>
      </div>

      {/* Risk badge */}
      <div>
        <span
          className={styles.riskBadge}
          style={badgeStyle}
          aria-label={`Risk level: ${node.risk_level}`}
        >
          {node.risk_level} &mdash; {node.risk_score.toFixed(3)}
        </span>
      </div>

      {/* Stats grid */}
      <div className={styles.statsGrid}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Sessions</span>
          <span className={styles.statValue}>{fmt(node.session_count)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>LLM Calls</span>
          <span className={styles.statValue}>{fmt(node.llm_call_count)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Tool Calls</span>
          <span className={styles.statValue}>{fmt(node.tool_call_count)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Errors</span>
          <span className={styles.statValue} style={{ color: node.error_count > 0 ? "#dc2626" : undefined }}>
            {fmt(node.error_count)}
          </span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Violations</span>
          <span className={styles.statValue} style={{ color: node.violation_count > 0 ? "#f97316" : undefined }}>
            {fmt(node.violation_count)}
          </span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Anomalies</span>
          <span className={styles.statValue} style={{ color: node.anomaly_count > 0 ? "#eab308" : undefined }}>
            {fmt(node.anomaly_count)}
          </span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>PII Events</span>
          <span className={styles.statValue}>{fmt(node.pii_event_count)}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Max Inj. Score</span>
          <span
            className={styles.statValue}
            style={{
              color:
                node.injection_score_max >= 0.7
                  ? "#dc2626"
                  : node.injection_score_max >= 0.4
                  ? "#f97316"
                  : undefined,
            }}
          >
            {node.injection_score_max.toFixed(3)}
          </span>
        </div>
      </div>

      <div className={styles.divider} />

      {/* Providers */}
      {node.providers.length > 0 && (
        <div>
          <span className={styles.statLabel}>Providers</span>
          <div className={styles.tagRow}>
            {node.providers.map((p) => (
              <span key={p} className={styles.tag}>{p}</span>
            ))}
          </div>
        </div>
      )}

      {/* Models */}
      {node.models.length > 0 && (
        <div style={{ marginTop: "8px" }}>
          <span className={styles.statLabel}>Models</span>
          <div className={styles.tagRow}>
            {node.models.map((m) => (
              <span key={m} className={styles.tag}>{m}</span>
            ))}
          </div>
        </div>
      )}

      {/* Date range */}
      <div className={styles.dateRow}>
        <span>First seen: {fmtDate(node.first_seen)}</span>
        <span>Last seen: {fmtDate(node.last_seen)}</span>
      </div>
    </div>
  );
}
