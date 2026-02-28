/**
 * Agent Topology Page.
 *
 * Shows a force-directed graph of agent-to-agent relationships.
 * Clicking a node opens a detail panel with aggregated security + perf stats.
 */
import { useState } from "react";
import { useTopology } from "./topology.hooks";
import { TopologyGraph } from "./TopologyGraph";
import { AgentCard } from "./AgentCard";
import type { TopologyFilters } from "./topology.types";
import styles from "./TopologyPage.module.css";

// ─── Main page ─────────────────────────────────────────────────────────────────

export function TopologyPage() {
  const [filters, setFilters] = useState<TopologyFilters>({
    include_isolated: true,
    min_edge_calls: 1,
  });
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  const { data, isLoading, isError, isFetching, dataUpdatedAt } = useTopology(filters);

  const nodes = data?.nodes ?? [];
  const edges = data?.edges ?? [];

  const selectedNode =
    selectedAgentId != null
      ? nodes.find((n) => n.agent_id === selectedAgentId) ?? null
      : null;

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>Agent Topology</h1>
        <p className={styles.subtitle}>
          Directed graph of inter-agent call relationships with risk scoring.
          Nodes are sized by LLM call count; colour indicates risk level.
        </p>
      </div>

      {/* Toolbar */}
      <div className={styles.toolbar}>
        <div className={styles.toolbarGroup}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={filters.include_isolated}
              onChange={(e) =>
                setFilters((f) => ({ ...f, include_isolated: e.target.checked }))
              }
              aria-label="Include isolated agents"
            />
            Include isolated agents
          </label>
        </div>

        <div className={styles.toolbarGroup}>
          <span className={styles.toolbarLabel}>Min edge calls:</span>
          <input
            type="number"
            className={styles.filterInput}
            min={1}
            max={100}
            value={filters.min_edge_calls}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10);
              if (v >= 1) setFilters((f) => ({ ...f, min_edge_calls: v }));
            }}
            aria-label="Minimum edge call count"
          />
        </div>

        <div className={styles.statsBar}>
          <span className={styles.statBadge}>
            Nodes: <strong>{data?.total_nodes ?? 0}</strong>
          </span>
          <span className={styles.statBadge}>
            Edges: <strong>{data?.total_edges ?? 0}</strong>
          </span>
          {isFetching && (
            <span className={styles.statBadge} style={{ color: "#6366f1" }}>
              Refreshing...
            </span>
          )}
        </div>
      </div>

      {/* Error state */}
      {isError && (
        <div className={styles.errorBox} role="alert">
          Failed to load topology. Ensure the backend is running and reachable.
        </div>
      )}

      {/* Loading skeleton */}
      {isLoading && !isError && (
        <div
          style={{
            height: 560,
            background: "#f8fafc",
            borderRadius: 8,
            border: "1px solid #e2e8f0",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#94a3b8",
            fontSize: 14,
          }}
        >
          Loading topology graph...
        </div>
      )}

      {/* Graph + side panel */}
      {!isLoading && !isError && (
        <div className={styles.graphRow}>
          <div className={styles.graphCol}>
            <TopologyGraph
              nodes={nodes}
              edges={edges}
              selectedAgentId={selectedAgentId}
              onSelectAgent={setSelectedAgentId}
            />
            {dataUpdatedAt > 0 && (
              <div className={styles.refreshHint}>
                Last updated:{" "}
                {new Date(dataUpdatedAt).toLocaleTimeString()}
                {" "}
                &middot; Auto-refreshes every 30s
              </div>
            )}
          </div>

          {selectedNode && (
            <div className={styles.sidePanel}>
              <AgentCard
                node={selectedNode}
                onClose={() => setSelectedAgentId(null)}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
