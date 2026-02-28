/**
 * D3 force-directed graph for agent topology visualisation.
 *
 * Nodes are scaled by llm_call_count and coloured by risk_level.
 * Edges are directed arrows; thickness scales logarithmically with call_count.
 * Supports drag, zoom/pan, and click-to-select.
 */
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { TopologyEdge, TopologyNode } from "./topology.types";
import styles from "./TopologyGraph.module.css";

// ─── Constants ────────────────────────────────────────────────────────────────

const RISK_FILL: Record<string, string> = {
  low: "#22c55e",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

const RISK_LABEL_COLOR: Record<string, string> = {
  low: "#15803d",
  medium: "#854d0e",
  high: "#9a3412",
  critical: "#991b1b",
};

// ─── Types ────────────────────────────────────────────────────────────────────

type SimNode = d3.SimulationNodeDatum & TopologyNode;

type SimLink = d3.SimulationLinkDatum<SimNode> & {
  call_count: number;
  avg_latency_ms: number | null;
};

// ─── Component ────────────────────────────────────────────────────────────────

interface TopologyGraphProps {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string | null) => void;
}

export function TopologyGraph({
  nodes,
  edges,
  selectedAgentId,
  onSelectAgent,
}: TopologyGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    const svgEl = svgRef.current;
    if (!svgEl) return;

    const width = svgEl.clientWidth || 800;
    const height = svgEl.clientHeight || 560;

    const svg = d3.select(svgEl);
    svg.selectAll("*").remove();

    // ── Arrow marker ──────────────────────────────────────────────────────────
    svg
      .append("defs")
      .append("marker")
      .attr("id", "abb-arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 7)
      .attr("markerHeight", 7)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#94a3b8");

    const g = svg.append("g");

    // ── Zoom / pan ────────────────────────────────────────────────────────────
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.15, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // ── Simulation data ───────────────────────────────────────────────────────
    const simNodes: SimNode[] = nodes.map((n) => ({ ...n }));
    const indexByAgent = new Map(simNodes.map((n, i) => [n.agent_id, i]));

    const simLinks: SimLink[] = edges
      .filter(
        (e) => indexByAgent.has(e.source) && indexByAgent.has(e.target)
      )
      .map((e) => ({
        source: indexByAgent.get(e.source)!,
        target: indexByAgent.get(e.target)!,
        call_count: e.call_count,
        avg_latency_ms: e.avg_latency_ms,
      }));

    // ── Force simulation ──────────────────────────────────────────────────────
    const maxCalls = Math.max(...simNodes.map((n) => n.llm_call_count), 1);
    const rScale = d3.scaleSqrt().domain([0, maxCalls]).range([12, 38]);

    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3.forceLink<SimNode, SimLink>(simLinks).distance(130).strength(0.4)
      )
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force(
        "collision",
        d3.forceCollide<SimNode>().radius((d) => rScale(d.llm_call_count) + 8)
      );

    // ── Edges ─────────────────────────────────────────────────────────────────
    const linkG = g.append("g").attr("class", "abb-links");
    const linkEl = linkG
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", "#94a3b8")
      .attr("stroke-opacity", 0.55)
      .attr(
        "stroke-width",
        (d) => Math.min(6, 1 + Math.log10(d.call_count + 1) * 2.5)
      )
      .attr("marker-end", "url(#abb-arrow)");

    // ── Nodes ─────────────────────────────────────────────────────────────────
    const nodeG = g.append("g").attr("class", "abb-nodes");

    const drag = d3
      .drag<SVGGElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    const nodeEl = nodeG
      .selectAll<SVGGElement, SimNode>("g.abb-node")
      .data(simNodes)
      .join("g")
      .attr("class", "abb-node")
      .attr("cursor", "pointer")
      .call(drag)
      .on("click", (event, d) => {
        event.stopPropagation();
        onSelectAgent(selectedAgentId === d.agent_id ? null : d.agent_id);
      });

    // Background circle (selection ring)
    nodeEl
      .append("circle")
      .attr("r", (d) => rScale(d.llm_call_count) + 4)
      .attr("fill", "transparent")
      .attr("stroke", (d) =>
        selectedAgentId === d.agent_id ? "#1e40af" : "transparent"
      )
      .attr("stroke-width", 2.5)
      .attr("stroke-dasharray", "4 2");

    // Main node circle
    nodeEl
      .append("circle")
      .attr("r", (d) => rScale(d.llm_call_count))
      .attr("fill", (d) => RISK_FILL[d.risk_level] ?? "#6b7280")
      .attr("fill-opacity", 0.82)
      .attr("stroke", "white")
      .attr("stroke-width", 2);

    // Risk score text inside node
    nodeEl
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em")
      .attr("font-size", "10px")
      .attr("font-weight", "700")
      .attr("fill", "white")
      .attr("pointer-events", "none")
      .text((d) => d.risk_score.toFixed(2));

    // Agent ID label below node
    nodeEl
      .append("text")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => rScale(d.llm_call_count) + 15)
      .attr("font-size", "11px")
      .attr("font-weight", "500")
      .attr("fill", (d) => RISK_LABEL_COLOR[d.risk_level] ?? "#374151")
      .attr("pointer-events", "none")
      .text((d) =>
        d.agent_id.length > 18 ? d.agent_id.slice(0, 16) + "\u2026" : d.agent_id
      );

    // ── Click outside to deselect ─────────────────────────────────────────────
    svg.on("click", () => onSelectAgent(null));

    // ── Simulation tick ───────────────────────────────────────────────────────
    simulation.on("tick", () => {
      linkEl
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);

      nodeEl.attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, selectedAgentId, onSelectAgent]);

  const isEmpty = nodes.length === 0;

  return (
    <div className={styles.graphWrapper}>
      <svg ref={svgRef} className={styles.graphSvg} aria-label="Agent topology graph" />

      {isEmpty && (
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon}>&#9702;</span>
          <span>No agents found. Route LLM traffic through the proxy to see the graph.</span>
        </div>
      )}

      {!isEmpty && (
        <>
          <div className={styles.legend}>
            {(["low", "medium", "high", "critical"] as const).map((level) => (
              <div key={level} className={styles.legendItem}>
                <span
                  className={styles.legendDot}
                  style={{ background: RISK_FILL[level] }}
                />
                <span>{level}</span>
              </div>
            ))}
          </div>
          <span className={styles.hint}>Drag nodes • Scroll to zoom • Click to select</span>
        </>
      )}
    </div>
  );
}
