/**
 * Unit tests for the Agent Topology page.
 *
 * The D3 force-directed graph is rendered in jsdom which does not support
 * SVG layout, so TopologyGraph is mocked to a simple placeholder.
 *
 * Run: npm run test:run
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { TopologyPage } from "./TopologyPage";

// ─── Mocks ─────────────────────────────────────────────────────────────────────

vi.mock("./TopologyGraph", () => ({
  TopologyGraph: ({
    nodes,
    edges,
    onSelectAgent,
  }: {
    nodes: { agent_id: string }[];
    edges: unknown[];
    onSelectAgent: (id: string | null) => void;
  }) => (
    <div data-testid="topology-graph">
      <span data-testid="node-count">{nodes.length}</span>
      <span data-testid="edge-count">{edges.length}</span>
      {nodes.map((n) => (
        <button
          key={n.agent_id}
          data-testid={`node-${n.agent_id}`}
          onClick={() => onSelectAgent(n.agent_id)}
        >
          {n.agent_id}
        </button>
      ))}
    </div>
  ),
}));

vi.mock("../../api/client", () => ({
  getTopology: vi.fn(),
}));

import { getTopology } from "../../api/client";

// ─── Test helpers ──────────────────────────────────────────────────────────────

const makeNode = (agent_id: string, risk_level = "low" as const) => ({
  agent_id,
  session_count: 5,
  llm_call_count: 20,
  tool_call_count: 10,
  error_count: 0,
  violation_count: 0,
  anomaly_count: 0,
  pii_event_count: 0,
  injection_score_max: 0.0,
  risk_score: 0.0,
  risk_level,
  first_seen: "2025-01-01T00:00:00+00:00",
  last_seen: "2025-06-01T00:00:00+00:00",
  providers: ["openai"],
  models: ["gpt-4o"],
});

const makeEdge = (source: string, target: string) => ({
  source,
  target,
  call_count: 5,
  avg_latency_ms: 120,
  first_seen: null,
  last_seen: null,
});

function makeGraph(nodes = [] as ReturnType<typeof makeNode>[], edges = [] as ReturnType<typeof makeEdge>[]) {
  return {
    nodes,
    edges,
    total_nodes: nodes.length,
    total_edges: edges.length,
    computed_at: "2025-06-01T12:00:00+00:00",
  };
}

function renderPage() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={qc}>
        <TopologyPage />
      </QueryClientProvider>
    </MemoryRouter>
  );
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe("TopologyPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders page title", async () => {
    vi.mocked(getTopology).mockResolvedValueOnce(makeGraph());
    renderPage();
    expect(screen.getByText("Agent Topology")).toBeInTheDocument();
  });

  it("shows loading state initially", () => {
    vi.mocked(getTopology).mockReturnValue(new Promise(() => {})); // never resolves
    renderPage();
    expect(screen.getByText(/loading topology/i)).toBeInTheDocument();
  });

  it("shows error state when fetch fails", async () => {
    vi.mocked(getTopology).mockRejectedValueOnce(new Error("Network error"));
    renderPage();
    // Wait for error state to appear
    await vi.waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeNull()
    );
  });

  it("displays node and edge counts from response", async () => {
    const graph = makeGraph(
      [makeNode("agent-a"), makeNode("agent-b")],
      [makeEdge("agent-a", "agent-b")]
    );
    vi.mocked(getTopology).mockResolvedValueOnce(graph);
    renderPage();
    await vi.waitFor(() =>
      expect(screen.getByTestId("topology-graph")).toBeInTheDocument()
    );
    expect(screen.getByTestId("node-count").textContent).toBe("2");
    expect(screen.getByTestId("edge-count").textContent).toBe("1");
  });

  it("shows AgentCard when a node is clicked", async () => {
    const graph = makeGraph([makeNode("orchestrator")]);
    vi.mocked(getTopology).mockResolvedValueOnce(graph);
    renderPage();
    await vi.waitFor(() =>
      expect(screen.getByTestId("node-orchestrator")).toBeInTheDocument()
    );
    fireEvent.click(screen.getByTestId("node-orchestrator"));
    // AgentCard renders with role="complementary" and aria-label for the agent
    expect(
      screen.getByRole("complementary", { name: /agent details: orchestrator/i })
    ).toBeInTheDocument();
  });

  it("hides AgentCard after close button clicked", async () => {
    const graph = makeGraph([makeNode("agent-x")]);
    vi.mocked(getTopology).mockResolvedValueOnce(graph);
    renderPage();
    await vi.waitFor(() =>
      expect(screen.getByTestId("node-agent-x")).toBeInTheDocument()
    );
    fireEvent.click(screen.getByTestId("node-agent-x"));
    const closeBtn = screen.getByRole("button", { name: /close agent details/i });
    fireEvent.click(closeBtn);
    // AgentCard should no longer show
    expect(screen.queryByRole("complementary")).toBeNull();
  });

  it("toolbar checkbox toggles include_isolated filter", async () => {
    vi.mocked(getTopology).mockResolvedValue(makeGraph());
    renderPage();
    await vi.waitFor(() =>
      expect(screen.getByLabelText("Include isolated agents")).toBeInTheDocument()
    );
    const checkbox = screen.getByLabelText("Include isolated agents") as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(false);
  });

  it("toolbar number input changes min_edge_calls", async () => {
    vi.mocked(getTopology).mockResolvedValue(makeGraph());
    renderPage();
    await vi.waitFor(() =>
      expect(screen.getByLabelText("Minimum edge call count")).toBeInTheDocument()
    );
    const input = screen.getByLabelText("Minimum edge call count") as HTMLInputElement;
    expect(input.value).toBe("1");
    fireEvent.change(input, { target: { value: "5" } });
    expect(input.value).toBe("5");
  });
});
