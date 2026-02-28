/**
 * Metrics page — per-agent and per-model performance statistics.
 *
 * Shows a platform-wide overview, an agent performance table, and a model
 * usage table.  All data comes from GET /v1/metrics/* on the backend.
 */
import { useQuery } from "@tanstack/react-query";
import {
  getMetricsOverview,
  listAgentMetrics,
  listModelMetrics,
} from "../api/client";
import type { AgentMetrics, ModelMetrics } from "../types/events";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function fmt(n: number | null | undefined, decimals = 0): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  return `${ms.toLocaleString(undefined, { maximumFractionDigits: 0 })} ms`;
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function injectionBadge(score: number) {
  if (score >= 0.7)
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">
        {score.toFixed(2)} HIGH
      </span>
    );
  if (score >= 0.4)
    return (
      <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700">
        {score.toFixed(2)} MED
      </span>
    );
  return (
    <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-50 text-green-700">
      {score.toFixed(2)}
    </span>
  );
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function OverviewCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string | number;
  sub?: string;
  accent?: "red" | "amber" | "green" | "blue";
}) {
  const accentClass =
    accent === "red"
      ? "border-l-red-500 bg-red-50"
      : accent === "amber"
      ? "border-l-amber-500 bg-amber-50"
      : accent === "green"
      ? "border-l-green-500 bg-green-50"
      : "border-l-blue-500 bg-blue-50";

  return (
    <div className={`border-l-4 rounded-lg p-4 ${accentClass}`}>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-sm font-medium text-gray-700 mt-0.5">{label}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function AgentsTable({ agents }: { agents: AgentMetrics[] }) {
  if (agents.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400 text-sm">
        No agents have been observed yet. Route LLM traffic through the proxy to
        see metrics here.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            <th className="text-left px-4 py-2 font-medium text-gray-600">Agent ID</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Sessions</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">LLM Calls</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Tool Calls</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Avg Latency</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Tokens</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Errors</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">PII Events</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Max Inj. Score</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Anomalies</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Last Active</th>
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.agent_id} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="px-4 py-2 font-mono text-xs text-gray-800 max-w-[12rem] truncate">
                {a.agent_id}
              </td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.session_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.llm_call_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.tool_call_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmtMs(a.avg_latency_ms)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.total_tokens)}</td>
              <td className="px-4 py-2 text-right">
                {a.error_count > 0 ? (
                  <span className="text-red-600 font-medium">{a.error_count}</span>
                ) : (
                  <span className="text-gray-400">0</span>
                )}
              </td>
              <td className="px-4 py-2 text-right">
                {a.pii_event_count > 0 ? (
                  <span className="text-amber-600 font-medium">{a.pii_event_count}</span>
                ) : (
                  <span className="text-gray-400">0</span>
                )}
              </td>
              <td className="px-4 py-2 text-right">{injectionBadge(a.injection_score_max)}</td>
              <td className="px-4 py-2 text-right">
                {a.anomaly_count > 0 ? (
                  <span className="text-orange-600 font-medium">{a.anomaly_count}</span>
                ) : (
                  <span className="text-gray-400">0</span>
                )}
              </td>
              <td className="px-4 py-2 text-right text-gray-500 text-xs">
                {fmtDate(a.last_seen)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelsTable({ models }: { models: ModelMetrics[] }) {
  if (models.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400 text-sm">
        No models recorded yet.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            <th className="text-left px-4 py-2 font-medium text-gray-600">Model</th>
            <th className="text-left px-4 py-2 font-medium text-gray-600">Provider</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Sessions</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Calls</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Avg Latency</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Total Tokens</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">Errors</th>
            <th className="text-right px-4 py-2 font-medium text-gray-600">First Used</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="px-4 py-2 font-mono text-xs font-medium text-gray-800">
                {m.model}
              </td>
              <td className="px-4 py-2">
                <span className="px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600">
                  {m.provider}
                </span>
              </td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(m.session_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700 font-medium">{fmt(m.call_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmtMs(m.avg_latency_ms)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(m.total_tokens)}</td>
              <td className="px-4 py-2 text-right">
                {m.error_count > 0 ? (
                  <span className="text-red-600 font-medium">{m.error_count}</span>
                ) : (
                  <span className="text-gray-400">0</span>
                )}
              </td>
              <td className="px-4 py-2 text-right text-gray-500 text-xs">
                {fmtDate(m.first_used)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export function Metrics() {
  const overviewQ = useQuery({
    queryKey: ["metrics-overview"],
    queryFn: getMetricsOverview,
    refetchInterval: 30_000,
  });

  const agentsQ = useQuery({
    queryKey: ["metrics-agents"],
    queryFn: () => listAgentMetrics({ limit: 100 }),
    refetchInterval: 30_000,
  });

  const modelsQ = useQuery({
    queryKey: ["metrics-models"],
    queryFn: listModelMetrics,
    refetchInterval: 30_000,
  });

  const ov = overviewQ.data;
  const agents = agentsQ.data?.agents ?? [];
  const models = modelsQ.data?.models ?? [];

  return (
    <div className="max-w-screen-xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Metrics</h1>
        <p className="text-gray-500 text-sm mt-1">
          Performance and security statistics across all agents and models.
        </p>
      </div>

      {/* Overview cards */}
      {overviewQ.isLoading ? (
        <div className="text-sm text-gray-400">Loading overview...</div>
      ) : overviewQ.isError ? (
        <div className="text-sm text-red-500">Failed to load overview metrics.</div>
      ) : ov ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
          <OverviewCard
            label="Sessions"
            value={fmt(ov.session_count)}
            accent="blue"
          />
          <OverviewCard
            label="Agents"
            value={fmt(ov.agent_count)}
            accent="blue"
          />
          <OverviewCard
            label="LLM Calls"
            value={fmt(ov.llm_call_count)}
            sub={`${fmt(ov.tool_call_count)} tool calls`}
            accent="blue"
          />
          <OverviewCard
            label="Total Tokens"
            value={ov.total_tokens >= 1_000_000
              ? `${(ov.total_tokens / 1_000_000).toFixed(1)}M`
              : fmt(ov.total_tokens)}
            accent="blue"
          />
          <OverviewCard
            label="Violations"
            value={fmt(ov.total_violations)}
            sub={`${fmt(ov.blocked_count)} blocked`}
            accent={ov.blocked_count > 0 ? "red" : "green"}
          />
          <OverviewCard
            label="Anomalies"
            value={fmt(ov.total_anomalies)}
            sub={`${fmt(ov.high_severity_anomalies)} high/critical`}
            accent={ov.high_severity_anomalies > 0 ? "amber" : "green"}
          />
        </div>
      ) : null}

      {/* Secondary metrics row */}
      {ov && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <OverviewCard
            label="Avg LLM Latency"
            value={fmtMs(ov.avg_latency_ms)}
            accent="blue"
          />
          <OverviewCard
            label="PII Exposures"
            value={fmt(ov.pii_event_count)}
            accent={ov.pii_event_count > 0 ? "amber" : "green"}
          />
          <OverviewCard
            label="Errors"
            value={fmt(ov.error_count)}
            accent={ov.error_count > 0 ? "red" : "green"}
          />
          <OverviewCard
            label="Alerts"
            value={fmt(ov.alert_count)}
            accent={ov.alert_count > 0 ? "amber" : "green"}
          />
        </div>
      )}

      {/* Agent metrics table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Agent Performance</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {agentsQ.data?.total ?? 0} agent{agentsQ.data?.total !== 1 ? "s" : ""} — sorted by most recent activity
            </p>
          </div>
          {agentsQ.isFetching && (
            <span className="text-xs text-gray-400">Refreshing...</span>
          )}
        </div>
        <div className="p-0">
          {agentsQ.isLoading ? (
            <div className="py-8 text-center text-sm text-gray-400">Loading...</div>
          ) : agentsQ.isError ? (
            <div className="py-8 text-center text-sm text-red-500">
              Failed to load agent metrics.
            </div>
          ) : (
            <AgentsTable agents={agents} />
          )}
        </div>
      </div>

      {/* Model metrics table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Model Usage</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {modelsQ.data?.total ?? 0} model{modelsQ.data?.total !== 1 ? "s" : ""} — sorted by call count
            </p>
          </div>
          {modelsQ.isFetching && (
            <span className="text-xs text-gray-400">Refreshing...</span>
          )}
        </div>
        <div className="p-0">
          {modelsQ.isLoading ? (
            <div className="py-8 text-center text-sm text-gray-400">Loading...</div>
          ) : modelsQ.isError ? (
            <div className="py-8 text-center text-sm text-red-500">
              Failed to load model metrics.
            </div>
          ) : (
            <ModelsTable models={models} />
          )}
        </div>
      </div>
    </div>
  );
}
