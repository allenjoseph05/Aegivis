import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  TrendingUp, TrendingDown, Minus, RefreshCw, BarChart2,
} from "lucide-react";
import {
  getTrends, getAgentComparison,
  getMetricsOverview, listAgentMetrics, listModelMetrics,
  getViolationsStats,
} from "../../api/client";
import type { TrendPoint } from "../../api/client";
import type { AgentMetrics, ModelMetrics } from "../../types/events";

// ─── Shared helpers ────────────────────────────────────────────────────────────

function fmt(n: number | null | undefined, decimals = 0): string {
  if (n == null) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return `${ms.toLocaleString(undefined, { maximumFractionDigits: 0 })} ms`;
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric", month: "short", day: "numeric",
  });
}

// ─── Trends tab ────────────────────────────────────────────────────────────────

function TrendChart({ points }: { points: TrendPoint[] }) {
  if (points.length === 0) return null;

  // Single data point — can't draw a line; render a single dot + label instead
  if (points.length === 1) {
    return (
      <div className="flex items-center justify-center h-[180px] text-sm text-gray-400">
        Only 1 data point — more data needed to show a trend.
      </div>
    );
  }

  const W = 800;
  const H = 180;
  const PAD = { top: 16, right: 16, bottom: 32, left: 36 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;

  const scores = points.map((p) => p.posture_score);
  const minScore = Math.max(0, Math.min(...scores) - 5);
  const maxScore = Math.min(100, Math.max(...scores) + 5);
  const scoreRange = maxScore - minScore || 1;

  const x = (i: number) => PAD.left + (i / (points.length - 1)) * chartW;
  const y = (score: number) =>
    PAD.top + chartH - ((score - minScore) / scoreRange) * chartH;

  const polyline = points.map((p, i) => `${x(i)},${y(p.posture_score)}`).join(" ");
  const areaPath =
    `M ${x(0)},${y(points[0].posture_score)} ` +
    points.slice(1).map((p, i) => `L ${x(i + 1)},${y(p.posture_score)}`).join(" ") +
    ` L ${x(points.length - 1)},${PAD.top + chartH} L ${x(0)},${PAD.top + chartH} Z`;

  const yLabels = [0, 25, 50, 75, 100].filter(
    (v) => v >= minScore - 5 && v <= maxScore + 5
  );
  const xLabelStep = Math.ceil(points.length / 6);
  const xLabels = points.filter(
    (_, i) => i % xLabelStep === 0 || i === points.length - 1
  );

  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const lineStroke =
    avgScore >= 80 ? "#22c55e" :
    avgScore >= 60 ? "#3b82f6" :
    avgScore >= 40 ? "#f59e0b" : "#ef4444";

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
      <defs>
        <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={lineStroke} stopOpacity="0.15" />
          <stop offset="100%" stopColor={lineStroke} stopOpacity="0.01" />
        </linearGradient>
      </defs>
      {yLabels.map((v) => (
        <g key={v}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(v)} y2={y(v)}
            stroke="#e5e7eb" strokeWidth="1" />
          <text x={PAD.left - 6} y={y(v) + 4} textAnchor="end" fontSize="10" fill="#9ca3af">
            {v}
          </text>
        </g>
      ))}
      <path d={areaPath} fill="url(#area-grad)" />
      <polyline points={polyline} fill="none" stroke={lineStroke}
        strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
      {points.map((p, i) =>
        p.blocked > 0 ? (
          <circle key={i} cx={x(i)} cy={y(p.posture_score)} r="3"
            fill="#ef4444" stroke="white" strokeWidth="1.5" />
        ) : null
      )}
      {xLabels.map((p) => {
        const i = points.indexOf(p);
        const parts = p.date.split("-");
        return (
          <text key={p.date} x={x(i)} y={H - 6} textAnchor="middle" fontSize="10" fill="#9ca3af">
            {`${parts[1]}/${parts[2]}`}
          </text>
        );
      })}
    </svg>
  );
}

function PostureTrendCard({ days, setDays }: { days: number; setDays: (d: number) => void }) {
  const { data, isFetching } = useQuery({
    queryKey: ["trends", days],
    queryFn: () => getTrends(days),
    staleTime: 60_000,
  });

  const points = data?.points ?? [];
  const latest = points[points.length - 1];
  const weekAgo = points[points.length - 8];
  const trend = latest && weekAgo ? latest.posture_score - weekAgo.posture_score : null;

  const TrendIcon = trend === null ? Minus : trend > 0 ? TrendingUp : TrendingDown;
  const trendColor =
    trend === null ? "text-gray-400" :
    trend > 0 ? "text-green-600" : "text-red-500";

  const totalBlocked = points.reduce((s, p) => s + p.blocked, 0);
  const totalAlerted = points.reduce((s, p) => s + p.alerted, 0);
  const totalSessions = points.reduce((s, p) => s + p.sessions, 0);

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div>
            <h2 className="font-semibold text-gray-900">Security Posture Trend</h2>
            <p className="text-xs text-gray-400 mt-0.5">0–100 score · red dots = BLOCK events</p>
          </div>
          {latest && (
            <div className="flex items-center gap-1.5 ml-4">
              <span className="text-2xl font-bold text-gray-900">{latest.posture_score}</span>
              <span className={`flex items-center gap-0.5 text-sm font-medium ${trendColor}`}>
                <TrendIcon size={14} />
                {trend !== null ? `${trend > 0 ? "+" : ""}${trend}` : "—"}
              </span>
              <span className="text-xs text-gray-400">vs 7d ago</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isFetching && <RefreshCw size={13} className="animate-spin text-gray-400" />}
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            {[7, 14, 30, 60].map((d) => (
              <button key={d} onClick={() => setDays(d)}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  days === d ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
                }`}>
                {d}d
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="px-5 py-3">
        {points.length === 0 ? (
          <div className="h-44 flex items-center justify-center text-gray-400 text-sm">
            No data yet. Traffic appears here once agents start using the proxy.
          </div>
        ) : (
          <TrendChart points={points} />
        )}
      </div>
      {points.length > 0 && (
        <div className="px-5 py-3 border-t border-gray-50 grid grid-cols-3 gap-4">
          {[
            { label: `Blocked (${days}d)`, value: totalBlocked, color: "text-red-600" },
            { label: `Alerted (${days}d)`, value: totalAlerted, color: "text-amber-600" },
            { label: `Sessions (${days}d)`, value: totalSessions, color: "text-gray-700" },
          ].map(({ label, value, color }) => (
            <div key={label} className="text-center">
              <div className={`text-xl font-bold ${color}`}>{value.toLocaleString()}</div>
              <div className="text-xs text-gray-400">{label}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Violations over time ──────────────────────────────────────────────────────

interface PivotedBucket {
  bucket: string;
  block: number;
  alert: number;
}

function ViolationsTrendCard({ days }: { days: number }) {
  const [interval, setInterval] = useState<"hour" | "day">("day");

  const { data, isFetching } = useQuery({
    queryKey: ["violations-stats", interval, days],
    queryFn: () => getViolationsStats(interval, days),
    staleTime: 60_000,
  });

  // Pivot flat buckets into {bucket, block, alert}
  const pivotMap = new Map<string, PivotedBucket>();
  for (const b of data?.buckets ?? []) {
    if (!pivotMap.has(b.bucket)) pivotMap.set(b.bucket, { bucket: b.bucket, block: 0, alert: 0 });
    const entry = pivotMap.get(b.bucket)!;
    if (b.action === "BLOCK") entry.block = b.count;
    else entry.alert = b.count;
  }
  const buckets = [...pivotMap.values()].sort((a, b) => a.bucket.localeCompare(b.bucket));

  const totalBlock = buckets.reduce((s, b) => s + b.block, 0);
  const totalAlert = buckets.reduce((s, b) => s + b.alert, 0);
  const maxCount = Math.max(1, ...buckets.map((b) => b.block + b.alert));

  const W = 800;
  const H = 140;
  const PAD = { top: 8, right: 8, bottom: 24, left: 36 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;
  const barSlotW = chartW / Math.max(buckets.length, 1);
  const barW = Math.max(2, barSlotW - 2);
  const xPos = (i: number) => PAD.left + i * barSlotW + (barSlotW - barW) / 2;
  const yScale = (n: number) => (n / maxCount) * chartH;

  const xStep = Math.max(1, Math.ceil(buckets.length / 6));
  const xLabelBuckets = buckets.filter((_, i) => i % xStep === 0 || i === buckets.length - 1);

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
        <div>
          <h2 className="font-semibold text-gray-900">Violations Over Time</h2>
          <p className="text-xs text-gray-400 mt-0.5">
            <span className="text-red-500 font-medium">{totalBlock.toLocaleString()} blocked</span>
            {" · "}
            <span className="text-amber-500 font-medium">{totalAlert.toLocaleString()} alerted</span>
            {" — last "}{days}d
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isFetching && <RefreshCw size={13} className="animate-spin text-gray-400" />}
          <div className="flex gap-1 bg-gray-100 rounded-lg p-1">
            {(["hour", "day"] as const).map((iv) => (
              <button key={iv} onClick={() => setInterval(iv)}
                className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                  interval === iv ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
                }`}>
                {iv === "hour" ? "Hourly" : "Daily"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="px-5 py-3">
        {buckets.length === 0 ? (
          <div className="h-36 flex items-center justify-center text-gray-400 text-sm">
            No violations in this period.
          </div>
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: H }}>
            {/* Y-axis gridlines */}
            {[0, Math.round(maxCount / 2), maxCount].map((v) => (
              <g key={v}>
                <line
                  x1={PAD.left} x2={W - PAD.right}
                  y1={PAD.top + chartH - yScale(v)} y2={PAD.top + chartH - yScale(v)}
                  stroke="#e5e7eb" strokeWidth="1"
                />
                <text x={PAD.left - 4} y={PAD.top + chartH - yScale(v) + 4}
                  textAnchor="end" fontSize="10" fill="#9ca3af">{v}</text>
              </g>
            ))}

            {/* Stacked bars: ALERT (amber, bottom) + BLOCK (red, top) */}
            {buckets.map((b, i) => {
              const alertH = yScale(b.alert);
              const blockH = yScale(b.block);
              const baseY = PAD.top + chartH;
              return (
                <g key={b.bucket}>
                  {b.alert > 0 && (
                    <rect x={xPos(i)} y={baseY - alertH} width={barW} height={alertH}
                      fill="#f59e0b" fillOpacity="0.8" rx="1" />
                  )}
                  {b.block > 0 && (
                    <rect x={xPos(i)} y={baseY - alertH - blockH} width={barW} height={blockH}
                      fill="#ef4444" fillOpacity="0.85" rx="1" />
                  )}
                </g>
              );
            })}

            {/* X-axis labels */}
            {xLabelBuckets.map((b) => {
              const i = buckets.indexOf(b);
              const label = interval === "hour"
                ? b.bucket.slice(11, 16)       // HH:MM
                : b.bucket.slice(5, 10);        // MM-DD
              return (
                <text key={b.bucket} x={xPos(i) + barW / 2} y={H - 4}
                  textAnchor="middle" fontSize="9" fill="#9ca3af">{label}</text>
              );
            })}
          </svg>
        )}
      </div>

      {/* Legend */}
      <div className="px-5 pb-3 flex gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-400 inline-block" /> BLOCK
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-amber-400 inline-block" /> ALERT
        </span>
      </div>
    </div>
  );
}

function RiskBadge({ violations }: { violations: number }) {
  if (violations === 0)
    return <span className="text-xs text-green-600 font-medium">Clean</span>;
  if (violations < 5)
    return <span className="text-xs text-amber-600 font-medium">{violations} violations</span>;
  return <span className="text-xs text-red-600 font-semibold">{violations} violations</span>;
}

function AgentComparisonTable() {
  const { data, isFetching } = useQuery({
    queryKey: ["agent-comparison"],
    queryFn: getAgentComparison,
    staleTime: 60_000,
  });

  const agents = data?.agents ?? [];

  return (
    <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center justify-between">
        <div>
          <h2 className="font-semibold text-gray-900">Agent Comparison</h2>
          <p className="text-xs text-gray-400 mt-0.5">Ranked by total violations</p>
        </div>
        {isFetching && <RefreshCw size={13} className="animate-spin text-gray-400" />}
      </div>
      {agents.length === 0 ? (
        <div className="px-5 py-12 text-center text-gray-400 text-sm">No agent data yet.</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-100">
                {["Agent", "Sessions", "LLM Calls", "Tool Calls", "Tokens", "Avg Latency", "Anomalies", "Violations"].map((h) => (
                  <th key={h} className="px-4 py-2.5 text-left text-xs font-medium text-gray-500 uppercase tracking-wide whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {agents.map((a) => (
                <tr key={a.agent_id} className="hover:bg-gray-50">
                  <td className="px-4 py-3"><span className="font-mono text-xs text-gray-800">{a.agent_id}</span></td>
                  <td className="px-4 py-3 text-xs text-gray-600">{a.session_count.toLocaleString()}</td>
                  <td className="px-4 py-3 text-xs text-gray-600">{a.llm_call_count.toLocaleString()}</td>
                  <td className="px-4 py-3 text-xs text-gray-600">{a.tool_call_count.toLocaleString()}</td>
                  <td className="px-4 py-3 text-xs text-gray-600">
                    {a.total_tokens >= 1_000_000 ? `${(a.total_tokens / 1_000_000).toFixed(1)}M`
                      : a.total_tokens >= 1_000 ? `${(a.total_tokens / 1_000).toFixed(1)}K`
                      : a.total_tokens}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-600">
                    {a.avg_latency_ms != null ? `${Math.round(a.avg_latency_ms)}ms` : "—"}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-600">{a.anomaly_count}</td>
                  <td className="px-4 py-3"><RiskBadge violations={a.total_violations} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ─── Performance tab (previously Metrics page) ────────────────────────────────

function injectionBadge(score: number) {
  if (score >= 0.7)
    return <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-700">{score.toFixed(2)} HIGH</span>;
  if (score >= 0.4)
    return <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-700">{score.toFixed(2)} MED</span>;
  return <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-50 text-green-700">{score.toFixed(2)}</span>;
}

function OverviewCard({ label, value, sub, accent }: {
  label: string; value: string | number; sub?: string;
  accent?: "red" | "amber" | "green" | "blue";
}) {
  const accentClass =
    accent === "red" ? "border-l-red-500 bg-red-50" :
    accent === "amber" ? "border-l-amber-500 bg-amber-50" :
    accent === "green" ? "border-l-green-500 bg-green-50" :
    "border-l-blue-500 bg-blue-50";
  return (
    <div className={`border-l-4 rounded-lg p-4 ${accentClass}`}>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-sm font-medium text-gray-700 mt-0.5">{label}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function AgentsTable({ agents }: { agents: AgentMetrics[] }) {
  if (agents.length === 0)
    return (
      <div className="text-center py-12 text-gray-400 text-sm">
        No agents observed yet. Route LLM traffic through the proxy to see metrics here.
      </div>
    );
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            {["Agent ID", "Sessions", "LLM Calls", "Tool Calls", "Avg Latency", "Tokens", "Errors", "PII Events", "Max Inj. Score", "Anomalies", "Last Active"].map((h) => (
              <th key={h} className={`px-4 py-2 font-medium text-gray-600 ${h === "Agent ID" ? "text-left" : "text-right"}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {agents.map((a) => (
            <tr key={a.agent_id} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="px-4 py-2 font-mono text-xs text-gray-800 max-w-[12rem] truncate">{a.agent_id}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.session_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.llm_call_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.tool_call_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmtMs(a.avg_latency_ms)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(a.total_tokens)}</td>
              <td className="px-4 py-2 text-right">
                {a.error_count > 0 ? <span className="text-red-600 font-medium">{a.error_count}</span> : <span className="text-gray-400">0</span>}
              </td>
              <td className="px-4 py-2 text-right">
                {a.pii_event_count > 0 ? <span className="text-amber-600 font-medium">{a.pii_event_count}</span> : <span className="text-gray-400">0</span>}
              </td>
              <td className="px-4 py-2 text-right">{injectionBadge(a.injection_score_max)}</td>
              <td className="px-4 py-2 text-right">
                {a.anomaly_count > 0 ? <span className="text-orange-600 font-medium">{a.anomaly_count}</span> : <span className="text-gray-400">0</span>}
              </td>
              <td className="px-4 py-2 text-right text-gray-500 text-xs">{fmtDate(a.last_seen)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelsTable({ models }: { models: ModelMetrics[] }) {
  if (models.length === 0)
    return <div className="text-center py-8 text-gray-400 text-sm">No models recorded yet.</div>;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 bg-gray-50">
            {["Model", "Provider", "Sessions", "Calls", "Avg Latency", "Total Tokens", "Errors", "First Used"].map((h, i) => (
              <th key={h} className={`px-4 py-2 font-medium text-gray-600 ${i < 2 ? "text-left" : "text-right"}`}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model} className="border-b border-gray-100 hover:bg-gray-50">
              <td className="px-4 py-2 font-mono text-xs font-medium text-gray-800">{m.model}</td>
              <td className="px-4 py-2">
                <span className="px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-600">{m.provider}</span>
              </td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(m.session_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700 font-medium">{fmt(m.call_count)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmtMs(m.avg_latency_ms)}</td>
              <td className="px-4 py-2 text-right text-gray-700">{fmt(m.total_tokens)}</td>
              <td className="px-4 py-2 text-right">
                {m.error_count > 0 ? <span className="text-red-600 font-medium">{m.error_count}</span> : <span className="text-gray-400">0</span>}
              </td>
              <td className="px-4 py-2 text-right text-gray-500 text-xs">{fmtDate(m.first_used)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PerformanceTab() {
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
    <div className="space-y-8">
      {/* Overview cards */}
      {overviewQ.isLoading ? (
        <div className="text-sm text-gray-400">Loading…</div>
      ) : ov ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
            <OverviewCard label="Sessions" value={fmt(ov.session_count)} accent="blue" />
            <OverviewCard label="Agents" value={fmt(ov.agent_count)} accent="blue" />
            <OverviewCard label="LLM Calls" value={fmt(ov.llm_call_count)} sub={`${fmt(ov.tool_call_count)} tool calls`} accent="blue" />
            <OverviewCard label="Total Tokens"
              value={ov.total_tokens >= 1_000_000 ? `${(ov.total_tokens / 1_000_000).toFixed(1)}M` : fmt(ov.total_tokens)}
              accent="blue" />
            <OverviewCard label="Violations" value={fmt(ov.total_violations)} sub={`${fmt(ov.blocked_count)} blocked`}
              accent={ov.blocked_count > 0 ? "red" : "green"} />
            <OverviewCard label="Anomalies" value={fmt(ov.total_anomalies)} sub={`${fmt(ov.high_severity_anomalies)} high/critical`}
              accent={ov.high_severity_anomalies > 0 ? "amber" : "green"} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <OverviewCard label="Avg Latency" value={fmtMs(ov.avg_latency_ms)} accent="blue" />
            <OverviewCard label="PII Exposures" value={fmt(ov.pii_event_count)} accent={ov.pii_event_count > 0 ? "amber" : "green"} />
            <OverviewCard label="Errors" value={fmt(ov.error_count)} accent={ov.error_count > 0 ? "red" : "green"} />
            <OverviewCard label="Alerts" value={fmt(ov.alert_count)} accent={ov.alert_count > 0 ? "amber" : "green"} />
          </div>
        </>
      ) : null}

      {/* Agent performance table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Agent Performance</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {agentsQ.data?.total ?? 0} agent{agentsQ.data?.total !== 1 ? "s" : ""} · sorted by most recent activity
            </p>
          </div>
          {agentsQ.isFetching && <span className="text-xs text-gray-400">Refreshing…</span>}
        </div>
        <div>
          {agentsQ.isLoading ? (
            <div className="py-8 text-center text-sm text-gray-400">Loading…</div>
          ) : (
            <AgentsTable agents={agents} />
          )}
        </div>
      </div>

      {/* Model usage table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200">
        <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">Model Usage</h2>
            <p className="text-xs text-gray-500 mt-0.5">
              {modelsQ.data?.total ?? 0} model{modelsQ.data?.total !== 1 ? "s" : ""} · sorted by call count
            </p>
          </div>
          {modelsQ.isFetching && <span className="text-xs text-gray-400">Refreshing…</span>}
        </div>
        <div>
          {modelsQ.isLoading ? (
            <div className="py-8 text-center text-sm text-gray-400">Loading…</div>
          ) : (
            <ModelsTable models={models} />
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Page ──────────────────────────────────────────────────────────────────────

type Tab = "trends" | "performance";

export function AnalyticsPage() {
  const [days, setDays] = useState(30);
  const [tab, setTab] = useState<Tab>("trends");

  return (
    <div className="max-w-screen-xl mx-auto px-6 py-8 space-y-6">
      <div className="flex items-center gap-3">
        <BarChart2 size={22} className="text-gray-700" />
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          <p className="text-sm text-gray-500">Security posture trends and performance metrics.</p>
        </div>
      </div>

      {/* Tab bar */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1">
          {(["trends", "performance"] as Tab[]).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
                tab === t
                  ? "border-blue-600 text-blue-600"
                  : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
              }`}>
              {t === "trends" ? "Trends" : "Performance"}
            </button>
          ))}
        </div>
      </div>

      {tab === "trends" && (
        <div className="space-y-6">
          <PostureTrendCard days={days} setDays={setDays} />
          <ViolationsTrendCard days={days} />
          <AgentComparisonTable />
        </div>
      )}

      {tab === "performance" && <PerformanceTab />}
    </div>
  );
}
