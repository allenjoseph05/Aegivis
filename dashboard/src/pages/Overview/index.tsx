import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  Shield,
  AlertTriangle,
  XCircle,
  CheckCircle,
  Clock,
  Activity,
  ChevronRight,
  Zap,
  Users,
  TrendingUp,
  Bell,
  type LucideIcon,
} from "lucide-react";
import {
  getMetricsOverview,
  listViolations,
  listAgents,
  listBaselines,
  listAgentMetrics,
} from "../../api/client";
import type { Violation, Agent, BaselineSummary } from "../../api/client";
import type { AgentMetrics } from "../../types/events";

// ─── Posture score ─────────────────────────────────────────────────────────────

function computePostureScore(
  blockedCount: number,
  totalViolations: number,
  pendingTools: number,
  agentCount: number
): number {
  let score = 100;
  if (blockedCount > 0) score -= Math.min(30, blockedCount * 5);
  if (totalViolations > 10) score -= Math.min(15, Math.floor((totalViolations - 10) / 5));
  if (pendingTools > 0) score -= Math.min(20, pendingTools * 3);
  if (agentCount === 0) score -= 15;
  return Math.max(0, score);
}

function PostureScoreRing({ score }: { score: number }) {
  const r = 52;
  const circ = 2 * Math.PI * r;
  const dash = (score / 100) * circ;
  const color =
    score >= 80 ? "#22c55e" : score >= 60 ? "#f59e0b" : "#ef4444";
  const label =
    score >= 80 ? "Good" : score >= 60 ? "Fair" : "At Risk";

  return (
    <div className="flex flex-col items-center justify-center">
      <svg width="136" height="136" className="-rotate-90">
        <circle cx="68" cy="68" r={r} fill="none" stroke="#e5e7eb" strokeWidth="12" />
        <circle
          cx="68"
          cy="68"
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
      </svg>
      <div className="absolute flex flex-col items-center pointer-events-none">
        <span className="text-3xl font-bold text-gray-900">{score}</span>
        <span className="text-xs font-medium" style={{ color }}>{label}</span>
      </div>
    </div>
  );
}

// ─── Agent status ──────────────────────────────────────────────────────────────

function agentStatus(
  agent: Agent,
  metrics: AgentMetrics | undefined,
  recentBlocks: number
): { label: string; color: string; dot: string } {
  if (!metrics?.last_seen) return { label: "No traffic", color: "text-gray-400", dot: "bg-gray-300" };
  const lastSeen = new Date(metrics.last_seen).getTime();
  const minutesAgo = (Date.now() - lastSeen) / 60_000;
  if (minutesAgo > 60) return { label: "Idle", color: "text-gray-400", dot: "bg-gray-300" };
  if (recentBlocks > 0) return { label: "Blocked", color: "text-red-600", dot: "bg-red-500" };
  if ((metrics?.anomaly_count ?? 0) > 0) return { label: "Alerting", color: "text-amber-600", dot: "bg-amber-400" };
  return { label: "Healthy", color: "text-green-600", dot: "bg-green-500" };
}

function timeAgo(ns: number): string {
  const ms = ns / 1_000_000;
  const diff = Date.now() - ms;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

// ─── Severity badge ────────────────────────────────────────────────────────────

function ActionBadge({ action }: { action: string }) {
  if (action === "BLOCK")
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-red-100 text-red-700">
        <XCircle size={11} /> BLOCK
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 text-amber-700">
      <AlertTriangle size={11} /> ALERT
    </span>
  );
}

// ─── Stat card ─────────────────────────────────────────────────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  accent,
}: {
  icon: LucideIcon;
  label: string;
  value: string | number;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 flex items-center gap-4 shadow-sm">
      <div className={`p-2.5 rounded-lg ${accent ?? "bg-blue-50"}`}>
        <Icon size={20} className={accent ? "text-current" : "text-blue-600"} />
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-900">{value}</p>
        <p className="text-sm text-gray-500">{label}</p>
        {sub && <p className="text-xs text-gray-400">{sub}</p>}
      </div>
    </div>
  );
}

// ─── Agent health card ─────────────────────────────────────────────────────────

function AgentCard({
  agent,
  metrics,
  baseline,
  recentBlocks,
}: {
  agent: Agent;
  metrics: AgentMetrics | undefined;
  baseline: BaselineSummary | undefined;
  recentBlocks: number;
}) {
  const status = agentStatus(agent, metrics, recentBlocks);
  const enforcementMode = baseline?.enforcement_active ? "Enforcement" : "Audit mode";
  const pendingTools = baseline?.pending ?? 0;

  return (
    <Link
      to={`/agents`}
      className="block bg-white rounded-xl border border-gray-200 p-4 shadow-sm hover:shadow-md hover:border-gray-300 transition-all"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0">
          <p className="font-semibold text-gray-900 truncate">{agent.name || agent.agent_id}</p>
          <p className="text-xs text-gray-400 font-mono truncate">{agent.agent_id}</p>
        </div>
        <div className="flex items-center gap-1.5 ml-2 flex-shrink-0">
          <span className={`w-2 h-2 rounded-full ${status.dot}`} />
          <span className={`text-xs font-medium ${status.color}`}>{status.label}</span>
        </div>
      </div>

      <div className="space-y-1.5 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Sessions</span>
          <span className="font-medium text-gray-700">{metrics?.session_count ?? 0}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Anomalies</span>
          <span className={`font-medium ${(metrics?.anomaly_count ?? 0) > 0 ? "text-amber-600" : "text-gray-700"}`}>
            {metrics?.anomaly_count ?? 0}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-500">Tool baseline</span>
          <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${
            baseline?.enforcement_active
              ? "bg-green-50 text-green-700"
              : "bg-gray-100 text-gray-500"
          }`}>
            {enforcementMode}
          </span>
        </div>
        {pendingTools > 0 && (
          <div className="flex items-center gap-1.5 text-amber-600 text-xs mt-1">
            <Bell size={11} />
            {pendingTools} tool{pendingTools > 1 ? "s" : ""} pending review
          </div>
        )}
      </div>
    </Link>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export function OverviewPage() {
  const { data: metrics } = useQuery({
    queryKey: ["metrics-overview"],
    queryFn: getMetricsOverview,
    refetchInterval: 15_000,
  });

  const { data: violationsData } = useQuery({
    queryKey: ["violations-overview"],
    queryFn: () => listViolations({ limit: 20 }),
    refetchInterval: 15_000,
  });

  const { data: agentsData } = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    refetchInterval: 30_000,
  });

  const { data: baselinesData } = useQuery({
    queryKey: ["baselines"],
    queryFn: listBaselines,
    refetchInterval: 30_000,
  });

  const { data: agentMetricsData } = useQuery({
    queryKey: ["agent-metrics-overview"],
    queryFn: () => listAgentMetrics({ limit: 50 }),
    refetchInterval: 30_000,
  });

  const agents = agentsData?.agents ?? [];
  const violations = violationsData?.violations ?? [];
  const baselines = baselinesData?.baselines ?? [];
  const agentMetricsList = agentMetricsData?.agents ?? [];

  // Build lookup maps
  const metricsById = Object.fromEntries(agentMetricsList.map((m) => [m.agent_id, m]));
  const baselineById = Object.fromEntries(baselines.map((b) => [b.agent_id, b]));

  // Per-agent recent block count (from last 20 violations)
  const blocksByAgent: Record<string, number> = {};
  for (const v of violations) {
    if (v.action === "BLOCK") blocksByAgent[v.agent_id] = (blocksByAgent[v.agent_id] ?? 0) + 1;
  }

  // Posture score
  const totalPending = baselines.reduce((s, b) => s + b.pending, 0);
  const postureScore = computePostureScore(
    metrics?.blocked_count ?? 0,
    metrics?.total_violations ?? 0,
    totalPending,
    agents.length
  );

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Overview</h1>
        <p className="text-sm text-gray-500 mt-1">Security posture across all agents</p>
      </div>

      {/* Top row: posture + stats */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Posture score card */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm p-6 flex flex-col items-center justify-center gap-4">
          <p className="text-sm font-semibold text-gray-600 uppercase tracking-wide">Security Posture</p>
          <div className="relative flex items-center justify-center">
            <PostureScoreRing score={postureScore} />
          </div>
          <div className="text-center space-y-1">
            {totalPending > 0 && (
              <Link
                to="/baselines"
                className="text-xs text-amber-600 hover:text-amber-700 flex items-center justify-center gap-1"
              >
                <Bell size={11} />
                {totalPending} tool{totalPending > 1 ? "s" : ""} pending review
                <ChevronRight size={11} />
              </Link>
            )}
            {(metrics?.blocked_count ?? 0) > 0 && (
              <p className="text-xs text-red-500">
                {metrics?.blocked_count} blocked request{(metrics?.blocked_count ?? 0) > 1 ? "s" : ""}
              </p>
            )}
            {totalPending === 0 && (metrics?.blocked_count ?? 0) === 0 && (
              <p className="text-xs text-green-600">No active threats detected</p>
            )}
          </div>
        </div>

        {/* Quick stats */}
        <div className="lg:col-span-3 grid grid-cols-2 gap-4">
          <StatCard
            icon={Users}
            label="Active agents"
            value={agents.length}
            accent="bg-blue-50"
          />
          <StatCard
            icon={Activity}
            label="Total sessions"
            value={metrics?.session_count ?? 0}
            accent="bg-purple-50"
          />
          <StatCard
            icon={XCircle}
            label="Blocked requests"
            value={metrics?.blocked_count ?? 0}
            accent={metrics?.blocked_count ? "bg-red-50" : "bg-gray-50"}
          />
          <StatCard
            icon={AlertTriangle}
            label="Alerts today"
            value={metrics?.alert_count ?? 0}
            accent={metrics?.alert_count ? "bg-amber-50" : "bg-gray-50"}
          />
        </div>
      </div>

      {/* Agent health cards */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-base font-semibold text-gray-800">Agent Health</h2>
          <Link to="/agents" className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1">
            Manage agents <ChevronRight size={14} />
          </Link>
        </div>

        {agents.length === 0 ? (
          <div className="bg-white rounded-xl border border-dashed border-gray-300 p-10 text-center">
            <Shield size={32} className="mx-auto text-gray-300 mb-3" />
            <p className="text-gray-500 font-medium">No agents registered yet</p>
            <p className="text-sm text-gray-400 mt-1">
              Register your first agent to start monitoring
            </p>
            <Link
              to="/agents"
              className="mt-4 inline-flex items-center gap-1.5 px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Zap size={14} /> Register agent
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {agents.map((agent) => (
              <AgentCard
                key={agent.agent_id}
                agent={agent}
                metrics={metricsById[agent.agent_id]}
                baseline={baselineById[agent.agent_id]}
                recentBlocks={blocksByAgent[agent.agent_id] ?? 0}
              />
            ))}
          </div>
        )}
      </div>

      {/* Bottom row: incident feed + quick actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Incident feed */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm">
          <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
            <h2 className="font-semibold text-gray-800">Recent Incidents</h2>
            <Link to="/violations" className="text-sm text-blue-600 hover:text-blue-700 flex items-center gap-1">
              View all <ChevronRight size={14} />
            </Link>
          </div>

          {violations.length === 0 ? (
            <div className="px-5 py-10 text-center">
              <CheckCircle size={28} className="mx-auto text-green-400 mb-2" />
              <p className="text-sm text-gray-500">No violations recorded</p>
            </div>
          ) : (
            <ul className="divide-y divide-gray-50">
              {violations.slice(0, 10).map((v) => (
                <IncidentRow key={v.id} violation={v} />
              ))}
            </ul>
          )}
        </div>

        {/* Quick actions */}
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
          <div className="px-5 py-4 border-b border-gray-100">
            <h2 className="font-semibold text-gray-800">Quick Actions</h2>
          </div>
          <div className="p-4 space-y-2">
            <QuickAction
              icon={Shield}
              label="Review pending tools"
              sub={totalPending > 0 ? `${totalPending} awaiting review` : "All clear"}
              to="/baselines"
              badge={totalPending > 0 ? totalPending : undefined}
              urgent={totalPending > 0}
            />
            <QuickAction
              icon={Activity}
              label="View all events"
              sub="Full session audit trail"
              to="/events"
            />
            <QuickAction
              icon={TrendingUp}
              label="Security stack"
              sub="Configure layers & thresholds"
              to="/security"
            />
            <QuickAction
              icon={Clock}
              label="Export audit report"
              sub="Compliance frameworks"
              to="/export"
            />
            <QuickAction
              icon={Users}
              label="Agent registry"
              sub="Manage your agents"
              to="/agents"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

function IncidentRow({ violation }: { violation: Violation }) {
  return (
    <li className="px-5 py-3 flex items-start gap-3 hover:bg-gray-50 transition-colors">
      <div className="flex-shrink-0 mt-0.5">
        <ActionBadge action={violation.action} />
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-sm font-medium text-gray-800 truncate">{violation.rule_name}</p>
        <p className="text-xs text-gray-400 truncate">
          {violation.agent_id} · {violation.event_type}
        </p>
      </div>
      <span className="flex-shrink-0 text-xs text-gray-400">{timeAgo(violation.timestamp_ns)}</span>
    </li>
  );
}

function QuickAction({
  icon: Icon,
  label,
  sub,
  to,
  badge,
  urgent,
}: {
  icon: LucideIcon;
  label: string;
  sub: string;
  to: string;
  badge?: number;
  urgent?: boolean;
}) {
  return (
    <Link
      to={to}
      className="flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-gray-50 transition-colors group"
    >
      <div className={`p-1.5 rounded-lg ${urgent ? "bg-amber-50" : "bg-gray-100"} group-hover:bg-gray-200 transition-colors`}>
        <Icon size={16} className={urgent ? "text-amber-600" : "text-gray-500"} />
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-700">{label}</p>
        <p className="text-xs text-gray-400">{sub}</p>
      </div>
      {badge != null && (
        <span className="flex-shrink-0 text-xs font-semibold bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full">
          {badge}
        </span>
      )}
      <ChevronRight size={14} className="flex-shrink-0 text-gray-300 group-hover:text-gray-400" />
    </Link>
  );
}
