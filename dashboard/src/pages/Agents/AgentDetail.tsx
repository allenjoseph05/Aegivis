import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ChevronLeft,
  Activity,
  Shield,
  Wrench,
  Gauge,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  RefreshCw,
  Check,
  X,
} from "lucide-react";
import {
  listAgentMetrics,
  listViolations,
  listSessions,
  listAgentTools,
  approveTool,
  denyTool,
  approveAllTools,
  listPolicyRules,
  listAgents,
  getAgentSecurityFeatures,
  updateAgentSecurityFeatures,
} from "../../api/client";
import type { Agent, AgentTool, SecurityFeature } from "../../api/client";

// ─── Shared helpers ────────────────────────────────────────────────────────────

function timeAgo(ns: number): string {
  const diff = Date.now() - ns / 1_000_000;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(ns / 1_000_000).toLocaleDateString();
}

function fmt(n: number | null | undefined, digits = 0): string {
  if (n == null) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

// ─── Tab definitions ───────────────────────────────────────────────────────────

type Tab = "health" | "baselines" | "policies" | "security" | "ratelimits";

const TABS: { id: Tab; label: string }[] = [
  { id: "health", label: "Health" },
  { id: "baselines", label: "Tool Baseline" },
  { id: "policies", label: "Policies" },
  { id: "security", label: "Security Profile" },
  { id: "ratelimits", label: "Rate Limits" },
];

// ─── Health Tab ────────────────────────────────────────────────────────────────

function HealthTab({ agentId }: { agentId: string }) {
  const metricsQ = useQuery({
    queryKey: ["agent-metrics", agentId],
    queryFn: () => listAgentMetrics({ agent_id: agentId, limit: 1 }),
  });
  const violationsQ = useQuery({
    queryKey: ["violations", agentId],
    queryFn: () => listViolations({ agent_id: agentId, limit: 20 }),
  });
  const sessionsQ = useQuery({
    queryKey: ["sessions", agentId],
    queryFn: () => listSessions({ agent_id: agentId, limit: 5 }),
  });

  const m = metricsQ.data?.agents?.[0];
  const violations = violationsQ.data?.violations ?? [];
  const sessions = sessionsQ.data?.sessions ?? [];
  const blockCount = violations.filter((v) => v.action === "BLOCK").length;
  const alertCount = violations.filter((v) => v.action === "ALERT").length;

  return (
    <div className="space-y-6">
      {/* Stat grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: "Sessions", value: fmt(m?.session_count) },
          { label: "LLM Calls", value: fmt(m?.llm_call_count) },
          { label: "Tool Calls", value: fmt(m?.tool_call_count) },
          { label: "Total Tokens", value: fmt(m?.total_tokens) },
          { label: "Avg Latency", value: m?.avg_latency_ms != null ? `${fmt(m.avg_latency_ms, 0)} ms` : "—" },
          { label: "Errors", value: fmt(m?.error_count) },
          { label: "Anomalies", value: fmt(m?.anomaly_count) },
          { label: "PII Events", value: fmt(m?.pii_event_count) },
        ].map(({ label, value }) => (
          <div key={label} className="bg-white border border-gray-200 rounded-xl p-4">
            <div className="text-2xl font-bold text-gray-900">{value}</div>
            <div className="text-xs text-gray-500 mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {/* Violation summary */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
          <h3 className="font-semibold text-gray-800">Recent Violations</h3>
          <div className="flex gap-3 text-xs">
            <span className="text-red-600 font-medium">{blockCount} blocked</span>
            <span className="text-amber-600 font-medium">{alertCount} alerts</span>
          </div>
        </div>
        {violations.length === 0 ? (
          <div className="px-5 py-8 text-center text-gray-400 text-sm">No violations recorded</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-100">
                <th className="text-left px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide">Action</th>
                <th className="text-left px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide">Rule</th>
                <th className="text-left px-4 py-2 text-xs font-medium text-gray-500 uppercase tracking-wide">When</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {violations.map((v) => (
                <tr key={v.id} className="hover:bg-gray-50">
                  <td className="px-4 py-2">
                    {v.action === "BLOCK" ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-red-100 text-red-700">
                        <XCircle size={10} /> BLOCK
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 text-amber-700">
                        <AlertTriangle size={10} /> ALERT
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2 font-mono text-xs text-gray-700">{v.rule_name}</td>
                  <td className="px-4 py-2 text-xs text-gray-400">{timeAgo(v.timestamp_ns)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Recent sessions */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100">
          <h3 className="font-semibold text-gray-800">Recent Sessions</h3>
        </div>
        {sessions.length === 0 ? (
          <div className="px-5 py-8 text-center text-gray-400 text-sm">No sessions recorded</div>
        ) : (
          <div className="divide-y divide-gray-50">
            {sessions.map((s) => (
              <div key={s.session_id} className="px-5 py-3 flex items-center justify-between hover:bg-gray-50">
                <div className="flex items-center gap-3 min-w-0">
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${s.has_errors ? "bg-red-400" : s.chain_valid ? "bg-green-400" : "bg-amber-400"}`} />
                  <Link
                    to={`/sessions/${s.session_id}`}
                    className="font-mono text-xs text-blue-600 hover:underline truncate"
                  >
                    {s.session_id.slice(0, 20)}…
                  </Link>
                </div>
                <div className="flex items-center gap-4 text-xs text-gray-500 flex-shrink-0 ml-4">
                  <span>{s.llm_call_count} LLM · {s.tool_call_count} tools</span>
                  <span>{s.model}</span>
                  <span>{new Date(s.started_at).toLocaleDateString()}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Tool Baseline Tab ─────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: AgentTool["status"] }) {
  if (status === "approved")
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700">
        <CheckCircle size={10} /> Approved
      </span>
    );
  if (status === "denied")
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700">
        <XCircle size={10} /> Denied
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700">
      <Clock size={10} /> Pending
    </span>
  );
}

function ToolBaselineTab({ agentId }: { agentId: string }) {
  const qc = useQueryClient();
  const toolsQ = useQuery({
    queryKey: ["agent-tools", agentId],
    queryFn: () => listAgentTools(agentId),
  });

  const invalidate = () => qc.invalidateQueries({ queryKey: ["agent-tools", agentId] });

  const approveMut = useMutation({
    mutationFn: ({ toolName }: { toolName: string }) => approveTool(agentId, toolName),
    onSuccess: invalidate,
  });
  const denyMut = useMutation({
    mutationFn: ({ toolName }: { toolName: string }) => denyTool(agentId, toolName),
    onSuccess: invalidate,
  });
  const approveAllMut = useMutation({
    mutationFn: () => approveAllTools(agentId),
    onSuccess: invalidate,
  });

  const data = toolsQ.data;
  const tools = data?.tools ?? [];
  const pending = tools.filter((t) => t.status === "pending");

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex gap-4 text-sm">
          {data && (
            <>
              <span className="text-amber-600 font-medium">{data.counts.pending} pending</span>
              <span className="text-green-600 font-medium">{data.counts.approved} approved</span>
              <span className="text-red-600 font-medium">{data.counts.denied} denied</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          {data?.enforcement_active ? (
            <span className="text-xs px-2 py-0.5 bg-green-50 text-green-700 rounded-full font-medium">
              Enforcement active
            </span>
          ) : (
            <span className="text-xs px-2 py-0.5 bg-gray-100 text-gray-500 rounded-full font-medium">
              Enforcement off
            </span>
          )}
          {pending.length > 0 && (
            <button
              onClick={() => approveAllMut.mutate()}
              disabled={approveAllMut.isPending}
              className="px-3 py-1.5 text-xs bg-green-600 hover:bg-green-700 text-white rounded font-medium disabled:opacity-60"
            >
              {approveAllMut.isPending ? "Approving…" : `Approve all (${pending.length})`}
            </button>
          )}
        </div>
      </div>

      {/* Tool table */}
      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        {tools.length === 0 ? (
          <div className="px-5 py-12 text-center text-gray-400 text-sm">
            No tools observed yet for this agent.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-100">
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Tool</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Status</th>
                <th className="text-right px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Calls</th>
                <th className="text-left px-4 py-3 text-xs font-medium text-gray-500 uppercase tracking-wide">Last seen</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {tools.map((tool) => (
                <tr key={tool.tool_name} className="hover:bg-gray-50">
                  <td className="px-4 py-3 font-mono text-xs text-gray-800">{tool.tool_name}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={tool.status} />
                  </td>
                  <td className="px-4 py-3 text-right text-xs text-gray-500">{tool.call_count}</td>
                  <td className="px-4 py-3 text-xs text-gray-400">
                    {tool.last_seen_at ? new Date(tool.last_seen_at).toLocaleDateString() : "—"}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-2">
                      {tool.status !== "approved" && (
                        <button
                          onClick={() => approveMut.mutate({ toolName: tool.tool_name })}
                          disabled={approveMut.isPending}
                          title="Approve"
                          className="p-1 rounded hover:bg-green-50 text-gray-400 hover:text-green-600 transition-colors"
                        >
                          <Check size={14} />
                        </button>
                      )}
                      {tool.status !== "denied" && (
                        <button
                          onClick={() => denyMut.mutate({ toolName: tool.tool_name })}
                          disabled={denyMut.isPending}
                          title="Deny"
                          className="p-1 rounded hover:bg-red-50 text-gray-400 hover:text-red-600 transition-colors"
                        >
                          <X size={14} />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ─── Policies Tab ──────────────────────────────────────────────────────────────

function PoliciesTab() {
  const rulesQ = useQuery({
    queryKey: ["policy-rules"],
    queryFn: listPolicyRules,
    staleTime: 30_000,
  });

  const rules = rulesQ.data?.rules ?? [];
  const enabled = rules.filter((r) => r.enabled).length;

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-500">
        These are the global policy rules currently active in the proxy. Per-agent rule overrides
        are coming in a future release.
      </p>

      <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
        <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
          <h3 className="font-semibold text-gray-800">Active Policy Rules</h3>
          <span className="text-xs text-gray-500">{enabled} / {rules.length} enabled</span>
        </div>

        {rules.length === 0 ? (
          <div className="px-5 py-8 text-center text-gray-400 text-sm">
            {rulesQ.isFetching ? (
              <span className="flex items-center justify-center gap-2">
                <RefreshCw size={14} className="animate-spin" /> Loading…
              </span>
            ) : (
              "Could not load policy rules. Is the proxy running?"
            )}
          </div>
        ) : (
          <div className="divide-y divide-gray-50">
            {rules.map((rule) => (
              <div key={rule.name} className="px-5 py-3 flex items-center gap-4">
                <div className={`w-2 h-2 rounded-full flex-shrink-0 ${rule.enabled ? "bg-green-400" : "bg-gray-300"}`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-mono text-gray-800 truncate">{rule.name}</p>
                  <p className="text-xs text-gray-400 mt-0.5 truncate">{rule.reason}</p>
                </div>
                <div className="flex-shrink-0 flex items-center gap-2">
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-medium ${
                      rule.action === "BLOCK"
                        ? "bg-red-50 text-red-700"
                        : rule.action === "ALERT"
                        ? "bg-amber-50 text-amber-700"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {rule.action}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <p className="text-xs text-gray-400 text-center">
        Edit rules in the{" "}
        <Link to="/policy" className="text-blue-500 hover:underline">
          Policy Builder
        </Link>
        .
      </p>
    </div>
  );
}

// ─── Security Profile Tab ──────────────────────────────────────────────────────

function OverrideBadge({ feat }: { feat: SecurityFeature }) {
  if (feat.is_agent_override)
    return (
      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-blue-100 text-blue-700 ml-2">
        Agent
      </span>
    );
  if (feat.is_org_override)
    return (
      <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-gray-100 text-gray-500 ml-2">
        Org
      </span>
    );
  return null;
}

function FeatureToggle({
  feat,
  saving,
  onToggle,
}: {
  feat: SecurityFeature;
  saving: boolean;
  onToggle: (key: string, newValue: string) => void;
}) {
  if (feat.type === "boolean") {
    const on = feat.value === "true";
    return (
      <button
        onClick={() => onToggle(feat.key, on ? "false" : "true")}
        disabled={saving}
        className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full border-2 border-transparent transition-colors duration-200 focus:outline-none disabled:opacity-50 ${
          on ? "bg-blue-600" : "bg-gray-200"
        }`}
      >
        <span
          className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow ring-0 transition duration-200 ${
            on ? "translate-x-4" : "translate-x-0"
          }`}
        />
      </button>
    );
  }

  if (feat.type === "enum" && feat.options) {
    return (
      <select
        value={feat.value}
        disabled={saving}
        onChange={(e) => onToggle(feat.key, e.target.value)}
        className="text-xs border border-gray-200 rounded px-2 py-0.5 bg-white text-gray-700 disabled:opacity-50"
      >
        {feat.options.map((opt) => (
          <option key={opt} value={opt}>{opt}</option>
        ))}
      </select>
    );
  }

  if (feat.type === "number") {
    return (
      <input
        type="number"
        value={feat.value}
        disabled={saving}
        onChange={(e) => onToggle(feat.key, e.target.value)}
        onBlur={(e) => onToggle(feat.key, e.target.value)}
        className="text-xs border border-gray-200 rounded px-2 py-0.5 w-20 text-gray-700 disabled:opacity-50"
      />
    );
  }

  return <span className="text-xs text-gray-500">{feat.value}</span>;
}

function SecurityProfileTab({ agentId }: { agentId: string }) {
  const qc = useQueryClient();

  const featuresQ = useQuery({
    queryKey: ["agent-security-features", agentId],
    queryFn: () => getAgentSecurityFeatures(agentId),
    staleTime: 30_000,
  });

  const saveMut = useMutation({
    mutationFn: (updates: Record<string, string>) =>
      updateAgentSecurityFeatures(agentId, updates),
    onMutate: async (updates) => {
      await qc.cancelQueries({ queryKey: ["agent-security-features", agentId] });
      const previous = qc.getQueryData<{ features: SecurityFeature[] }>(
        ["agent-security-features", agentId],
      );
      qc.setQueryData<{ features: SecurityFeature[] }>(
        ["agent-security-features", agentId],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            features: old.features.map((f) => {
              if (!(f.key in updates)) return f;
              const newVal = updates[f.key];
              const isReset = newVal === "";
              return {
                ...f,
                value: isReset ? f.default : newVal,
                is_agent_override: !isReset,
              };
            }),
          };
        },
      );
      return { previous };
    },
    onError: (_err, _updates, ctx) => {
      if (ctx?.previous) {
        qc.setQueryData(["agent-security-features", agentId], ctx.previous);
      }
    },
    onSettled: () =>
      qc.invalidateQueries({ queryKey: ["agent-security-features", agentId] }),
  });

  const features = featuresQ.data?.features ?? [];
  const agentOverrideCount = features.filter((f) => f.is_agent_override).length;

  // Group features by their "group" field
  const groups = features.reduce<Record<string, SecurityFeature[]>>((acc, feat) => {
    if (!acc[feat.group]) acc[feat.group] = [];
    acc[feat.group].push(feat);
    return acc;
  }, {});

  function handleToggle(key: string, newValue: string) {
    saveMut.mutate({ [key]: newValue });
  }

  function handleReset(key: string) {
    saveMut.mutate({ [key]: "" });
  }

  if (featuresQ.isLoading) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400 text-sm gap-2">
        <RefreshCw size={14} className="animate-spin" /> Loading security profile…
      </div>
    );
  }

  if (featuresQ.isError) {
    return (
      <div className="px-5 py-8 text-center text-red-500 text-sm">
        Failed to load security features. Is the backend running?
      </div>
    );
  }

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="bg-white border border-gray-200 rounded-xl px-5 py-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="font-semibold text-gray-800 flex items-center gap-2">
              <Shield size={16} className="text-blue-600" />
              Security Profile
            </h3>
            <p className="text-xs text-gray-500 mt-1">
              Overrides apply to this agent only. Global defaults (Settings → Security Features) are
              the baseline.
            </p>
          </div>
          {agentOverrideCount > 0 && (
            <span className="flex-shrink-0 text-xs px-2 py-0.5 bg-blue-50 text-blue-700 rounded-full font-medium">
              {agentOverrideCount} agent-level override{agentOverrideCount !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      </div>

      {/* Feature groups */}
      {Object.entries(groups).map(([group, feats]) => (
        <div key={group} className="bg-white border border-gray-200 rounded-xl overflow-hidden">
          <div className="px-5 py-2.5 border-b border-gray-100 bg-gray-50">
            <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              {group}
            </h4>
          </div>
          <div className="divide-y divide-gray-50">
            {feats.map((feat) => (
              <div key={feat.key} className="px-5 py-4 flex items-start gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center">
                    <span className="text-sm font-medium text-gray-800">{feat.name}</span>
                    <OverrideBadge feat={feat} />
                    {!feat.is_agent_override && (
                      <span className="ml-2 text-[10px] text-gray-400 italic">
                        {feat.is_org_override ? "from org" : "using default"}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5 leading-relaxed">
                    {feat.description}
                  </p>
                  {feat.requires && (
                    <p className="text-[10px] text-amber-600 mt-1">Requires: {feat.requires}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 pt-0.5">
                  <FeatureToggle
                    feat={feat}
                    saving={saveMut.isPending}
                    onToggle={handleToggle}
                  />
                  {feat.is_agent_override && (
                    <button
                      onClick={() => handleReset(feat.key)}
                      disabled={saveMut.isPending}
                      title="Reset to org/default"
                      className="text-[10px] text-gray-400 hover:text-red-500 transition-colors disabled:opacity-50 ml-1"
                    >
                      Reset
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

// ─── Rate Limits Tab ───────────────────────────────────────────────────────────

function RateLimitsTab({ agent }: { agent: Agent | null }) {
  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-100 rounded-xl p-5">
        <p className="text-sm font-semibold text-blue-800">Per-agent rate limits coming in Phase 6</p>
        <p className="text-sm text-blue-600 mt-1">
          Currently rate limits apply globally via environment variables. Set
          <code className="font-mono bg-blue-100 px-1 mx-1 rounded">AEGIVIS_RATE_LIMIT_MAX_TOKENS_PER_SESSION</code>
          and
          <code className="font-mono bg-blue-100 px-1 mx-1 rounded">AEGIVIS_RATE_LIMIT_MAX_LLM_CALLS_PER_SESSION</code>
          in your docker-compose.yml.
        </p>
      </div>

      {agent && (
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="font-semibold text-gray-800 mb-3">Declared Tool Scope</h3>
          <p className="text-xs text-gray-500 mb-3">
            These tools are declared in the agent registry. The proxy enforces this list when
            baseline enforcement is active.
          </p>
          {agent.allowed_tools.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {agent.allowed_tools.map((t) => (
                <span
                  key={t}
                  className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded font-mono"
                >
                  {t}
                </span>
              ))}
            </div>
          ) : (
            <p className="text-sm text-gray-400 italic">No tool restrictions declared.</p>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main AgentDetail page ─────────────────────────────────────────────────────

export function AgentDetail() {
  const { agentId } = useParams<{ agentId: string }>();
  const [activeTab, setActiveTab] = useState<Tab>("health");

  const agentsQ = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    staleTime: 30_000,
  });

  const agent = agentsQ.data?.agents.find((a) => a.agent_id === agentId) ?? null;

  if (!agentId) return null;

  return (
    <div className="max-w-5xl mx-auto px-6 py-8 space-y-6">
      {/* Back + header */}
      <div>
        <Link
          to="/agents"
          className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-4"
        >
          <ChevronLeft size={16} /> All Agents
        </Link>

        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {agent?.name ?? agentId}
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <code className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                {agentId}
              </code>
              {agent?.owner && (
                <span className="text-xs text-gray-400">{agent.owner}</span>
              )}
            </div>
            {agent?.declared_purpose && (
              <p className="text-sm text-gray-500 mt-2 max-w-2xl">{agent.declared_purpose}</p>
            )}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <div className="flex gap-1">
          {TABS.map((tab) => {
            const icons: Record<Tab, typeof Activity> = {
              health: Activity,
              baselines: Wrench,
              policies: FileText,
              security: Shield,
              ratelimits: Gauge,
            };
            const Icon = icons[tab.id];
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors -mb-px ${
                  activeTab === tab.id
                    ? "border-blue-600 text-blue-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                <Icon size={14} />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "health" && <HealthTab agentId={agentId} />}
        {activeTab === "baselines" && <ToolBaselineTab agentId={agentId} />}
        {activeTab === "policies" && <PoliciesTab />}
        {activeTab === "security" && <SecurityProfileTab agentId={agentId} />}
        {activeTab === "ratelimits" && <RateLimitsTab agent={agent} />}
      </div>
    </div>
  );
}
