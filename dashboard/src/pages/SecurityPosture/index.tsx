import { useState, useEffect, Fragment } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Shield, ShieldCheck, RefreshCw, Download,
  AlertTriangle, CheckCircle, Globe, Wrench, Clock, XCircle, Plus, Trash2,
  ChevronRight, ChevronLeft, ListChecks, Lock,
} from "lucide-react";
import clsx from "clsx";
import {
  getPostureSummary,
  getObservedToolsPosture,
  getPostureAnomalies,
  getObservedNetwork,
  generateManifest,
  getManifestCompliance,
  getSecurityPackageUrl,
  revokeManifest,
  listAgents,
  listEgressRules,
  addEgressRule,
  deleteEgressRule,
  type Agent,
  type EgressRule,
  type ObservedToolPosture,
  type NetworkDestination,
  type PostureSummary,
} from "../../api/client";

// ---------------------------------------------------------------------------
// Trust classification helpers (module-level — shared across components)
// ---------------------------------------------------------------------------

const TRUST_BADGE: Record<string, string> = {
  external_source: "bg-blue-100 text-blue-800",
  internal_sink: "bg-red-100 text-red-800",
  internal: "bg-gray-100 text-gray-700",
};

const TRUST_LABEL: Record<string, string> = {
  external_source: "External Source",
  internal_sink: "Internal Sink ⚠️",
  internal: "Internal",
};

function classifyTrust(name: string): string {
  const n = name.toLowerCase();
  if (/search|fetch|browse|retrieve|scrape|crawl|download|read|get|query|lookup|rss|news|email_read|web|http_get|request|external/.test(n))
    return "external_source";
  if (/send|email|mail|webhook|upload|push|publish|notify|post|export|relay|forward|submit|dispatch|write|save|store|insert|create/.test(n))
    return "internal_sink";
  return "internal";
}

interface ManifestTool {
  name: string;
  trust_classification: string;
  permitted_arguments: Record<string, { type: string }>;
  requires_hitl: boolean;
  hitl_reason: string;
  notes: string;
}

// ---------------------------------------------------------------------------
// Agent selector
// ---------------------------------------------------------------------------

function AgentSelector({
  selectedAgent, onChange,
}: { selectedAgent: string; onChange: (id: string) => void }) {
  const { data, isLoading } = useQuery({ queryKey: ["agents"], queryFn: listAgents, staleTime: 30_000 });
  const agents: Agent[] = data?.agents ?? [];

  if (isLoading) {
    return (
      <div className="border rounded-lg px-3 py-1.5 text-sm text-gray-400 bg-gray-50 animate-pulse w-48">
        Loading agents…
      </div>
    );
  }

  if (agents.length === 0) {
    return (
      <div className="flex items-center gap-2 text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
        <AlertTriangle size={14} className="flex-shrink-0" />
        <span>No agents registered yet — <a href="/agents" className="underline font-medium">register one first</a></span>
      </div>
    );
  }

  return (
    <select
      value={selectedAgent}
      onChange={(e) => onChange(e.target.value)}
      className="border rounded-lg px-3 py-1.5 text-sm text-gray-700 bg-white shadow-sm"
    >
      <option value="">— Select an agent —</option>
      {agents.map((a) => (
        <option key={a.agent_id} value={a.agent_id}>{a.agent_id}</option>
      ))}
    </select>
  );
}

// ---------------------------------------------------------------------------
// Workflow stepper
// ---------------------------------------------------------------------------

const WORKFLOW_STEPS = [
  { n: 1, label: "Observe", desc: "Monitor behavior" },
  { n: 2, label: "Review", desc: "Approve tools & networks" },
  { n: 3, label: "Enforce", desc: "Generate & lock down" },
] as const;

function WorkflowStepper({ step }: { step: 1 | 2 | 3 }) {
  return (
    <div className="flex items-center bg-white border rounded-xl px-6 py-4 shadow-sm">
      {WORKFLOW_STEPS.map((s, i) => (
        <Fragment key={s.n}>
          <div className="flex items-center gap-3">
            <div className={clsx(
              "w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold border-2 transition-colors shrink-0",
              step > s.n ? "bg-green-600 border-green-600 text-white"
                : step === s.n ? "bg-blue-600 border-blue-600 text-white"
                : "border-gray-300 text-gray-400",
            )}>
              {step > s.n ? "✓" : s.n}
            </div>
            <div className="hidden sm:block">
              <div className={clsx("text-sm font-semibold", step >= s.n ? "text-gray-800" : "text-gray-400")}>
                {s.label}
              </div>
              <div className="text-xs text-gray-400">{s.desc}</div>
            </div>
          </div>
          {i < 2 && (
            <div className={clsx("flex-1 h-0.5 mx-4 transition-colors", step > i + 1 ? "bg-green-400" : "bg-gray-200")} />
          )}
        </Fragment>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Status banner
// ---------------------------------------------------------------------------

function StatusBanner({ agentId }: { agentId: string }) {
  const { data: summary, isLoading } = useQuery({
    queryKey: ["posture-summary", agentId],
    queryFn: () => getPostureSummary(agentId),
    enabled: !!agentId,
    refetchInterval: 30_000,
  });

  if (isLoading) return <div className="h-16 bg-gray-100 rounded-xl animate-pulse" />;
  if (!summary) return null;

  if (summary.has_active_manifest) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-start gap-3">
        <ShieldCheck className="text-green-600 mt-0.5" size={20} />
        <div>
          <p className="font-semibold text-green-800">
            Protected — Manifest active since{" "}
            {summary.active_manifest_issued
              ? new Date(summary.active_manifest_issued).toLocaleDateString()
              : "—"}
          </p>
          <p className="text-sm text-green-700 mt-0.5">
            Manifest ID: <code className="font-mono text-xs">{summary.active_manifest_id}</code>
            {summary.active_manifest_expires && (
              <> · Expires {new Date(summary.active_manifest_expires).toLocaleDateString()}</>
            )}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
      <Clock className="text-amber-600 mt-0.5" size={20} />
      <div>
        <p className="font-semibold text-amber-800">Learning Mode — No manifest yet</p>
        <p className="text-sm text-amber-700 mt-0.5">
          {summary.tool_calls} tool calls observed across {summary.total_sessions} sessions.
          Follow the steps below to generate a capability manifest.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stats row
// ---------------------------------------------------------------------------

function StatsRow({ agentId }: { agentId: string }) {
  const { data: summary } = useQuery({
    queryKey: ["posture-summary", agentId],
    queryFn: () => getPostureSummary(agentId),
    enabled: !!agentId,
  });
  const { data: compliance } = useQuery({
    queryKey: ["manifest-compliance", agentId],
    queryFn: () => getManifestCompliance(agentId, 24),
    enabled: !!agentId && !!summary?.has_active_manifest,
  });

  const stats = [
    { label: "Sessions", value: summary?.total_sessions ?? "—" },
    { label: "Tool Calls", value: summary?.tool_calls ?? "—" },
    { label: "Unique Tools", value: summary?.unique_tools ?? "—" },
    { label: "Anomalies", value: summary?.anomaly_count ?? "—", warn: (summary?.anomaly_count ?? 0) > 0 },
    ...(compliance ? [
      { label: "Allowed (24h)", value: compliance.allowed_tool_calls, ok: true },
      { label: "Blocked (24h)", value: compliance.blocked_tool_calls, warn: compliance.blocked_tool_calls > 0 },
    ] : []),
  ];

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
      {stats.map((s) => (
        <div key={s.label} className="bg-white border rounded-xl p-3 shadow-sm">
          <div className="text-xs text-gray-500">{s.label}</div>
          <div className={clsx(
            "text-xl font-bold mt-0.5",
            s.warn ? "text-amber-600" : s.ok ? "text-green-600" : "text-gray-800",
          )}>
            {s.value}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tools table (Step 1 — read-only)
// ---------------------------------------------------------------------------

function ToolsTable({ agentId }: { agentId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["posture-tools", agentId],
    queryFn: () => getObservedToolsPosture(agentId),
    enabled: !!agentId,
  });

  if (isLoading) return <div className="h-32 bg-gray-100 rounded-xl animate-pulse" />;
  if (!data?.tools.length) return <p className="text-sm text-gray-500">No tool calls observed yet.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b bg-gray-50">
            <th className="text-left py-2 px-3 font-medium text-gray-600">Tool</th>
            <th className="text-left py-2 px-3 font-medium text-gray-600">Classification</th>
            <th className="text-right py-2 px-3 font-medium text-gray-600">Calls</th>
            <th className="text-left py-2 px-3 font-medium text-gray-600">Arg Keys</th>
            <th className="text-right py-2 px-3 font-medium text-gray-600">Last Seen</th>
          </tr>
        </thead>
        <tbody>
          {data.tools.map((t) => {
            const trust = classifyTrust(t.tool_name);
            return (
              <tr key={t.tool_name} className="border-b hover:bg-gray-50">
                <td className="py-2 px-3 font-mono text-xs font-semibold text-gray-900">{t.tool_name}</td>
                <td className="py-2 px-3">
                  <span className={clsx("px-2 py-0.5 rounded text-xs font-medium", TRUST_BADGE[trust])}>
                    {TRUST_LABEL[trust]}
                  </span>
                </td>
                <td className="py-2 px-3 text-right tabular-nums">{t.call_count}</td>
                <td className="py-2 px-3 text-xs text-gray-500 max-w-xs truncate">
                  {t.observed_arg_keys.slice(0, 6).join(", ")}
                  {t.observed_arg_keys.length > 6 && " …"}
                </td>
                <td className="py-2 px-3 text-right text-xs text-gray-500">
                  {t.last_seen ? new Date(t.last_seen).toLocaleDateString() : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Anomalies panel (Step 1 — read-only)
// ---------------------------------------------------------------------------

function AnomaliesPanel({ agentId }: { agentId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["posture-anomalies", agentId],
    queryFn: () => getPostureAnomalies(agentId, 30),
    enabled: !!agentId,
  });

  if (isLoading) return <div className="h-20 bg-gray-100 rounded-xl animate-pulse" />;

  if (!data?.anomalies.length) {
    return (
      <div className="flex items-center gap-2 text-sm text-green-600">
        <CheckCircle size={15} /> No anomalies in the last 30 days
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {data.anomalies.slice(0, 5).map((a) => (
        <div key={a.id} className="flex items-start gap-3 bg-amber-50 border border-amber-200 rounded-lg p-3">
          <AlertTriangle size={14} className="text-amber-600 mt-0.5 shrink-0" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-gray-800">{a.rule_name}</p>
            <p className="text-xs text-gray-600 mt-0.5 truncate">{a.reason}</p>
            <p className="text-xs text-gray-400 mt-1">{new Date(a.created_at).toLocaleString()} · {a.action}</p>
          </div>
        </div>
      ))}
      {data.count > 5 && (
        <p className="text-xs text-gray-500 text-right">+{data.count - 5} more in Violations page</p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Network panel (Step 1 — read-only)
// ---------------------------------------------------------------------------

function NetworkPanel({ agentId }: { agentId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["posture-network", agentId],
    queryFn: () => getObservedNetwork(agentId, 30),
    enabled: !!agentId,
  });

  if (isLoading) return <div className="h-20 bg-gray-100 rounded-xl animate-pulse" />;
  if (!data?.destinations.length) return <p className="text-sm text-gray-500">No outbound network calls observed.</p>;

  return (
    <div className="space-y-1">
      {data.destinations.map((d) => (
        <div key={d.destination} className="flex items-center gap-2 text-sm py-1 border-b last:border-0">
          <CheckCircle size={13} className="text-green-500 shrink-0" />
          <span className="font-mono text-xs text-gray-700">{d.destination}</span>
          <span className="ml-auto text-xs text-gray-400">{d.seen_count}×</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Readiness panel (bottom of Step 1 — prompts user to proceed)
// ---------------------------------------------------------------------------

function ReadinessPanel({
  summary,
  onNext,
}: { summary: PostureSummary | undefined; onNext: () => void }) {
  const toolCount = summary?.unique_tools ?? 0;
  const sessionCount = summary?.total_sessions ?? 0;
  const isReady = toolCount >= 1 || sessionCount >= 1;

  return (
    <div className={clsx(
      "rounded-xl border p-4 flex items-center justify-between gap-4 flex-wrap",
      isReady ? "bg-blue-50 border-blue-200" : "bg-gray-50 border-gray-200",
    )}>
      <div>
        <p className={clsx("font-semibold text-sm", isReady ? "text-blue-800" : "text-gray-600")}>
          {isReady ? "Enough data collected — ready to review" : "Collecting baseline data…"}
        </p>
        <p className="text-xs text-gray-500 mt-0.5">
          {sessionCount} session{sessionCount !== 1 ? "s" : ""} · {toolCount} unique tool{toolCount !== 1 ? "s" : ""} observed
        </p>
      </div>
      <button
        onClick={onNext}
        disabled={!isReady}
        className={clsx(
          "inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
          isReady ? "bg-blue-600 text-white hover:bg-blue-700" : "bg-gray-200 text-gray-400 cursor-not-allowed",
        )}
      >
        Review & Approve
        <ChevronRight size={14} />
      </button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Review panel (Step 2)
// ---------------------------------------------------------------------------

function ReviewPanel({
  tools,
  dests,
  anomalyCount,
  onBack,
  onNext,
}: {
  tools: ObservedToolPosture[];
  dests: NetworkDestination[];
  anomalyCount: number;
  onBack: () => void;
  onNext: (tools: ManifestTool[], dests: string[]) => void;
}) {
  const [checkedTools, setCheckedTools] = useState<Set<string>>(() => new Set(tools.map((t) => t.tool_name)));
  const [checkedDests, setCheckedDests] = useState<Set<string>>(() => new Set(dests.map((d) => d.destination)));

  function toManifestTool(t: ObservedToolPosture): ManifestTool {
    const trust = classifyTrust(t.tool_name);
    return {
      name: t.tool_name,
      trust_classification: trust,
      permitted_arguments: Object.fromEntries(t.observed_arg_keys.map((k) => [k, { type: "string" }])),
      requires_hitl: trust === "internal_sink",
      hitl_reason: trust === "internal_sink" ? "Writes data — review required" : "",
      notes: `Observed ${t.call_count} times in last 30 days`,
    };
  }

  function handleNext() {
    onNext(
      tools.filter((t) => checkedTools.has(t.tool_name)).map(toManifestTool),
      dests.filter((d) => checkedDests.has(d.destination)).map((d) => d.destination),
    );
  }

  const sinkCount = tools.filter((t) => classifyTrust(t.tool_name) === "internal_sink").length;

  return (
    <div className="space-y-6">
      {/* Anomaly warning */}
      {anomalyCount > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="text-amber-600 shrink-0 mt-0.5" size={18} />
          <div>
            <p className="font-semibold text-amber-800 text-sm">
              {anomalyCount} anomalies detected during observation
            </p>
            <p className="text-xs text-amber-700 mt-0.5">
              Review the anomalies shown in Step 1. Consider excluding tools associated with violations.
            </p>
          </div>
        </div>
      )}

      {/* Tools checklist */}
      <div className="bg-white border rounded-xl shadow-sm overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b bg-gray-50">
          <div className="flex items-center gap-2">
            <Wrench size={14} className="text-gray-500" />
            <h2 className="font-semibold text-gray-700 text-sm">
              Approve Tools — {checkedTools.size} of {tools.length} selected
            </h2>
          </div>
          <div className="flex items-center gap-3">
            {sinkCount > 0 && (
              <button
                onClick={() =>
                  setCheckedTools(() => {
                    const next = new Set(tools.map((t) => t.tool_name));
                    tools.filter((t) => classifyTrust(t.tool_name) === "internal_sink").forEach((t) => next.delete(t.tool_name));
                    return next;
                  })
                }
                className="text-xs text-amber-600 hover:underline"
              >
                Exclude sinks
              </button>
            )}
            <button
              onClick={() => setCheckedTools(new Set(tools.map((t) => t.tool_name)))}
              className="text-xs text-blue-600 hover:underline"
            >
              Select all
            </button>
            <button onClick={() => setCheckedTools(new Set())} className="text-xs text-gray-500 hover:underline">
              None
            </button>
          </div>
        </div>

        {tools.length === 0 ? (
          <p className="p-5 text-sm text-gray-500">No tools observed yet. Go back and collect more data.</p>
        ) : (
          <div className="divide-y">
            {tools.map((t) => {
              const trust = classifyTrust(t.tool_name);
              const isChecked = checkedTools.has(t.tool_name);
              return (
                <label
                  key={t.tool_name}
                  className={clsx(
                    "flex items-center gap-3 px-5 py-3 cursor-pointer hover:bg-gray-50 transition-colors",
                    !isChecked && "opacity-50",
                    trust === "internal_sink" && "bg-red-50/40",
                  )}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={(e) => {
                      const s = new Set(checkedTools);
                      e.target.checked ? s.add(t.tool_name) : s.delete(t.tool_name);
                      setCheckedTools(s);
                    }}
                    className="rounded border-gray-300"
                  />
                  <span className="font-mono text-sm font-semibold flex-1 text-gray-900">{t.tool_name}</span>
                  <span className={clsx("px-2 py-0.5 rounded text-xs font-medium", TRUST_BADGE[trust])}>
                    {TRUST_LABEL[trust]}
                  </span>
                  <span className="text-xs text-gray-400 shrink-0">{t.call_count}×</span>
                </label>
              );
            })}
          </div>
        )}
      </div>

      {/* Destinations checklist */}
      <div className="bg-white border rounded-xl shadow-sm overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3 border-b bg-gray-50">
          <div className="flex items-center gap-2">
            <Globe size={14} className="text-gray-500" />
            <h2 className="font-semibold text-gray-700 text-sm">
              Network Destinations — {checkedDests.size} of {dests.length} selected
            </h2>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setCheckedDests(new Set(dests.map((d) => d.destination)))}
              className="text-xs text-blue-600 hover:underline"
            >
              Select all
            </button>
            <button onClick={() => setCheckedDests(new Set())} className="text-xs text-gray-500 hover:underline">
              None
            </button>
          </div>
        </div>

        {dests.length === 0 ? (
          <p className="p-5 text-sm text-gray-500">No network destinations observed.</p>
        ) : (
          <div className="divide-y">
            {dests.map((d) => {
              const isChecked = checkedDests.has(d.destination);
              return (
                <label
                  key={d.destination}
                  className={clsx(
                    "flex items-center gap-3 px-5 py-3 cursor-pointer hover:bg-gray-50 transition-colors",
                    !isChecked && "opacity-50",
                  )}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={(e) => {
                      const s = new Set(checkedDests);
                      e.target.checked ? s.add(d.destination) : s.delete(d.destination);
                      setCheckedDests(s);
                    }}
                    className="rounded border-gray-300"
                  />
                  <span className="font-mono text-xs flex-1 text-gray-800">{d.destination}</span>
                  <span className="text-xs text-gray-400 shrink-0">{d.seen_count}×</span>
                </label>
              );
            })}
          </div>
        )}
      </div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 px-4 py-2 border border-gray-200 text-gray-600 rounded-lg text-sm hover:bg-gray-50 transition-colors"
        >
          <ChevronLeft size={14} />
          Back to Observe
        </button>
        <button
          onClick={handleNext}
          disabled={checkedTools.size === 0}
          className="inline-flex items-center gap-2 px-5 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
        >
          {checkedTools.size} tools · {checkedDests.size} destinations approved
          <ChevronRight size={14} />
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Egress allowlist panel (Step 3)
// ---------------------------------------------------------------------------

function EgressRulesPanel({ observedDestinations }: { observedDestinations: string[] }) {
  const qc = useQueryClient();
  const [newDest, setNewDest] = useState("");
  const [newNotes, setNewNotes] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["egress-rules"],
    queryFn: listEgressRules,
    refetchInterval: 30_000,
  });

  const rules: EgressRule[] = data?.rules ?? [];
  const ruleSet = new Set(rules.map((r) => r.destination));

  const add = useMutation({
    mutationFn: ({ dest, notes }: { dest: string; notes: string }) => addEgressRule(dest, notes),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["egress-rules"] });
      setNewDest("");
      setNewNotes("");
    },
  });

  const remove = useMutation({
    mutationFn: (dest: string) => deleteEgressRule(dest),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["egress-rules"] }),
  });

  function handleAdd(dest: string, notes = "") {
    const d = dest.trim().toLowerCase();
    if (!d) return;
    add.mutate({ dest: d, notes });
  }

  return (
    <div className="bg-white border rounded-xl shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b bg-gray-50">
        <div className="flex items-center gap-2">
          <Globe size={15} className="text-gray-500" />
          <h2 className="font-semibold text-gray-700 text-sm">Egress Allowlist</h2>
        </div>
        <span className="text-xs text-gray-400">
          {rules.length} rule{rules.length !== 1 ? "s" : ""} · proxy re-reads every 60s
        </span>
      </div>
      <div className="p-5 space-y-4">
        {/* Add rule form */}
        <div className="flex gap-2 flex-wrap">
          <input
            type="text"
            value={newDest}
            onChange={(e) => setNewDest(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd(newDest, newNotes)}
            placeholder="api.openai.com  ·  *.anthropic.com  ·  10.0.0.0/8"
            className="flex-1 min-w-[200px] border rounded-lg px-3 py-1.5 text-sm"
          />
          <input
            type="text"
            value={newNotes}
            onChange={(e) => setNewNotes(e.target.value)}
            placeholder="Notes (optional)"
            className="w-40 border rounded-lg px-3 py-1.5 text-sm"
          />
          <button
            onClick={() => handleAdd(newDest, newNotes)}
            disabled={add.isPending || !newDest.trim()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            <Plus size={13} />
            Allow
          </button>
        </div>

        {/* Approved destinations quick-add */}
        {observedDestinations.length > 0 && (
          <div>
            <p className="text-xs text-gray-500 mb-2">
              Approved destinations — click to add to allowlist:
            </p>
            <div className="flex flex-wrap gap-2">
              {observedDestinations.map((d) => (
                <button
                  key={d}
                  onClick={() => handleAdd(d, "approved in manifest review")}
                  disabled={ruleSet.has(d) || add.isPending}
                  className={clsx(
                    "px-2 py-1 rounded text-xs font-mono transition-colors",
                    ruleSet.has(d)
                      ? "bg-green-50 text-green-700 border border-green-200 cursor-default"
                      : "bg-gray-100 text-gray-700 hover:bg-blue-50 hover:text-blue-700 border border-gray-200",
                  )}
                >
                  {ruleSet.has(d)
                    ? <><CheckCircle size={11} className="inline mr-1" />{d}</>
                    : <><Plus size={11} className="inline mr-1" />{d}</>
                  }
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Current rules */}
        {isLoading ? (
          <div className="h-20 bg-gray-100 rounded-lg animate-pulse" />
        ) : rules.length === 0 ? (
          <p className="text-sm text-amber-600 flex items-center gap-2">
            <AlertTriangle size={14} />
            No allowlist rules configured. All destinations permitted (audit mode).
          </p>
        ) : (
          <div className="divide-y border rounded-lg overflow-hidden">
            {rules.map((rule) => (
              <div key={rule.destination} className="flex items-center gap-3 px-3 py-2 bg-white hover:bg-gray-50">
                <CheckCircle size={13} className="text-green-500 shrink-0" />
                <span className="font-mono text-xs text-gray-800 flex-1">{rule.destination}</span>
                {rule.notes && (
                  <span className="text-xs text-gray-400 truncate max-w-[160px]">{rule.notes}</span>
                )}
                <button
                  onClick={() => remove.mutate(rule.destination)}
                  disabled={remove.isPending}
                  className="text-gray-400 hover:text-red-500 disabled:opacity-40 transition-colors"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            ))}
          </div>
        )}

        {add.isError && (
          <p className="text-xs text-red-600">Failed to add rule. Check destination format.</p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Enforce panel (Step 3)
// ---------------------------------------------------------------------------

function EnforcePanel({
  agentId,
  approvedTools,
  approvedDests,
  onBack,
}: {
  agentId: string;
  approvedTools: ManifestTool[];
  approvedDests: string[];
  onBack: () => void;
}) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const [validityDays, setValidityDays] = useState(30);
  const [generated, setGenerated] = useState(false);

  const generate = useMutation({
    mutationFn: () =>
      generateManifest(agentId, {
        validity_days: validityDays,
        notes: note,
        permitted_tools: approvedTools as unknown as Record<string, unknown>[],
        permitted_network_destinations: approvedDests,
      }),
    onSuccess: () => {
      setGenerated(true);
      qc.invalidateQueries({ queryKey: ["posture-summary", agentId] });
      qc.invalidateQueries({ queryKey: ["manifest-compliance", agentId] });
    },
  });

  const revoke = useMutation({
    mutationFn: () => revokeManifest(agentId),
    onSuccess: () => {
      setGenerated(false);
      qc.invalidateQueries({ queryKey: ["posture-summary", agentId] });
    },
  });

  const sinkCount = approvedTools.filter((t) => t.trust_classification === "internal_sink").length;

  return (
    <div className="space-y-6">
      {/* Approved summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 flex items-start gap-3">
        <ListChecks className="text-blue-600 shrink-0 mt-0.5" size={20} />
        <div>
          <p className="font-semibold text-blue-800 text-sm">Review Complete</p>
          <p className="text-xs text-blue-700 mt-0.5">
            {approvedTools.length} tool{approvedTools.length !== 1 ? "s" : ""} approved ·{" "}
            {approvedDests.length} network destination{approvedDests.length !== 1 ? "s" : ""} selected
            {sinkCount > 0 && <> · {sinkCount} sink{sinkCount !== 1 ? "s" : ""} flagged for HITL review</>}
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {approvedTools.map((t) => (
              <span
                key={t.name}
                className={clsx(
                  "px-2 py-0.5 rounded text-xs font-mono font-medium",
                  TRUST_BADGE[t.trust_classification] ?? "bg-gray-100 text-gray-700",
                )}
              >
                {t.name}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Generate section */}
      <div className="bg-white border rounded-xl p-5 space-y-4">
        <h3 className="font-semibold text-gray-800 flex items-center gap-2">
          <Lock size={16} className="text-blue-600" />
          Generate Capability Manifest
        </h3>
        <p className="text-sm text-gray-600">
          Creates a cryptographically signed manifest from your {approvedTools.length} approved tools.
          The proxy enforces this list within 60 seconds — any tool outside it is blocked.
        </p>
        <div className="flex flex-wrap gap-4">
          <div>
            <label className="text-xs text-gray-500 block mb-1">Validity (days)</label>
            <select
              value={validityDays}
              onChange={(e) => setValidityDays(Number(e.target.value))}
              className="border rounded px-2 py-1 text-sm"
            >
              <option value={7}>7 days</option>
              <option value={30}>30 days</option>
              <option value={90}>90 days</option>
            </select>
          </div>
          <div className="flex-1">
            <label className="text-xs text-gray-500 block mb-1">Notes (optional)</label>
            <input
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g. Production baseline March 2026"
              className="border rounded px-2 py-1 text-sm w-full"
            />
          </div>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={() => generate.mutate()}
            disabled={generate.isPending}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {generate.isPending ? <RefreshCw size={14} className="animate-spin" /> : <Shield size={14} />}
            Generate Manifest
          </button>
          {generated && (
            <>
              <a
                href={getSecurityPackageUrl(agentId)}
                className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium hover:bg-green-700 transition-colors"
              >
                <Download size={14} />
                Download Package
              </a>
              <button
                onClick={() => revoke.mutate()}
                disabled={revoke.isPending}
                className="inline-flex items-center gap-2 px-3 py-2 text-red-600 border border-red-200 rounded-lg text-sm hover:bg-red-50 disabled:opacity-50 transition-colors ml-auto"
              >
                <XCircle size={14} />
                Revoke
              </button>
            </>
          )}
        </div>
        {generated && (
          <p className="text-sm text-green-600 flex items-center gap-1">
            <CheckCircle size={13} />
            Manifest generated! Proxy enforcement starts within 60 seconds.
          </p>
        )}
        {generate.isError && (
          <p className="text-sm text-red-600">Failed to generate manifest. Is the backend running?</p>
        )}
      </div>

      {/* Egress allowlist — quick-add the approved destinations */}
      <EgressRulesPanel observedDestinations={approvedDests} />

      {/* Navigation */}
      <div>
        <button
          onClick={onBack}
          className="inline-flex items-center gap-2 px-4 py-2 border border-gray-200 text-gray-600 rounded-lg text-sm hover:bg-gray-50 transition-colors"
        >
          <ChevronLeft size={14} />
          Back to Review
        </button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function SecurityPosturePage() {
  const [selectedAgent, setSelectedAgent] = useState("");
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [approvedTools, setApprovedTools] = useState<ManifestTool[]>([]);
  const [approvedDests, setApprovedDests] = useState<string[]>([]);

  // Reset workflow when agent changes
  useEffect(() => {
    setStep(1);
    setApprovedTools([]);
    setApprovedDests([]);
  }, [selectedAgent]);

  // Hoist data queries so ReviewPanel can access raw arrays without re-fetching
  const { data: summary } = useQuery({
    queryKey: ["posture-summary", selectedAgent],
    queryFn: () => getPostureSummary(selectedAgent),
    enabled: !!selectedAgent,
    refetchInterval: 30_000,
  });

  const { data: toolsData } = useQuery({
    queryKey: ["posture-tools", selectedAgent],
    queryFn: () => getObservedToolsPosture(selectedAgent),
    enabled: !!selectedAgent,
  });

  const { data: networkData } = useQuery({
    queryKey: ["posture-network", selectedAgent],
    queryFn: () => getObservedNetwork(selectedAgent, 30),
    enabled: !!selectedAgent,
    staleTime: 30_000,
  });

  const { data: anomaliesData } = useQuery({
    queryKey: ["posture-anomalies", selectedAgent],
    queryFn: () => getPostureAnomalies(selectedAgent, 30),
    enabled: !!selectedAgent,
  });

  const tools = toolsData?.tools ?? [];
  const dests = networkData?.destinations ?? [];
  const anomalyCount = anomaliesData?.count ?? 0;

  function handleReviewConfirm(tools: ManifestTool[], dests: string[]) {
    setApprovedTools(tools);
    setApprovedDests(dests);
    setStep(3);
  }

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <ShieldCheck size={24} className="text-blue-600" />
            Security Posture
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Observe agent behavior, review tool access, and generate a signed capability manifest.
          </p>
        </div>
        <AgentSelector
          selectedAgent={selectedAgent}
          onChange={(id) => setSelectedAgent(id)}
        />
      </div>

      {!selectedAgent ? (
        <div className="text-center py-16 text-gray-400">
          <Shield size={48} className="mx-auto mb-3 opacity-30" />
          <p>Select an agent to view its security posture</p>
        </div>
      ) : (
        <>
          {/* Status banner — always visible */}
          <StatusBanner agentId={selectedAgent} />

          {/* Workflow stepper */}
          <WorkflowStepper step={step} />

          {/* Step 1: Observe */}
          {step === 1 && (
            <div className="space-y-6">
              <StatsRow agentId={selectedAgent} />

              <section className="bg-white border rounded-xl shadow-sm overflow-hidden">
                <div className="flex items-center gap-2 px-5 py-3 border-b bg-gray-50">
                  <Wrench size={15} className="text-gray-500" />
                  <h2 className="font-semibold text-gray-700 text-sm">Observed Tools</h2>
                </div>
                <div className="p-5">
                  <ToolsTable agentId={selectedAgent} />
                </div>
              </section>

              <section className="bg-white border rounded-xl shadow-sm overflow-hidden">
                <div className="flex items-center gap-2 px-5 py-3 border-b bg-gray-50">
                  <AlertTriangle size={15} className="text-gray-500" />
                  <h2 className="font-semibold text-gray-700 text-sm">Anomalies (last 30 days)</h2>
                </div>
                <div className="p-5">
                  <AnomaliesPanel agentId={selectedAgent} />
                </div>
              </section>

              <section className="bg-white border rounded-xl shadow-sm overflow-hidden">
                <div className="flex items-center gap-2 px-5 py-3 border-b bg-gray-50">
                  <Globe size={15} className="text-gray-500" />
                  <h2 className="font-semibold text-gray-700 text-sm">Observed Network Destinations</h2>
                </div>
                <div className="p-5">
                  <NetworkPanel agentId={selectedAgent} />
                </div>
              </section>

              <ReadinessPanel summary={summary} onNext={() => setStep(2)} />
            </div>
          )}

          {/* Step 2: Review */}
          {step === 2 && (
            <ReviewPanel
              tools={tools}
              dests={dests}
              anomalyCount={anomalyCount}
              onBack={() => setStep(1)}
              onNext={handleReviewConfirm}
            />
          )}

          {/* Step 3: Enforce */}
          {step === 3 && (
            <EnforcePanel
              agentId={selectedAgent}
              approvedTools={approvedTools}
              approvedDests={approvedDests}
              onBack={() => setStep(2)}
            />
          )}
        </>
      )}
    </div>
  );
}
