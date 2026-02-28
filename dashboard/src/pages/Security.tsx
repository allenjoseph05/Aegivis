import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { formatDistanceToNow } from "date-fns";
import {
  Shield,
  ShieldAlert,
  RefreshCw,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Lock,
  Unlock,
} from "lucide-react";
import {
  listToolPermissions,
  reloadToolPermissions,
  listViolations,
  getViolationsSummary,
  getMetricsOverview,
} from "../api/client";
import type { ToolPermissionRule, Violation } from "../api/client";
import clsx from "clsx";

// ---------------------------------------------------------------------------
// Badge helpers
// ---------------------------------------------------------------------------

function ActionBadge({ action }: { action: string }) {
  const cls: Record<string, string> = {
    BLOCK: "bg-red-100 text-red-700 border-red-200",
    ALERT: "bg-amber-100 text-amber-700 border-amber-200",
    LOG: "bg-gray-100 text-gray-600 border-gray-200",
  };
  return (
    <span
      className={clsx(
        "px-2 py-0.5 rounded border text-xs font-semibold tracking-wide",
        cls[action] ?? "bg-gray-100 text-gray-600 border-gray-200"
      )}
    >
      {action}
    </span>
  );
}

function EnabledBadge({ enabled }: { enabled: boolean }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
        enabled
          ? "bg-green-100 text-green-700"
          : "bg-gray-100 text-gray-500"
      )}
    >
      {enabled ? <Lock size={10} /> : <Unlock size={10} />}
      {enabled ? "enabled" : "disabled"}
    </span>
  );
}

function SeverityBadge({ action }: { action: string }) {
  // Reuse ActionBadge for violations table (same palette)
  return <ActionBadge action={action} />;
}

// ---------------------------------------------------------------------------
// Tool Permission Rules panel
// ---------------------------------------------------------------------------

function ToolPermissionsPanel() {
  const queryClient = useQueryClient();
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ["toolPermissions"],
    queryFn: listToolPermissions,
    refetchInterval: 30_000,
  });

  const reloadMutation = useMutation({
    mutationFn: () => reloadToolPermissions(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["toolPermissions"] });
    },
  });

  const rules = data?.rules ?? [];

  return (
    <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b bg-gray-50">
        <div className="flex items-center gap-2">
          <Shield size={18} className="text-blue-600" />
          <h2 className="font-semibold text-gray-900">Tool Permission Rules</h2>
          {data && (
            <span className="text-xs text-gray-500 ml-1">
              {data.enabled} of {data.total} enabled
            </span>
          )}
        </div>
        <button
          onClick={() => reloadMutation.mutate()}
          disabled={reloadMutation.isPending || isFetching}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw
            size={13}
            className={clsx(
              (reloadMutation.isPending || isFetching) && "animate-spin"
            )}
          />
          Reload
        </button>
      </div>

      {error ? (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-4 text-sm">
          <AlertCircle size={14} />
          Could not reach proxy at {import.meta.env.VITE_PROXY_URL || "http://localhost:8080"}.
          Is it running?
        </div>
      ) : isLoading ? (
        <div className="p-4 space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />
          ))}
        </div>
      ) : rules.length === 0 ? (
        <div className="py-10 text-center text-gray-400 text-sm">
          No tool permission rules loaded.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide w-6" />
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Rule
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Tools
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Agents
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Action
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rules.map((rule) => (
              <ToolRuleRow
                key={rule.name}
                rule={rule}
                expanded={expanded === rule.name}
                onToggle={() =>
                  setExpanded(expanded === rule.name ? null : rule.name)
                }
              />
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function ToolRuleRow({
  rule,
  expanded,
  onToggle,
}: {
  rule: ToolPermissionRule;
  expanded: boolean;
  onToggle: () => void;
}) {
  const toolsLabel =
    rule.tools.includes("*")
      ? rule.except_tools.length
        ? `* except [${rule.except_tools.join(", ")}]`
        : "*"
      : rule.tools.join(", ");

  const agentsLabel =
    !rule.agents.length || rule.agents.includes("*")
      ? "all agents"
      : rule.agents.join(", ");

  const hasDetails =
    rule.arg_conditions.length > 0 || rule.reason.length > 0;

  return (
    <>
      <tr
        className={clsx(
          "hover:bg-gray-50 transition-colors",
          !rule.enabled && "opacity-50"
        )}
      >
        <td className="px-4 py-3">
          {hasDetails && (
            <button
              onClick={onToggle}
              className="text-gray-400 hover:text-gray-600"
            >
              {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>
          )}
        </td>
        <td className="px-4 py-3">
          <span className="font-mono text-xs text-gray-800">{rule.name}</span>
        </td>
        <td className="px-4 py-3">
          <span className="font-mono text-xs text-gray-600 truncate max-w-56 block">
            {toolsLabel}
          </span>
        </td>
        <td className="px-4 py-3 text-xs text-gray-600">{agentsLabel}</td>
        <td className="px-4 py-3">
          <ActionBadge action={rule.action} />
        </td>
        <td className="px-4 py-3">
          <EnabledBadge enabled={rule.enabled} />
        </td>
      </tr>
      {expanded && (
        <tr className="bg-gray-50 border-b">
          <td colSpan={6} className="px-8 py-3">
            <div className="space-y-2 text-xs text-gray-600">
              <div>
                <span className="font-semibold text-gray-700">Reason: </span>
                {rule.reason}
              </div>
              {rule.arg_conditions.length > 0 && (
                <div>
                  <span className="font-semibold text-gray-700">
                    Arg conditions:{" "}
                  </span>
                  <span className="font-mono">
                    {rule.arg_conditions
                      .map((c) => `${c.field} ${c.op} ${JSON.stringify(c.value ?? "")}`)
                      .join(" AND ")}
                  </span>
                </div>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Violation Summary panel
// ---------------------------------------------------------------------------

function ViolationSummaryPanel() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["violationsSummary"],
    queryFn: getViolationsSummary,
    refetchInterval: 15_000,
  });

  const summary = data?.summary ?? [];

  return (
    <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
      <div className="flex items-center gap-2 px-5 py-4 border-b bg-gray-50">
        <ShieldAlert size={18} className="text-amber-600" />
        <h2 className="font-semibold text-gray-900">Violation Summary</h2>
        <span className="text-xs text-gray-500 ml-1">by rule</span>
      </div>

      {error ? (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-4 text-sm">
          <AlertCircle size={14} />
          Failed to load. Is the backend running?
        </div>
      ) : isLoading ? (
        <div className="p-4 space-y-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="h-8 bg-gray-100 rounded animate-pulse" />
          ))}
        </div>
      ) : summary.length === 0 ? (
        <div className="py-8 text-center text-gray-400 text-sm">
          No violations recorded yet.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Rule
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Action
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Count
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Last Fired
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {summary.map((row) => (
              <tr key={`${row.rule_name}-${row.action}`} className="hover:bg-gray-50">
                <td className="px-4 py-2.5 font-mono text-xs text-gray-800">
                  {row.rule_name}
                </td>
                <td className="px-4 py-2.5">
                  <ActionBadge action={row.action} />
                </td>
                <td className="px-4 py-2.5 text-right font-semibold text-gray-700">
                  {row.count.toLocaleString()}
                </td>
                <td className="px-4 py-2.5 text-xs text-gray-500">
                  {row.last_fired_ns
                    ? formatDistanceToNow(
                        new Date(row.last_fired_ns / 1_000_000),
                        { addSuffix: true }
                      )
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Recent Violations panel
// ---------------------------------------------------------------------------

function RecentViolationsPanel() {
  const [actionFilter, setActionFilter] = useState("");
  const [ruleFilter, setRuleFilter] = useState("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 25;

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ["violations", actionFilter, ruleFilter, page],
    queryFn: () =>
      listViolations({
        action: actionFilter || undefined,
        rule_name: ruleFilter || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    refetchInterval: 10_000,
  });

  const violations = data?.violations ?? [];

  return (
    <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b bg-gray-50">
        <div className="flex items-center gap-2">
          <AlertCircle size={18} className="text-red-500" />
          <h2 className="font-semibold text-gray-900">Recent Violations</h2>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
        >
          <RefreshCw size={13} className={clsx(isFetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex gap-3 px-5 py-3 border-b bg-gray-50">
        <select
          value={actionFilter}
          onChange={(e) => { setActionFilter(e.target.value); setPage(0); }}
          className="border rounded-lg px-3 py-1.5 text-xs text-gray-600 bg-white shadow-sm outline-none"
        >
          <option value="">All actions</option>
          <option value="BLOCK">BLOCK</option>
          <option value="ALERT">ALERT</option>
          <option value="LOG">LOG</option>
        </select>
        <input
          type="text"
          placeholder="Filter by rule name..."
          value={ruleFilter}
          onChange={(e) => { setRuleFilter(e.target.value); setPage(0); }}
          className="border rounded-lg px-3 py-1.5 text-xs text-gray-600 bg-white shadow-sm outline-none w-52"
        />
      </div>

      {error ? (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 p-4 text-sm">
          <AlertCircle size={14} />
          Failed to load violations. Is the backend running?
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Time
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Rule
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Action
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Event
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Agent
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Session
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {isLoading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}>
                  {Array.from({ length: 6 }).map((_, j) => (
                    <td key={j} className="px-4 py-3">
                      <div className="h-4 bg-gray-100 rounded animate-pulse" />
                    </td>
                  ))}
                </tr>
              ))
            ) : violations.length === 0 ? (
              <tr>
                <td colSpan={6} className="text-center py-10 text-gray-400 text-sm">
                  No violations found.
                </td>
              </tr>
            ) : (
              violations.map((v) => <ViolationRow key={v.id} v={v} />)
            )}
          </tbody>
        </table>
      )}

      {/* Pagination */}
      {(data?.total ?? 0) >= PAGE_SIZE && (
        <div className="flex items-center justify-between px-5 py-3 border-t bg-gray-50">
          <span className="text-xs text-gray-500">
            Showing {page * PAGE_SIZE + 1}–
            {page * PAGE_SIZE + violations.length}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-3 py-1 text-xs border rounded hover:bg-gray-100 disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={violations.length < PAGE_SIZE}
              className="px-3 py-1 text-xs border rounded hover:bg-gray-100 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function ViolationRow({ v }: { v: Violation }) {
  const ts = v.timestamp_ns
    ? formatDistanceToNow(new Date(v.timestamp_ns / 1_000_000), {
        addSuffix: true,
      })
    : "—";

  return (
    <tr className="hover:bg-gray-50 transition-colors">
      <td className="px-4 py-2.5 text-xs text-gray-500 whitespace-nowrap">{ts}</td>
      <td className="px-4 py-2.5 font-mono text-xs text-gray-800">{v.rule_name}</td>
      <td className="px-4 py-2.5">
        <SeverityBadge action={v.action} />
      </td>
      <td className="px-4 py-2.5 text-xs text-gray-600">{v.event_type}</td>
      <td className="px-4 py-2.5 text-xs text-gray-600">{v.agent_id}</td>
      <td className="px-4 py-2.5 font-mono text-xs text-gray-500">
        {v.session_id ? v.session_id.slice(0, 16) + "..." : "—"}
      </td>
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Threat Intelligence panel  (Phase 3.2 security layer counters)
// ---------------------------------------------------------------------------

// Security-specific rule names produced by the 6-layer pipeline
const SECURITY_RULES: Array<{
  rule: string;
  label: string;
  layer: string;
  description: string;
}> = [
  {
    rule: "prompt-injection-detected",
    label: "Injection",
    layer: "Layers 0-4",
    description: "Prompt injection attempts detected by the 6-layer pipeline",
  },
  {
    rule: "crescendo-drift-detection",
    label: "Crescendo",
    layer: "Layer 3",
    description: "Multi-turn semantic drift — Crescendo jailbreak pattern",
  },
  {
    rule: "credential-in-prompt",
    label: "Credential Leak",
    layer: "Credential",
    description: "API keys / secrets detected in LLM input messages",
  },
  {
    rule: "output-threat-detected",
    label: "Output Threat",
    layer: "Output",
    description: "Canary leak / relay injection / credentials in LLM response",
  },
  {
    rule: "rce-in-tool-args",
    label: "RCE",
    layer: "Tool",
    description: "Remote code execution attempt in tool arguments",
  },
  {
    rule: "ssrf-in-tool-args",
    label: "SSRF",
    layer: "Tool",
    description: "Server-side request forgery in tool arguments",
  },
];

function MemoryGuardCard() {
  const { data: overview } = useQuery({
    queryKey: ["metricsOverview"],
    queryFn: getMetricsOverview,
    refetchInterval: 30_000,
  });

  const count = overview?.memory_blocked_count ?? 0;
  const cardColor = count > 0
    ? "border-red-300 bg-red-50"
    : "border-gray-200 bg-gray-50";
  const countColor = count > 0 ? "text-red-700" : "text-gray-400";

  return (
    <div
      title="Memory Commit Validator — L6 Memory Guard. Blocks injected instructions before they're written to vector stores."
      className={`rounded-lg border p-3 flex flex-col gap-1 cursor-default ${cardColor}`}
    >
      <div className={`text-2xl font-bold ${countColor}`}>{count}</div>
      <div className="text-sm font-semibold text-gray-800">Memory Guard</div>
      <div className="text-xs text-gray-500">L6 Memory Guard</div>
      {count === 0 && (
        <div className="text-xs text-green-600 mt-1 font-medium">No threats</div>
      )}
    </div>
  );
}

function ThreatIntelligencePanel() {
  const { data: summaryData, isLoading } = useQuery({
    queryKey: ["violationsSummary"],
    queryFn: getViolationsSummary,
    refetchInterval: 30_000,
  });

  // Build lookup: rule_name -> {count, action, last_fired_ns}
  const ruleMap = new Map<
    string,
    { count: number; action: string; last_fired_ns: number }
  >();
  for (const s of summaryData?.summary ?? []) {
    const existing = ruleMap.get(s.rule_name);
    if (!existing || s.count > existing.count) {
      ruleMap.set(s.rule_name, {
        count: s.count,
        action: s.action,
        last_fired_ns: s.last_fired_ns,
      });
    }
  }

  const totalThreats = SECURITY_RULES.reduce(
    (sum, r) => sum + (ruleMap.get(r.rule)?.count ?? 0),
    0
  );

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
      <div className="px-6 py-4 border-b border-gray-200 flex items-center gap-3">
        <ShieldAlert size={18} className="text-red-500" />
        <div>
          <h2 className="text-base font-semibold text-gray-900">Threat Intelligence</h2>
          <p className="text-xs text-gray-500 mt-0.5">
            Detections from the 6-layer security pipeline (Phase 3.2)
          </p>
        </div>
        {totalThreats > 0 && (
          <span className="ml-auto px-3 py-1 rounded-full text-sm font-bold bg-red-100 text-red-700">
            {totalThreats} total threat{totalThreats !== 1 ? "s" : ""}
          </span>
        )}
      </div>
      <div className="p-6">
        {isLoading ? (
          <div className="text-sm text-gray-400">Loading...</div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-7 gap-3">
            {SECURITY_RULES.map(({ rule, label, layer, description }) => {
              const entry = ruleMap.get(rule);
              const count = entry?.count ?? 0;
              const action = entry?.action ?? "";
              const lastFired = entry?.last_fired_ns
                ? new Date(entry.last_fired_ns / 1_000_000)
                : null;

              const cardColor =
                count === 0
                  ? "border-gray-200 bg-gray-50"
                  : action === "BLOCK"
                  ? "border-red-300 bg-red-50"
                  : "border-amber-300 bg-amber-50";

              const countColor =
                count === 0
                  ? "text-gray-400"
                  : action === "BLOCK"
                  ? "text-red-700"
                  : "text-amber-700";

              return (
                <div
                  key={rule}
                  title={description}
                  className={`rounded-lg border p-3 flex flex-col gap-1 cursor-default ${cardColor}`}
                >
                  <div className={`text-2xl font-bold ${countColor}`}>{count}</div>
                  <div className="text-sm font-semibold text-gray-800">{label}</div>
                  <div className="text-xs text-gray-500">{layer}</div>
                  {lastFired && (
                    <div className="text-xs text-gray-400 mt-1">
                      {formatDistanceToNow(lastFired, { addSuffix: true })}
                    </div>
                  )}
                  {count === 0 && (
                    <div className="text-xs text-green-600 mt-1 font-medium">No threats</div>
                  )}
                </div>
              );
            })}
            <MemoryGuardCard />
          </div>
        )}

        {/* Security layer status row */}
        <div className="mt-5 pt-4 border-t border-gray-100">
          <div className="text-xs font-medium text-gray-500 mb-2 uppercase tracking-wide">
            Active Defense Layers
          </div>
          <div className="flex flex-wrap gap-2">
            {[
              { name: "L0 Normalizer",   tip: "Base64/Hex/ROT13/FlipAttack decoder" },
              { name: "L1 Structural",   tip: "LLM delimiter + RTL override detection" },
              { name: "L2 Semantic",     tip: "55+ attack vector cosine similarity" },
              { name: "L3 Coherence",    tip: "Crescendo multi-turn drift tracking" },
              { name: "L4 DeBERTa",      tip: "Fine-tuned classifier, 88-99% accuracy" },
              { name: "Canary",          tip: "256-bit exfiltration token, zero FP" },
              { name: "Spotlighting",    tip: "IPI defense — randomized delimiters" },
              { name: "Output Scanner",  tip: "Post-response canary + relay detection" },
              { name: "RCE Scanner",     tip: "Python AST + shell + SQL analysis" },
              { name: "SSRF Scanner",    tip: "Private IP + cloud metadata blocking" },
              { name: "L6 Memory Guard", tip: "Vector store write scanner — blocks injected instructions before commit" },
            ].map(({ name, tip }) => (
              <span
                key={name}
                title={tip}
                className="px-2 py-1 rounded text-xs bg-green-100 text-green-700 border border-green-200 font-medium cursor-default"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function Security() {
  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Security</h1>
          <p className="text-sm text-gray-500 mt-1">
            Threat intelligence, tool permissions, and policy violation monitoring
          </p>
        </div>
      </div>

      {/* Threat intelligence — Phase 3.2 security pipeline counters */}
      <ThreatIntelligencePanel />

      {/* Tool permissions */}
      <ToolPermissionsPanel />

      {/* Bottom row: summary + recent violations */}
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(0,2fr)] gap-6">
        <ViolationSummaryPanel />
        <RecentViolationsPanel />
      </div>
    </div>
  );
}
