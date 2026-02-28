/**
 * PolicyBuilder — Dashboard Control Plane
 *
 * Three panels:
 *   1. Observed Behavior  — tool usage stats from the last 30 days
 *   2. Active Rules       — live policy rules in the proxy (toggle + save)
 *   3. Policy Suggestions — auto-generated rules from observed traffic (lazy load)
 */
import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listPolicyRules,
  listObservedTools,
  savePolicyRules,
  getPolicySuggestions,
  type PolicyRule,
  type ObservedTool,
  type PolicySuggestion,
} from "../../api/client";

const ACTION_COLORS: Record<string, string> = {
  BLOCK: "bg-red-100 text-red-800",
  ALERT: "bg-yellow-100 text-yellow-800",
  LOG: "bg-gray-100 text-gray-600",
  ALLOW: "bg-green-100 text-green-800",
  RATE_LIMIT: "bg-orange-100 text-orange-800",
};

export function PolicyBuilderPage() {
  const qc = useQueryClient();

  // ── server state ───────────────────────────────────────────────────────────
  const rulesQ = useQuery({
    queryKey: ["policy-rules"],
    queryFn: listPolicyRules,
    staleTime: 10_000,
  });

  const toolsQ = useQuery({
    queryKey: ["observed-tools"],
    queryFn: () => listObservedTools("default-org", 30),
    staleTime: 30_000,
  });

  // Suggestions are lazy — only fetched when the user clicks "Generate"
  const [suggestionsEnabled, setSuggestionsEnabled] = useState(false);
  const suggestionsQ = useQuery({
    queryKey: ["policy-suggestions"],
    queryFn: () => getPolicySuggestions("default-org", 30),
    enabled: suggestionsEnabled,
    staleTime: 60_000,
  });

  // ── local rules edit state ─────────────────────────────────────────────────
  const [rules, setRules] = useState<PolicyRule[] | null>(null);
  const effectiveRules = rules ?? rulesQ.data?.rules ?? [];

  React.useEffect(() => {
    if (rulesQ.data && rules === null) setRules(rulesQ.data.rules);
  }, [rulesQ.data]);

  const saveMut = useMutation({
    mutationFn: savePolicyRules,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["policy-rules"] });
      setRules(null);
    },
  });

  const isDirty =
    rules !== null &&
    JSON.stringify(rules) !== JSON.stringify(rulesQ.data?.rules ?? []);

  // ── rule actions ───────────────────────────────────────────────────────────
  function toggleRule(name: string) {
    setRules((prev) =>
      (prev ?? []).map((r) =>
        r.name === name ? { ...r, enabled: !r.enabled } : r
      )
    );
  }

  /** Add a suggested rule — replaces existing rule with same name, otherwise appends */
  function addSuggestedRule(rule: PolicyRule) {
    setRules((prev) => {
      const current = prev ?? [];
      const exists = current.some((r) => r.name === rule.name);
      if (exists) return current.map((r) => (r.name === rule.name ? rule : r));
      return [...current, rule];
    });
  }

  // ── derived data ───────────────────────────────────────────────────────────
  const tools: ObservedTool[] = toolsQ.data?.tools ?? [];
  const highRiskTools = tools.filter((t) => t.violation_count >= 3);
  const suggestions: PolicySuggestion[] = suggestionsQ.data?.suggestions ?? [];

  // Track which suggestions have already been added to rules
  const addedRuleNames = new Set(effectiveRules.map((r) => r.name));

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">Policy Builder</h1>
        <p className="text-sm text-gray-500 mt-1">
          View what your agents actually do, configure rules, and apply
          auto-generated suggestions. Changes are hot-reloaded — no restart needed.
        </p>
      </div>

      {/* ── Row 1: Observed Behavior + Active Rules ─────────────────────────── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">

        {/* Observed Behavior */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-base font-semibold text-gray-800">
              Observed Behavior
              <span className="ml-2 text-xs font-normal text-gray-500">(last 30 days)</span>
            </h2>
            {toolsQ.isFetching && (
              <span className="text-xs text-gray-400">Loading…</span>
            )}
          </div>

          {toolsQ.isError ? (
            <div className="text-sm text-red-500 bg-red-50 rounded p-3">
              Failed to load tool usage data. Is the backend running?
            </div>
          ) : tools.length === 0 && !toolsQ.isFetching ? (
            <div className="text-sm text-gray-400 bg-gray-50 rounded p-4 text-center">
              No tool calls recorded yet. Run an agent through the proxy to populate this table.
            </div>
          ) : (
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Agent</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">Tool</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Calls</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Violations</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">Last Seen</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 bg-white">
                  {tools.map((t, i) => (
                    <tr key={i} className={`hover:bg-gray-50 ${t.violation_count >= 3 ? "bg-red-50" : ""}`}>
                      <td className="px-4 py-2 font-mono text-xs text-gray-600 max-w-[120px] truncate">
                        {t.agent_id}
                      </td>
                      <td className="px-4 py-2 font-medium text-gray-900">
                        {t.tool_name}
                        {t.violation_count >= 3 && (
                          <span className="ml-2 text-xs text-red-500" title="High violation rate">⚠</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums">{t.call_count}</td>
                      <td className="px-4 py-2 text-right tabular-nums">
                        {t.violation_count > 0 ? (
                          <span className="text-red-600 font-medium">{t.violation_count}</span>
                        ) : (
                          <span className="text-gray-400">—</span>
                        )}
                      </td>
                      <td className="px-4 py-2 text-right text-xs text-gray-400">
                        {t.last_seen ? new Date(t.last_seen).toLocaleDateString() : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Active Rules */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-base font-semibold text-gray-800">
              Active Rules
              <span className="ml-2 text-xs font-normal text-gray-500">
                ({effectiveRules.filter((r) => r.enabled).length} enabled /{" "}
                {effectiveRules.length} total)
              </span>
            </h2>
            <button
              disabled={!isDirty || saveMut.isPending}
              onClick={() => saveMut.mutate(effectiveRules)}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                isDirty
                  ? "bg-blue-600 hover:bg-blue-700 text-white"
                  : "bg-gray-100 text-gray-400 cursor-not-allowed"
              }`}
            >
              {saveMut.isPending ? "Saving…" : isDirty ? "Save & Reload" : "Saved"}
            </button>
          </div>

          {saveMut.isError && (
            <div className="mb-3 text-sm text-red-600 bg-red-50 rounded p-2">
              Save failed — check proxy connectivity.
            </div>
          )}
          {saveMut.isSuccess && (
            <div className="mb-3 text-sm text-green-600 bg-green-50 rounded p-2">
              Policy reloaded successfully.
            </div>
          )}

          {rulesQ.isError ? (
            <div className="text-sm text-red-500 bg-red-50 rounded p-3">
              Failed to load policy rules. Is the proxy running?
            </div>
          ) : effectiveRules.length === 0 && !rulesQ.isFetching ? (
            <div className="text-sm text-gray-400 bg-gray-50 rounded p-4 text-center">
              No policy rules loaded.
            </div>
          ) : (
            <div className="space-y-2 max-h-[480px] overflow-y-auto pr-1">
              {effectiveRules.map((rule) => (
                <div
                  key={rule.name}
                  className={`border rounded-lg p-3 transition-opacity ${
                    rule.enabled ? "border-gray-200 bg-white" : "border-gray-100 bg-gray-50 opacity-60"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="font-medium text-sm text-gray-900 truncate">
                          {rule.name}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-xs font-semibold ${
                            ACTION_COLORS[rule.action] ?? "bg-gray-100 text-gray-600"
                          }`}
                        >
                          {rule.action}
                        </span>
                        <span className="text-xs text-gray-400">
                          {rule.event_types.join(", ")}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5 truncate">{rule.reason}</p>
                      {rule.conditions.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          {rule.conditions.map((c, ci) => (
                            <code
                              key={ci}
                              className="text-xs bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded font-mono"
                            >
                              {c.field} {c.op}{" "}
                              {c.value !== undefined ? String(c.value) : ""}
                            </code>
                          ))}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => toggleRule(rule.name)}
                      title={rule.enabled ? "Disable rule" : "Enable rule"}
                      className={`relative inline-flex h-5 w-9 flex-shrink-0 rounded-full transition-colors ${
                        rule.enabled ? "bg-blue-500" : "bg-gray-300"
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform mt-0.5 ${
                          rule.enabled ? "translate-x-4" : "translate-x-0.5"
                        }`}
                      />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {/* ── Row 2: Policy Suggestions ────────────────────────────────────────── */}
      <section className="border border-gray-200 rounded-xl p-5 bg-white">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-semibold text-gray-800">
              Policy Suggestions
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Auto-generated rules based on what the proxy has observed.
              Review each suggestion before applying.
            </p>
          </div>
          <button
            onClick={() => {
              setSuggestionsEnabled(true);
              qc.invalidateQueries({ queryKey: ["policy-suggestions"] });
            }}
            disabled={suggestionsQ.isFetching}
            className="px-3 py-1.5 rounded text-sm font-medium bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {suggestionsQ.isFetching ? "Analysing…" : suggestionsEnabled ? "Refresh" : "Generate Suggestions"}
          </button>
        </div>

        {/* Not yet generated */}
        {!suggestionsEnabled && (
          <div className="text-sm text-gray-400 bg-gray-50 rounded p-4 text-center">
            Click "Generate Suggestions" to analyse your traffic and get rule recommendations.
          </div>
        )}

        {/* Loading */}
        {suggestionsEnabled && suggestionsQ.isFetching && (
          <div className="text-sm text-gray-500 bg-gray-50 rounded p-4 text-center">
            Analysing {suggestionsQ.data?.total_events_analyzed?.toLocaleString() ?? "…"} events…
          </div>
        )}

        {/* Error */}
        {suggestionsEnabled && suggestionsQ.isError && (
          <div className="text-sm text-red-500 bg-red-50 rounded p-3">
            Failed to load suggestions. Is the backend running?
          </div>
        )}

        {/* Insufficient data */}
        {suggestionsEnabled && suggestionsQ.data?.insufficient_data && (
          <div className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded p-3">
            Not enough traffic yet ({suggestionsQ.data.total_events_analyzed} events in last{" "}
            {suggestionsQ.data.window_days} days). Run some agents through the proxy and try again.
          </div>
        )}

        {/* Suggestions list */}
        {suggestionsEnabled && !suggestionsQ.isFetching && suggestionsQ.data && !suggestionsQ.data.insufficient_data && (
          <div className="space-y-3">
            {suggestions.length === 0 && (
              <div className="text-sm text-gray-400 bg-gray-50 rounded p-4 text-center">
                No suggestions — your current rules look well-tuned for the observed traffic.
              </div>
            )}

            {suggestions.map((s) => {
              const alreadyAdded = addedRuleNames.has(s.rule.name);
              return (
                <div
                  key={s.id}
                  className="flex items-start justify-between gap-4 border border-gray-200 rounded-lg p-4 bg-gray-50 hover:bg-white transition-colors"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-medium text-sm text-gray-900">{s.title}</span>
                      <span
                        className={`px-1.5 py-0.5 rounded text-xs font-semibold ${
                          ACTION_COLORS[s.rule.action] ?? "bg-gray-100 text-gray-600"
                        }`}
                      >
                        {s.rule.action}
                      </span>
                      {!s.rule.enabled && (
                        <span className="px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-500">
                          disabled by default
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-gray-600 mb-1">{s.description}</p>
                    <p className="text-xs text-gray-400 italic">{s.evidence}</p>
                    {/* Preview generated rule conditions */}
                    <div className="mt-2 flex flex-wrap gap-1">
                      {s.rule.conditions.map((c, ci) => (
                        <code
                          key={ci}
                          className="text-xs bg-white border border-gray-200 text-gray-600 px-1.5 py-0.5 rounded font-mono"
                        >
                          {c.field} {c.op}{" "}
                          {Array.isArray(c.value)
                            ? `[${(c.value as string[]).join(", ")}]`
                            : String(c.value ?? "")}
                        </code>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={() => addSuggestedRule(s.rule)}
                    disabled={alreadyAdded}
                    className={`flex-shrink-0 px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                      alreadyAdded
                        ? "bg-green-100 text-green-700 cursor-default"
                        : "bg-white border border-gray-300 hover:border-blue-400 hover:text-blue-600 text-gray-700"
                    }`}
                  >
                    {alreadyAdded ? "Added" : "+ Add Rule"}
                  </button>
                </div>
              );
            })}

            {/* High-risk tools (informational — derived from observed traffic) */}
            {highRiskTools.length > 0 && (
              <div className="border border-amber-200 rounded-lg p-4 bg-amber-50">
                <p className="text-sm font-medium text-amber-800 mb-2">
                  High-Violation Tools (informational)
                </p>
                <p className="text-xs text-amber-700 mb-3">
                  These tools appear frequently in violations. Consider tightening their
                  access via the Tool Permissions page.
                </p>
                <div className="flex flex-wrap gap-2">
                  {highRiskTools.map((t, i) => (
                    <span
                      key={i}
                      className="inline-flex items-center gap-1 px-2 py-1 bg-amber-100 border border-amber-300 rounded text-xs text-amber-800"
                    >
                      <span className="font-medium">{t.tool_name}</span>
                      <span className="text-amber-500">·</span>
                      <span>{t.violation_count} violations</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            <p className="text-xs text-gray-400 pt-1">
              {suggestionsQ.data.total_events_analyzed.toLocaleString()} events analysed
              over the last {suggestionsQ.data.window_days} days.
              After adding rules, click <strong>Save &amp; Reload</strong> to apply.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
