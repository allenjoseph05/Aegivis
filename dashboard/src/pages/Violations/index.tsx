import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  XCircle, AlertTriangle, Filter, RefreshCw,
  ChevronLeft, ChevronRight, ThumbsDown, ThumbsUp,
  X, ExternalLink, Layers, CheckSquare,
} from "lucide-react";
import {
  listViolations, getViolationContext, markFalsePositive,
  unmarkFalsePositive, countViolations,
} from "../../api/client";
import type { Violation } from "../../api/client";

const PAGE_SIZE = 50;

type GroupBy = "none" | "agent" | "rule" | "session";

// ─── helpers ──────────────────────────────────────────────────────────────────

function timeAgo(ns: number): string {
  const diff = Date.now() - ns / 1_000_000;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(ns / 1_000_000).toLocaleDateString();
}

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

// ─── Detail Drawer ─────────────────────────────────────────────────────────────

function ViolationDrawer({
  violationId,
  onClose,
}: {
  violationId: number;
  onClose: () => void;
}) {
  const qc = useQueryClient();

  const { data, isLoading, isError } = useQuery({
    queryKey: ["violation-context", violationId],
    queryFn: () => getViolationContext(violationId),
  });

  const fpMut = useMutation({
    mutationFn: (isFp: boolean) =>
      isFp ? markFalsePositive(violationId) : unmarkFalsePositive(violationId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["violation-context", violationId] });
      qc.invalidateQueries({ queryKey: ["violations"] });
    },
  });

  const v = data?.violation;
  const isFp = v?.is_false_positive ?? false;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-lg bg-white shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="font-semibold text-gray-900">Violation Detail</h2>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-gray-100 text-gray-500 hover:text-gray-700"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          {isLoading && (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <RefreshCw size={14} className="animate-spin" /> Loading…
            </div>
          )}
          {isError && (
            <p className="text-sm text-red-500">Failed to load violation detail.</p>
          )}
          {v && (
            <>
              {/* Action + FP badge */}
              <div className="flex items-center gap-3">
                <ActionBadge action={v.action} />
                {isFp && (
                  <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-500 font-medium">
                    False positive
                  </span>
                )}
              </div>

              {/* Core fields */}
              <div className="space-y-3">
                <Field label="Rule" value={v.rule_name} mono />
                <Field label="Agent" value={v.agent_id} mono />
                <Field label="Event type" value={v.event_type} />
                <Field label="When" value={timeAgo(v.timestamp_ns)} />
              </div>

              {/* Reason */}
              <div>
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-1">
                  Reason
                </p>
                <div className="bg-gray-50 border border-gray-100 rounded-lg px-3 py-2 text-sm text-gray-700">
                  {v.reason}
                </div>
              </div>

              {/* Session link */}
              <div className="flex items-center gap-2">
                <Link
                  to={`/sessions/${v.session_id}`}
                  className="text-sm text-blue-600 hover:underline flex items-center gap-1"
                >
                  <ExternalLink size={12} />
                  View session
                </Link>
                <span className="font-mono text-xs text-gray-400">
                  {v.session_id.slice(0, 20)}…
                </span>
              </div>

              {/* Other violations in same session */}
              {(data?.session_violations.length ?? 0) > 0 && (
                <div>
                  <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1">
                    <Layers size={11} />
                    Other violations this session ({data!.session_violation_count})
                  </p>
                  <div className="space-y-1">
                    {data!.session_violations.slice(0, 8).map((sv) => (
                      <div
                        key={sv.id}
                        className="flex items-center gap-2 text-xs py-1 border-b border-gray-50 last:border-0"
                      >
                        <ActionBadge action={sv.action} />
                        <span className="font-mono text-gray-700 truncate flex-1">
                          {sv.rule_name}
                        </span>
                        <span className="text-gray-400 flex-shrink-0">{timeAgo(sv.timestamp_ns)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* FP toggle */}
              <div className="pt-2 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-2">
                  Mark this violation as a false positive to flag it for threshold tuning.
                </p>
                <button
                  onClick={() => fpMut.mutate(!isFp)}
                  disabled={fpMut.isPending}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors disabled:opacity-60 ${
                    isFp
                      ? "bg-gray-100 text-gray-600 hover:bg-gray-200"
                      : "bg-amber-50 text-amber-700 hover:bg-amber-100 border border-amber-200"
                  }`}
                >
                  {isFp ? (
                    <><ThumbsUp size={12} /> Unmark false positive</>
                  ) : (
                    <><ThumbsDown size={12} /> Mark as false positive</>
                  )}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-xs text-gray-500 flex-shrink-0">{label}</span>
      <span className={`text-sm text-gray-800 truncate ${mono ? "font-mono text-xs" : ""}`}>
        {value}
      </span>
    </div>
  );
}

// ─── Group header ──────────────────────────────────────────────────────────────

function GroupHeader({ label, count }: { label: string; count: number }) {
  return (
    <tr>
      <td
        colSpan={8}
        className="px-4 py-2 bg-gray-100 text-xs font-semibold text-gray-600 uppercase tracking-wide"
      >
        {label}{" "}
        <span className="font-normal text-gray-400 normal-case">({count})</span>
      </td>
    </tr>
  );
}

// ─── Violation row ─────────────────────────────────────────────────────────────

function ViolationRow({
  violation: v,
  onClick,
  checked,
  onToggle,
}: {
  violation: Violation;
  onClick: () => void;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <tr
      className="hover:bg-blue-50/30 transition-colors cursor-pointer"
      onClick={onClick}
    >
      <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          checked={checked}
          onChange={onToggle}
          className="accent-blue-600 cursor-pointer"
        />
      </td>
      <td className="px-4 py-3">
        <ActionBadge action={v.action} />
      </td>
      <td className="px-4 py-3 font-mono text-xs text-gray-700">{v.rule_name}</td>
      <td className="px-4 py-3">
        <span className="font-mono text-xs text-gray-600">{v.agent_id}</span>
      </td>
      <td className="px-4 py-3 text-xs text-gray-500">{v.event_type}</td>
      <td className="px-4 py-3 text-xs text-gray-400 whitespace-nowrap">{timeAgo(v.timestamp_ns)}</td>
      <td className="px-4 py-3">
        <Link
          to={`/sessions/${v.session_id}`}
          onClick={(e) => e.stopPropagation()}
          className="font-mono text-xs text-blue-600 hover:text-blue-700 hover:underline truncate max-w-[120px] block"
        >
          {v.session_id.slice(0, 16)}…
        </Link>
      </td>
    </tr>
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export function ViolationsPage() {
  const [filterAction, setFilterAction] = useState<"" | "BLOCK" | "ALERT">("");
  const [filterAgent, setFilterAgent] = useState("");
  const [filterRule, setFilterRule] = useState("");
  const [groupBy, setGroupBy] = useState<GroupBy>("none");
  const [page, setPage] = useState(0);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [selectedViolationIds, setSelectedViolationIds] = React.useState<Set<number>>(new Set());

  const { data, isFetching, refetch } = useQuery({
    queryKey: ["violations", filterAction, filterAgent, filterRule, page],
    queryFn: () =>
      listViolations({
        action: filterAction || undefined,
        agent_id: filterAgent || undefined,
        rule_name: filterRule || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    refetchInterval: 15_000,
  });

  const countQ = useQuery({
    queryKey: ["violations-count", filterAction, filterAgent, filterRule],
    queryFn: () =>
      countViolations({
        action: filterAction || undefined,
        agent_id: filterAgent || undefined,
        rule_name: filterRule || undefined,
      }),
    staleTime: 10_000,
  });

  const violations = data?.violations ?? [];
  const total = countQ.data?.total ?? data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);
  const blockCount = violations.filter((v) => v.action === "BLOCK").length;
  const alertCount = violations.filter((v) => v.action === "ALERT").length;

  const qc = useQueryClient();

  const bulkFpMut = useMutation({
    mutationFn: () =>
      Promise.all([...selectedViolationIds].map((id) => markFalsePositive(id))),
    onSuccess: () => {
      setSelectedViolationIds(new Set());
      qc.invalidateQueries({ queryKey: ["violations"] });
      qc.invalidateQueries({ queryKey: ["violations-count"] });
    },
  });

  function toggleSelect(id: number) {
    setSelectedViolationIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  function toggleSelectAll() {
    const allIds = violations.map((v) => v.id);
    const allSelected = allIds.every((id) => selectedViolationIds.has(id));
    if (allSelected) {
      setSelectedViolationIds(new Set());
    } else {
      setSelectedViolationIds(new Set(allIds));
    }
  }

  const knownAgents = [...new Set(violations.map((v) => v.agent_id))];
  const knownRules = [...new Set(violations.map((v) => v.rule_name))];

  function resetFilters() {
    setFilterAction("");
    setFilterAgent("");
    setFilterRule("");
    setPage(0);
  }

  const hasFilters = filterAction || filterAgent || filterRule;

  // Group violations
  function groupedRows(): Array<{ groupLabel?: string; violation: Violation }> {
    if (groupBy === "none") return violations.map((v) => ({ violation: v }));

    const groups = new Map<string, Violation[]>();
    for (const v of violations) {
      const key =
        groupBy === "agent" ? v.agent_id :
        groupBy === "rule" ? v.rule_name :
        v.session_id;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(v);
    }

    const rows: Array<{ groupLabel?: string; violation: Violation }> = [];
    for (const [key, vs] of groups) {
      rows.push({ groupLabel: `${key} (${vs.length})`, violation: vs[0] });
      for (let i = 1; i < vs.length; i++) rows.push({ violation: vs[i] });
    }
    return rows;
  }

  const rows = groupedRows();

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Drawer */}
      {selectedId !== null && (
        <ViolationDrawer violationId={selectedId} onClose={() => setSelectedId(null)} />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Violations</h1>
          <p className="text-sm text-gray-500 mt-1">
            {total > 0
              ? `${total} total · ${blockCount} blocked · ${alertCount} alerts on this page`
              : "No violations recorded"}
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
        >
          <RefreshCw size={14} className={isFetching ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Filter + group bar */}
      <div className="bg-white border border-gray-200 rounded-xl p-4 flex flex-wrap gap-3 items-center shadow-sm">
        <Filter size={16} className="text-gray-400" />

        {/* Action filter */}
        <div className="flex gap-1">
          {(["", "BLOCK", "ALERT"] as const).map((a) => (
            <button
              key={a}
              onClick={() => { setFilterAction(a); setPage(0); }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filterAction === a
                  ? a === "BLOCK"
                    ? "bg-red-100 text-red-700"
                    : a === "ALERT"
                    ? "bg-amber-100 text-amber-700"
                    : "bg-gray-900 text-white"
                  : "text-gray-500 hover:bg-gray-100"
              }`}
            >
              {a === "" ? "All" : a}
            </button>
          ))}
        </div>

        <div className="h-4 w-px bg-gray-200" />

        <select
          value={filterAgent}
          onChange={(e) => { setFilterAgent(e.target.value); setPage(0); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 text-gray-600 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All agents</option>
          {knownAgents.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>

        <select
          value={filterRule}
          onChange={(e) => { setFilterRule(e.target.value); setPage(0); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 text-gray-600 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All rules</option>
          {knownRules.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>

        <div className="h-4 w-px bg-gray-200" />

        {/* Group by */}
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <span>Group by</span>
          {(["none", "agent", "rule", "session"] as GroupBy[]).map((g) => (
            <button
              key={g}
              onClick={() => setGroupBy(g)}
              className={`px-2 py-1 rounded font-medium capitalize transition-colors ${
                groupBy === g ? "bg-gray-900 text-white" : "hover:bg-gray-100 text-gray-600"
              }`}
            >
              {g === "none" ? "None" : g}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2 ml-auto">
          {hasFilters && (
            <button
              onClick={resetFilters}
              className="text-xs text-blue-600 hover:text-blue-700"
            >
              Clear filters
            </button>
          )}
          {violations.length > 0 && (
            <button
              onClick={() => bulkFpMut.mutate()}
              disabled={bulkFpMut.isPending || selectedViolationIds.size === 0}
              title="Mark selected violations as false positives"
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-amber-700 bg-amber-50 border border-amber-200 rounded-lg hover:bg-amber-100 transition-colors disabled:opacity-60"
            >
              {bulkFpMut.isPending ? (
                <RefreshCw size={12} className="animate-spin" />
              ) : (
                <CheckSquare size={12} />
              )}
              {selectedViolationIds.size > 0
                ? `Mark ${selectedViolationIds.size} as FP`
                : "Mark selected as FP"}
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {violations.length === 0 && !isFetching ? (
          <div className="px-6 py-16 text-center">
            <XCircle size={32} className="mx-auto text-gray-300 mb-3" />
            <p className="text-gray-500">
              {hasFilters ? "No violations match the current filters" : "No violations recorded"}
            </p>
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50">
                <th className="px-4 py-3 w-10">
                  <input
                    type="checkbox"
                    checked={violations.length > 0 && violations.every((v) => selectedViolationIds.has(v.id))}
                    ref={(el) => {
                      if (el) {
                        el.indeterminate =
                          violations.some((v) => selectedViolationIds.has(v.id)) &&
                          !violations.every((v) => selectedViolationIds.has(v.id));
                      }
                    }}
                    onChange={toggleSelectAll}
                    className="accent-blue-600 cursor-pointer"
                  />
                </th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Action</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Rule</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Agent</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Event type</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">When</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Session</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {rows.map((row, i) => (
                row.groupLabel !== undefined && i > 0
                  ? <GroupHeader key={`g-${row.groupLabel}`} label={row.groupLabel.split(" (")[0]} count={parseInt(row.groupLabel.match(/\((\d+)\)/)?.[1] ?? "0")} />
                  : row.groupLabel
                  ? <GroupHeader key={`g-${row.groupLabel}`} label={row.groupLabel.split(" (")[0]} count={parseInt(row.groupLabel.match(/\((\d+)\)/)?.[1] ?? "0")} />
                  : <ViolationRow
                      key={row.violation.id}
                      violation={row.violation}
                      onClick={() => setSelectedId(row.violation.id)}
                      checked={selectedViolationIds.has(row.violation.id)}
                      onToggle={() => toggleSelect(row.violation.id)}
                    />
              ))}
              {isFetching && violations.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-gray-400">
                    <RefreshCw size={18} className="inline animate-spin mr-2" />
                    Loading…
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-500">
            Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, total)} of {total}
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="p-2 rounded-lg border border-gray-200 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="text-sm text-gray-600">{page + 1} / {totalPages}</span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="p-2 rounded-lg border border-gray-200 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
