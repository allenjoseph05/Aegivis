import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  XCircle,
  AlertTriangle,
  Filter,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { listViolations } from "../../api/client";
import type { Violation } from "../../api/client";

const PAGE_SIZE = 50;

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

function timeAgo(ns: number): string {
  const diff = Date.now() - ns / 1_000_000;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(ns / 1_000_000).toLocaleDateString();
}

export function ViolationsPage() {
  const [filterAction, setFilterAction] = useState<"" | "BLOCK" | "ALERT">("");
  const [filterAgent, setFilterAgent] = useState("");
  const [filterRule, setFilterRule] = useState("");
  const [page, setPage] = useState(0);

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

  const violations = data?.violations ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const blockCount = violations.filter((v) => v.action === "BLOCK").length;
  const alertCount = violations.filter((v) => v.action === "ALERT").length;

  // Collect unique agents + rules from current results for filter hints
  const knownAgents = [...new Set(violations.map((v) => v.agent_id))];
  const knownRules = [...new Set(violations.map((v) => v.rule_name))];

  function resetFilters() {
    setFilterAction("");
    setFilterAgent("");
    setFilterRule("");
    setPage(0);
  }

  const hasFilters = filterAction || filterAgent || filterRule;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Violations</h1>
          <p className="text-sm text-gray-500 mt-1">
            {total > 0 ? `${total} total · ${blockCount} blocked · ${alertCount} alerts` : "No violations recorded"}
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

      {/* Filter bar */}
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

        {/* Agent filter */}
        <select
          value={filterAgent}
          onChange={(e) => { setFilterAgent(e.target.value); setPage(0); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 text-gray-600 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All agents</option>
          {knownAgents.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>

        {/* Rule filter */}
        <select
          value={filterRule}
          onChange={(e) => { setFilterRule(e.target.value); setPage(0); }}
          className="text-sm border border-gray-200 rounded-lg px-3 py-1.5 text-gray-600 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All rules</option>
          {knownRules.map((r) => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>

        {hasFilters && (
          <button
            onClick={resetFilters}
            className="text-xs text-blue-600 hover:text-blue-700 ml-auto"
          >
            Clear filters
          </button>
        )}
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
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Action</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Rule</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Agent</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Event type</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">When</th>
                <th className="text-left px-4 py-3 font-medium text-gray-500 text-xs uppercase tracking-wide">Session</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {violations.map((v) => (
                <ViolationRow key={v.id} violation={v} />
              ))}
              {isFetching && violations.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-gray-400">
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
            <span className="text-sm text-gray-600">
              {page + 1} / {totalPages}
            </span>
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

function ViolationRow({ violation: v }: { violation: Violation }) {
  return (
    <tr className="hover:bg-gray-50 transition-colors">
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
          className="font-mono text-xs text-blue-600 hover:text-blue-700 hover:underline truncate max-w-[120px] block"
        >
          {v.session_id.slice(0, 16)}…
        </Link>
      </td>
    </tr>
  );
}
