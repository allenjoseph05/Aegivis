import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { formatDistanceToNow } from "date-fns";
import { RefreshCw, Bot, AlertCircle } from "lucide-react";
import { listSessions, listAgents } from "../api/client";
import type { Session } from "../types/events";
import { HashBadge } from "../components/HashBadge";
import { LiveAlertFeed } from "../components/LiveAlertFeed";
import clsx from "clsx";

const PROVIDERS = ["openai", "anthropic", "google", "azure", "ollama"] as const;

function ProviderBadge({ provider }: { provider: string }) {
  const colors: Record<string, string> = {
    openai: "bg-teal-100 text-teal-700",
    anthropic: "bg-orange-100 text-orange-700",
    google: "bg-blue-100 text-blue-700",
    azure: "bg-indigo-100 text-indigo-700",
    ollama: "bg-gray-100 text-gray-700",
  };
  return (
    <span className={clsx("px-2 py-0.5 rounded text-xs font-medium", colors[provider] ?? "bg-gray-100 text-gray-700")}>
      {provider}
    </span>
  );
}

export function Sessions() {
  const [agentFilter, setAgentFilter] = useState("");
  const [providerFilter, setProviderFilter] = useState("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 50;

  const agentsQ = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    staleTime: 60_000,
  });
  const agentOptions = agentsQ.data?.agents.map((a) => a.agent_id) ?? [];

  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ["sessions", agentFilter, providerFilter, page],
    queryFn: () =>
      listSessions({
        agent_id: agentFilter || undefined,
        provider: providerFilter || undefined,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    refetchInterval: 10_000,
  });

  const sessions = data?.sessions ?? [];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Agent Sessions</h1>
          <p className="text-sm text-gray-500 mt-1">
            Forensic audit trail — every LLM call and tool invocation captured
          </p>
        </div>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <RefreshCw size={16} className={clsx(isFetching && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Real-time alert feed */}
      <LiveAlertFeed />

      {/* Filters */}
      <div className="flex gap-3 mb-4 flex-wrap">
        <div className="flex items-center gap-2 bg-white border rounded-lg px-3 py-2 shadow-sm">
          <Bot size={14} className="text-gray-400" />
          <select
            value={agentFilter}
            onChange={(e) => { setAgentFilter(e.target.value); setPage(0); }}
            className="text-sm outline-none w-48 text-gray-700 bg-white"
          >
            <option value="">All agents</option>
            {agentOptions.map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </div>
        <select
          value={providerFilter}
          onChange={(e) => { setProviderFilter(e.target.value); setPage(0); }}
          className="border rounded-lg px-3 py-2 text-sm text-gray-600 bg-white shadow-sm outline-none"
        >
          <option value="">All providers</option>
          {PROVIDERS.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </div>

      {/* Table */}
      {error ? (
        <div className="flex items-center gap-2 text-red-600 bg-red-50 border border-red-200 rounded-lg p-4">
          <AlertCircle size={16} />
          Failed to load sessions. Is the backend running?
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Session</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Agent</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Provider / Model</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Events</th>
                <th className="px-4 py-3 text-right text-xs font-semibold text-gray-500 uppercase tracking-wider">Tokens</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Last Active</th>
                <th className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">Integrity</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {isLoading ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <tr key={i}>
                    {Array.from({ length: 7 }).map((_, j) => (
                      <td key={j} className="px-4 py-3">
                        <div className="h-4 bg-gray-100 rounded animate-pulse" />
                      </td>
                    ))}
                    <td />
                  </tr>
                ))
              ) : sessions.length === 0 ? (
                <tr>
                  <td colSpan={8} className="text-center py-12 text-gray-400">
                    No sessions found. Start an agent with{" "}
                    <code className="bg-gray-100 px-1 rounded">OPENAI_BASE_URL=http://localhost:8080/openai</code>
                  </td>
                </tr>
              ) : (
                sessions.map((session) => (
                  <SessionRow key={session.session_id} session={session} />
                ))
              )}
            </tbody>
          </table>

          {/* Pagination */}
          {(data?.count ?? 0) >= PAGE_SIZE && (
            <div className="flex items-center justify-between px-4 py-3 border-t bg-gray-50">
              <span className="text-sm text-gray-500">
                Showing {page * PAGE_SIZE + 1}–{page * PAGE_SIZE + sessions.length}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-3 py-1 text-sm border rounded hover:bg-gray-100 disabled:opacity-40"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage((p) => p + 1)}
                  disabled={sessions.length < PAGE_SIZE}
                  className="px-3 py-1 text-sm border rounded hover:bg-gray-100 disabled:opacity-40"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function SessionRow({ session }: { session: Session }) {
  return (
    <tr className="hover:bg-gray-50 transition-colors">
      <td className="px-4 py-3">
        <span className="font-mono text-xs text-gray-700">
          {session.session_id.slice(0, 16)}…
        </span>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-1.5">
          <Bot size={14} className="text-gray-400" />
          <span className="text-gray-700">{session.agent_id}</span>
        </div>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <ProviderBadge provider={session.provider} />
          <span className="text-gray-500 text-xs truncate max-w-32">{session.model}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <div className="flex flex-col items-end gap-0.5">
          <span className="font-medium text-gray-700">{session.event_count}</span>
          <span className="text-xs text-gray-400">
            {session.llm_call_count}L · {session.tool_call_count}T
          </span>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <span className="text-gray-600">
          {session.total_tokens != null
            ? session.total_tokens.toLocaleString()
            : "—"}
        </span>
      </td>
      <td className="px-4 py-3 text-gray-500 text-xs">
        {session.last_event_at
          ? formatDistanceToNow(new Date(session.last_event_at), { addSuffix: true })
          : "—"}
      </td>
      <td className="px-4 py-3">
        {session.has_errors ? (
          <span className="inline-flex items-center gap-1 text-xs text-red-600">
            <AlertCircle size={12} />
            errors
          </span>
        ) : (
          <HashBadge valid={null} />
        )}
      </td>
      <td className="px-4 py-3">
        <Link
          to={`/sessions/${session.session_id}`}
          className="text-xs text-blue-600 hover:text-blue-800 font-medium"
        >
          View →
        </Link>
      </td>
    </tr>
  );
}
