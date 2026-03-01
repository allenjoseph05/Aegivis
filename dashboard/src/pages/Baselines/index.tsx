/**
 * Baselines — Tool Baseline Enforcement
 *
 * Operators use this page to review the tools each agent actually uses
 * (auto-discovered by the proxy) and approve or deny each one.
 *
 * How enforcement works:
 *   1. The proxy intercepts every LLM API call and reads the tools[] array.
 *   2. On first call of a session it records the tool set and reports it here.
 *   3. Once an operator approves at least one tool, enforcement becomes active:
 *      any tool call not in the approved set is blocked with a 403.
 *   4. Mid-session tool set changes (possible injection signal) are always
 *      blocked regardless of baseline status.
 *
 * States:
 *   Audit mode  — no tools approved yet; tools are observed but not blocked
 *   Active      — at least one tool approved; unapproved calls are blocked
 */
import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listBaselines,
  listAgentTools,
  approveTool,
  denyTool,
  approveAllTools,
  type BaselineSummary,
  type AgentTool,
} from "../../api/client";

// ─────────────────────────────────────────────────────────────────────────────
// Status badge
// ─────────────────────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending:  "bg-yellow-50 text-yellow-800 border border-yellow-200",
    approved: "bg-green-50  text-green-800  border border-green-200",
    denied:   "bg-red-50    text-red-800    border border-red-200",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${styles[status] ?? "bg-gray-50 text-gray-600"}`}>
      {status}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool detail panel for a selected agent
// ─────────────────────────────────────────────────────────────────────────────

function AgentToolPanel({ agentId, onClose }: { agentId: string; onClose: () => void }) {
  const qc = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ["baselines", agentId, "tools"],
    queryFn: () => listAgentTools(agentId),
    refetchInterval: 10_000,
  });

  const approve = useMutation({
    mutationFn: ({ toolName }: { toolName: string }) => approveTool(agentId, toolName),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["baselines", agentId, "tools"] });
      qc.invalidateQueries({ queryKey: ["baselines"] });
    },
  });

  const deny = useMutation({
    mutationFn: ({ toolName }: { toolName: string }) => denyTool(agentId, toolName),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["baselines", agentId, "tools"] });
      qc.invalidateQueries({ queryKey: ["baselines"] });
    },
  });

  const approveAll = useMutation({
    mutationFn: () => approveAllTools(agentId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["baselines", agentId, "tools"] });
      qc.invalidateQueries({ queryKey: ["baselines"] });
    },
  });

  if (isLoading) return <div className="p-6 text-gray-500 text-sm">Loading tools…</div>;
  if (error)     return <div className="p-6 text-red-600 text-sm">Failed to load tools.</div>;

  const tools: AgentTool[] = data?.tools ?? [];
  const counts  = data?.counts ?? { pending: 0, approved: 0, denied: 0, total: 0 };
  const enforcing = data?.enforcement_active ?? false;

  return (
    <div className="flex-1 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white sticky top-0 z-10">
        <div>
          <h2 className="font-semibold text-gray-900 font-mono">{agentId}</h2>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-xs text-gray-500">{counts.total} tools observed</span>
            {enforcing ? (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-blue-600 text-white">
                Enforcement active
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-gray-200 text-gray-600">
                Audit mode
              </span>
            )}
            {counts.pending > 0 && (
              <span className="px-2 py-0.5 rounded text-xs font-semibold bg-yellow-100 text-yellow-800">
                {counts.pending} pending review
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {counts.pending > 0 && (
            <button
              onClick={() => approveAll.mutate()}
              disabled={approveAll.isPending}
              className="px-3 py-1.5 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
            >
              Approve all pending ({counts.pending})
            </button>
          )}
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-sm text-gray-600 border border-gray-300 rounded hover:bg-gray-50"
          >
            Close
          </button>
        </div>
      </div>

      {/* Explanation banner */}
      {!enforcing && (
        <div className="mx-6 mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded text-sm text-yellow-800">
          <strong>Audit mode:</strong> This agent has no approved tools yet. The proxy
          observes all tool calls but does not block any. Approve at least one tool to
          activate enforcement.
        </div>
      )}

      {/* Tool table */}
      <div className="px-6 py-4">
        {tools.length === 0 ? (
          <div className="text-center py-12 text-gray-400 text-sm">
            No tools observed yet. Tools are discovered automatically when the agent makes its first API call.
          </div>
        ) : (
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b border-gray-200">
                <th className="py-2 pr-4 font-medium">Tool name</th>
                <th className="py-2 pr-4 font-medium">Description</th>
                <th className="py-2 pr-4 font-medium">Status</th>
                <th className="py-2 pr-4 font-medium text-right">Calls</th>
                <th className="py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {tools.map((tool) => {
                const desc = (tool.tool_schema as Record<string, string>)?.description ?? "";
                return (
                  <tr key={tool.tool_name} className="hover:bg-gray-50">
                    <td className="py-2.5 pr-4 font-mono text-xs text-gray-900 whitespace-nowrap">
                      {tool.tool_name}
                    </td>
                    <td className="py-2.5 pr-4 text-gray-500 text-xs max-w-xs">
                      {desc || <span className="italic text-gray-300">no description</span>}
                    </td>
                    <td className="py-2.5 pr-4">
                      <StatusBadge status={tool.status} />
                    </td>
                    <td className="py-2.5 pr-4 text-right text-gray-600 tabular-nums">
                      {tool.call_count.toLocaleString()}
                    </td>
                    <td className="py-2.5">
                      <div className="flex items-center gap-1.5">
                        {tool.status !== "approved" && (
                          <button
                            onClick={() => approve.mutate({ toolName: tool.tool_name })}
                            disabled={approve.isPending}
                            className="px-2 py-1 text-xs bg-green-50 text-green-700 border border-green-200 rounded hover:bg-green-100 disabled:opacity-50"
                          >
                            Approve
                          </button>
                        )}
                        {tool.status !== "denied" && (
                          <button
                            onClick={() => deny.mutate({ toolName: tool.tool_name })}
                            disabled={deny.isPending}
                            className="px-2 py-1 text-xs bg-red-50 text-red-700 border border-red-200 rounded hover:bg-red-100 disabled:opacity-50"
                          >
                            Deny
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main page
// ─────────────────────────────────────────────────────────────────────────────

export function BaselinesPage() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  const { data, isLoading, error } = useQuery({
    queryKey: ["baselines"],
    queryFn: listBaselines,
    refetchInterval: 15_000,
  });

  const baselines: BaselineSummary[] = data?.baselines ?? [];

  return (
    <div className="flex h-[calc(100vh-52px)]">
      {/* Left: agent list */}
      <div className="w-80 border-r border-gray-200 bg-white flex flex-col">
        <div className="px-4 py-4 border-b border-gray-200">
          <h1 className="font-semibold text-gray-900">Tool Baselines</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Review and approve tools per agent. Unapproved tools are blocked once
            enforcement is active.
          </p>
        </div>

        <div className="flex-1 overflow-auto">
          {isLoading && (
            <div className="p-4 text-sm text-gray-400">Loading…</div>
          )}
          {error && (
            <div className="p-4 text-sm text-red-600">Failed to load baselines.</div>
          )}
          {!isLoading && baselines.length === 0 && (
            <div className="p-6 text-center text-sm text-gray-400">
              No agents observed yet.
              <br />
              <span className="text-xs mt-1 block">
                Tools are discovered automatically when an agent makes its first API call through the proxy.
              </span>
            </div>
          )}
          {baselines.map((b) => (
            <button
              key={b.agent_id}
              onClick={() => setSelectedAgent(b.agent_id)}
              className={`w-full text-left px-4 py-3 border-b border-gray-100 hover:bg-gray-50 transition-colors ${
                selectedAgent === b.agent_id ? "bg-blue-50 border-l-2 border-l-blue-500" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs text-gray-900 truncate max-w-[160px]">
                  {b.agent_id}
                </span>
                {b.enforcement_active ? (
                  <span className="px-1.5 py-0.5 rounded text-xs bg-blue-100 text-blue-700">Active</span>
                ) : (
                  <span className="px-1.5 py-0.5 rounded text-xs bg-gray-100 text-gray-500">Audit</span>
                )}
              </div>
              <div className="flex items-center gap-2 mt-1">
                {b.pending > 0 && (
                  <span className="text-xs font-semibold text-yellow-700">
                    {b.pending} pending
                  </span>
                )}
                <span className="text-xs text-gray-400">
                  {b.approved} approved · {b.denied} denied
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Right: tool detail */}
      <div className="flex-1 bg-gray-50 flex flex-col">
        {selectedAgent ? (
          <AgentToolPanel
            key={selectedAgent}
            agentId={selectedAgent}
            onClose={() => setSelectedAgent(null)}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
            Select an agent to review its tools
          </div>
        )}
      </div>
    </div>
  );
}
