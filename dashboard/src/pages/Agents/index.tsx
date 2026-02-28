/**
 * Agents — Phase 3 Agent Registry
 *
 * Lists all registered agents with their declared purpose and allowed tools.
 * Provides a form to register new agents.
 * Agents with no registration are visible in the topology as "unregistered".
 */
import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listAgents,
  registerAgent,
  deleteAgent,
  type Agent,
} from "../../api/client";

function ToolBadge({ tool }: { tool: string }) {
  return (
    <span className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded font-mono">
      {tool}
    </span>
  );
}

function RegisterForm({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [form, setForm] = useState({
    agent_id: "",
    name: "",
    declared_purpose: "",
    allowed_tools_raw: "",
    owner: "",
  });
  const [error, setError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: registerAgent,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["agents"] });
      onClose();
    },
    onError: (e: unknown) => {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg ?? "Registration failed");
    },
  });

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const allowed_tools = form.allowed_tools_raw
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
    mut.mutate({
      agent_id: form.agent_id.trim(),
      name: form.name.trim(),
      declared_purpose: form.declared_purpose.trim(),
      allowed_tools,
      owner: form.owner.trim() || undefined,
    });
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Register Agent</h2>
        {error && (
          <div className="mb-3 text-sm text-red-600 bg-red-50 rounded p-2">{error}</div>
        )}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Agent ID <span className="text-red-500">*</span>
            </label>
            <input
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g. research-bot-v1"
              required
              value={form.agent_id}
              onChange={(e) => setForm({ ...form, agent_id: e.target.value })}
            />
            <p className="text-xs text-gray-400 mt-0.5">
              Must match the agent_id sent in <code>x-abb-agent-id</code> header
            </p>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Display Name <span className="text-red-500">*</span>
            </label>
            <input
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g. Research Assistant"
              required
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Declared Purpose
            </label>
            <textarea
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
              rows={2}
              placeholder="What is this agent supposed to do?"
              value={form.declared_purpose}
              onChange={(e) => setForm({ ...form, declared_purpose: e.target.value })}
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Allowed Tools
            </label>
            <input
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="web_search, read_file, summarize"
              value={form.allowed_tools_raw}
              onChange={(e) => setForm({ ...form, allowed_tools_raw: e.target.value })}
            />
            <p className="text-xs text-gray-400 mt-0.5">Comma-separated tool names</p>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">Owner</label>
            <input
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="e.g. team-name or email"
              value={form.owner}
              onChange={(e) => setForm({ ...form, owner: e.target.value })}
            />
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={mut.isPending}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded font-medium disabled:opacity-60"
            >
              {mut.isPending ? "Registering…" : "Register Agent"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

function AgentCard({ agent, onDelete }: { agent: Agent; onDelete: () => void }) {
  const [confirming, setConfirming] = useState(false);

  return (
    <div className="border border-gray-200 rounded-lg bg-white p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-gray-900 truncate">{agent.name}</h3>
            <code className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
              {agent.agent_id}
            </code>
            {agent.owner && (
              <span className="text-xs text-gray-400">({agent.owner})</span>
            )}
          </div>

          {agent.declared_purpose && (
            <p className="text-sm text-gray-500 mt-1 line-clamp-2">{agent.declared_purpose}</p>
          )}

          <div className="mt-2">
            {agent.allowed_tools.length > 0 ? (
              <div className="flex flex-wrap gap-1">
                {agent.allowed_tools.map((t) => (
                  <ToolBadge key={t} tool={t} />
                ))}
              </div>
            ) : (
              <span className="text-xs text-gray-400 italic">No tool restrictions</span>
            )}
          </div>
        </div>

        <div className="flex-shrink-0 text-right">
          <p className="text-xs text-gray-400">
            {new Date(agent.registered_at).toLocaleDateString()}
          </p>
          {confirming ? (
            <div className="mt-1 flex gap-1">
              <button
                onClick={onDelete}
                className="text-xs text-red-600 hover:text-red-800 font-medium"
              >
                Confirm
              </button>
              <span className="text-gray-300">|</span>
              <button
                onClick={() => setConfirming(false)}
                className="text-xs text-gray-500 hover:text-gray-700"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirming(true)}
              className="mt-1 text-xs text-gray-400 hover:text-red-500 transition-colors"
            >
              Remove
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function AgentsPage() {
  const qc = useQueryClient();
  const [showForm, setShowForm] = useState(false);
  const [search, setSearch] = useState("");

  const agentsQ = useQuery({
    queryKey: ["agents"],
    queryFn: listAgents,
    staleTime: 15_000,
  });

  const deleteMut = useMutation({
    mutationFn: deleteAgent,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agents"] }),
  });

  const agents: Agent[] = agentsQ.data?.agents ?? [];
  const filtered = search
    ? agents.filter(
        (a) =>
          a.name.toLowerCase().includes(search.toLowerCase()) ||
          a.agent_id.toLowerCase().includes(search.toLowerCase()) ||
          a.declared_purpose.toLowerCase().includes(search.toLowerCase())
      )
    : agents;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {showForm && <RegisterForm onClose={() => setShowForm(false)} />}

      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Agent Registry</h1>
          <p className="text-sm text-gray-500 mt-1">
            Register agents to track their declared purpose and allowed tools.
            Unregistered agents appear with a warning in the Topology view.
          </p>
        </div>
        <button
          onClick={() => setShowForm(true)}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded font-medium"
        >
          + Register Agent
        </button>
      </div>

      {/* Stats bar */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-gray-900">{agents.length}</div>
          <div className="text-xs text-gray-500 mt-0.5">Registered</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-gray-900">
            {agents.filter((a) => a.allowed_tools.length > 0).length}
          </div>
          <div className="text-xs text-gray-500 mt-0.5">With Tool Scope</div>
        </div>
        <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-gray-900">
            {agents.filter((a) => !a.owner).length}
          </div>
          <div className="text-xs text-gray-500 mt-0.5">No Owner Set</div>
        </div>
      </div>

      {/* Search */}
      {agents.length > 0 && (
        <div className="mb-4">
          <input
            className="w-full border border-gray-200 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="Search agents by name, ID, or purpose…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      )}

      {/* Agent list */}
      {agentsQ.isError ? (
        <div className="text-sm text-red-500 bg-red-50 rounded p-3">
          Failed to load agents. Is the backend running?
        </div>
      ) : filtered.length === 0 && !agentsQ.isFetching ? (
        <div className="text-center py-12 text-gray-400">
          {agents.length === 0 ? (
            <>
              <div className="text-4xl mb-3">🤖</div>
              <p className="font-medium">No agents registered yet</p>
              <p className="text-sm mt-1">
                Click "Register Agent" to add your first agent to the registry.
              </p>
            </>
          ) : (
            <p>No agents match your search.</p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map((agent) => (
            <AgentCard
              key={agent.agent_id}
              agent={agent}
              onDelete={() => deleteMut.mutate(agent.agent_id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
