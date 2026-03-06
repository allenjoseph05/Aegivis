import { useState } from "react";
import { useParams, Link, useSearchParams } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ArrowLeft, FileText, BarChart2, AlertTriangle, Brain, GitBranch } from "lucide-react";
import { getSession, getSessionEvents, verifySession, getSessionReasoning, getSessionPDG } from "../api/client";
import type { ReasoningEntry, PDGEdge } from "../api/client";
import { EventTimeline } from "../components/EventTimeline";
import { HashBadge } from "../components/HashBadge";
import clsx from "clsx";

type Tab = "timeline" | "forensics" | "compliance" | "reasoning" | "dataflow";

const VALID_TABS: Tab[] = ["timeline", "forensics", "compliance", "reasoning", "dataflow"];

export function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();
  const rawTab = searchParams.get("tab") as Tab | null;
  const tab: Tab = rawTab && VALID_TABS.includes(rawTab) ? rawTab : "timeline";
  const setTab = (t: Tab) => setSearchParams({ tab: t }, { replace: true });
  const [verifyResult, setVerifyResult] = useState<{
    valid: boolean;
    total_events: number;
    first_failed_sequence: number | null;
  } | null>(null);

  if (!sessionId) return null;

  const { data: session, isLoading: sessionLoading } = useQuery({
    queryKey: ["session", sessionId],
    queryFn: () => getSession(sessionId),
  });

  const { data: eventsData, isLoading: eventsLoading } = useQuery({
    queryKey: ["session-events", sessionId],
    queryFn: () => getSessionEvents(sessionId, { limit: 2000 }),
  });

  const verifyMutation = useMutation({
    mutationFn: () => verifySession(sessionId),
    onSuccess: (data) => setVerifyResult(data),
  });

  const { data: reasoningData, isLoading: reasoningLoading } = useQuery({
    queryKey: ["session-reasoning", sessionId],
    queryFn: () => getSessionReasoning(sessionId),
    enabled: tab === "reasoning",
  });

  const { data: pdgData, isLoading: pdgLoading } = useQuery({
    queryKey: ["session-pdg", sessionId],
    queryFn: () => getSessionPDG(sessionId),
    enabled: tab === "dataflow",
  });

  const events = eventsData?.events ?? [];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Back nav */}
      <Link
        to="/sessions"
        className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700 mb-4"
      >
        <ArrowLeft size={14} />
        All Sessions
      </Link>

      {/* Session header */}
      <div className="bg-white rounded-xl shadow-sm border p-5 mb-6">
        {sessionLoading ? (
          <div className="space-y-2">
            <div className="h-6 bg-gray-100 rounded animate-pulse w-48" />
            <div className="h-4 bg-gray-100 rounded animate-pulse w-64" />
          </div>
        ) : (
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div>
              <h1 className="text-lg font-bold text-gray-900 font-mono">
                {sessionId}
              </h1>
              <div className="flex items-center gap-3 mt-2 text-sm text-gray-600 flex-wrap">
                <span>Agent: <strong>{session?.agent_id ?? "—"}</strong></span>
                <span>·</span>
                <span>Provider: <strong>{session?.provider ?? "—"}</strong></span>
                <span>·</span>
                <span>Model: <strong>{session?.model ?? "—"}</strong></span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <HashBadge
                valid={verifyResult?.valid ?? null}
                loading={verifyMutation.isPending}
                totalEvents={verifyResult?.total_events}
                failedAt={verifyResult?.first_failed_sequence}
                onClick={() => verifyMutation.mutate()}
              />
            </div>
          </div>
        )}

        {/* Stats row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-4 pt-4 border-t">
          <Stat label="Total Events" value={session?.event_count ?? "—"} />
          <Stat label="LLM Calls" value={session?.llm_call_count ?? "—"} />
          <Stat label="Tool Calls" value={session?.tool_call_count ?? "—"} />
          <Stat
            label="Tokens"
            value={session?.total_tokens != null
              ? Number(session.total_tokens).toLocaleString()
              : "—"}
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b mb-4">
        {(["timeline", "forensics", "compliance", "reasoning", "dataflow"] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={clsx(
              "flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px",
              tab === t
                ? "border-blue-600 text-blue-600"
                : "border-transparent text-gray-500 hover:text-gray-700"
            )}
          >
            {t === "timeline" && <BarChart2 size={15} />}
            {t === "forensics" && <AlertTriangle size={15} />}
            {t === "compliance" && <FileText size={15} />}
            {t === "reasoning" && <Brain size={15} />}
            {t === "dataflow" && <GitBranch size={15} />}
            {t === "dataflow" ? "Data Flow" : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "timeline" && (
        <EventTimeline events={events} loading={eventsLoading} />
      )}

      {tab === "forensics" && (
        <div className="bg-white rounded-xl shadow-sm border p-5">
          <p className="text-sm text-gray-500 mb-4">
            Load the full forensic replay report for anomaly analysis and risk flags.
          </p>
          <Link
            to={`/sessions/${sessionId}/forensics`}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <AlertTriangle size={15} />
            Open Forensic Replay
          </Link>
        </div>
      )}

      {tab === "compliance" && (
        <div className="bg-white rounded-xl shadow-sm border p-5">
          <Link
            to={`/sessions/${sessionId}/compliance`}
            className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <FileText size={15} />
            Generate Compliance Report
          </Link>
        </div>
      )}

      {tab === "reasoning" && (
        <ReasoningTab
          loading={reasoningLoading}
          entries={reasoningData?.reasoning ?? []}
          total={reasoningData?.total_calls_with_reasoning ?? 0}
          provider={session?.provider ?? ""}
        />
      )}

      {tab === "dataflow" && (
        <DataFlowTab
          loading={pdgLoading}
          totalEdges={pdgData?.total_edges ?? 0}
          totalSources={pdgData?.total_untrusted_sources ?? 0}
          untrustedSources={pdgData?.untrusted_sources ?? []}
          toolsWithViolations={pdgData?.tools_with_violations ?? []}
          edges={pdgData?.edges ?? []}
        />
      )}

    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-bold text-gray-800 mt-0.5">{value}</div>
    </div>
  );
}

function DataFlowTab({
  loading,
  totalEdges,
  totalSources,
  untrustedSources,
  toolsWithViolations,
  edges,
}: {
  loading: boolean;
  totalEdges: number;
  totalSources: number;
  untrustedSources: string[];
  toolsWithViolations: string[];
  edges: PDGEdge[];
}) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[0, 1].map((i) => (
          <div key={i} className="h-28 bg-gray-100 rounded-xl animate-pulse" />
        ))}
      </div>
    );
  }

  if (totalEdges === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-8 text-center">
        <GitBranch size={32} className="mx-auto text-gray-300 mb-3" />
        <p className="text-sm font-medium text-gray-700 mb-1">No data flow violations in this session</p>
        <p className="text-xs text-gray-500 max-w-md mx-auto">
          The Session PDG tracks when content from untrusted sources (web search, email, fetched URLs)
          flows into privileged tool arguments. No cross-source data flows were detected here.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border shadow-sm p-4">
          <div className="text-xs text-gray-500 mb-1">Total Edges</div>
          <div className="text-2xl font-bold text-red-600">{totalEdges}</div>
        </div>
        <div className="bg-white rounded-xl border shadow-sm p-4">
          <div className="text-xs text-gray-500 mb-1">Untrusted Sources</div>
          <div className="text-2xl font-bold text-orange-500">{totalSources}</div>
        </div>
        <div className="bg-white rounded-xl border shadow-sm p-4">
          <div className="text-xs text-gray-500 mb-1">Tools Involved</div>
          <div className="text-2xl font-bold text-gray-700">{toolsWithViolations.length}</div>
        </div>
      </div>

      {/* Untrusted sources */}
      {untrustedSources.length > 0 && (
        <div className="bg-white rounded-xl border shadow-sm p-4">
          <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5">
            <AlertTriangle size={14} className="text-orange-500" />
            Untrusted Source Fragments
          </h3>
          <div className="flex flex-wrap gap-2">
            {untrustedSources.map((src) => (
              <span
                key={src}
                className="text-xs font-mono bg-orange-50 text-orange-700 border border-orange-200 px-2 py-1 rounded"
              >
                {src.length > 60 ? src.slice(0, 60) + "…" : src}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Edge table */}
      <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
        <div className="px-4 py-3 border-b bg-gray-50">
          <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-1.5">
            <GitBranch size={14} className="text-blue-500" />
            Data Flow Edges
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b bg-gray-50 text-gray-500 text-left">
                <th className="px-4 py-2 font-medium">Source Tool</th>
                <th className="px-4 py-2 font-medium">Fragment</th>
                <th className="px-4 py-2 font-medium">→ Sink Tool</th>
                <th className="px-4 py-2 font-medium">Arg</th>
                <th className="px-4 py-2 font-medium">Hops</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {edges.map((edge, i) => (
                <tr key={i} className="hover:bg-red-50 transition-colors">
                  <td className="px-4 py-2 font-mono text-gray-600">{edge.source_tool ?? "—"}</td>
                  <td className="px-4 py-2 font-mono text-orange-700 max-w-xs truncate">
                    {edge.source_fragment
                      ? edge.source_fragment.length > 40
                        ? edge.source_fragment.slice(0, 40) + "…"
                        : edge.source_fragment
                      : "—"}
                  </td>
                  <td className="px-4 py-2 font-mono font-semibold text-red-700">{edge.sink_tool}</td>
                  <td className="px-4 py-2 font-mono text-gray-600">{edge.sink_arg ?? "—"}</td>
                  <td className="px-4 py-2 text-center text-gray-500">{edge.hop_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ReasoningProviderNote({ provider }: { provider: string }) {
  const p = provider.toLowerCase();
  if (p === "anthropic") {
    return (
      <p className="text-xs text-gray-500 max-w-lg mx-auto">
        Enable Extended Thinking by including{" "}
        <code className="bg-gray-100 px-1 rounded">
          {"\"thinking\": {\"type\": \"enabled\", \"budget_tokens\": 1000}"}
        </code>{" "}
        in your Anthropic API request.
      </p>
    );
  }
  if (p === "google") {
    return (
      <p className="text-xs text-gray-500 max-w-lg mx-auto">
        Reasoning traces are captured automatically for Gemini 2.5 Flash, Gemini 2.5 Pro, and
        Gemini 2.0 Flash Thinking — models that expose thinking blocks in their API response.
        Other Gemini models do not produce thinking traces.
      </p>
    );
  }
  if (p === "openai" || p === "azure") {
    return (
      <p className="text-xs text-gray-500 max-w-lg mx-auto">
        Reasoning traces are <strong>not available</strong> for OpenAI{p === "azure" ? " (Azure)" : ""}.
        The o1 / o3 / o4-mini internal chain-of-thought is not exposed through the Chat Completions API.
        No reasoning data can be captured for this provider.
      </p>
    );
  }
  return (
    <p className="text-xs text-gray-500 max-w-lg mx-auto">
      Reasoning traces are only captured from{" "}
      <strong>Anthropic</strong> (Extended Thinking) and{" "}
      <strong>Google Gemini</strong> (Thinking models). This provider does not expose reasoning data.
    </p>
  );
}

function ReasoningTab({
  loading,
  entries,
  total,
  provider,
}: {
  loading: boolean;
  entries: ReasoningEntry[];
  total: number;
  provider: string;
}) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  if (loading) {
    return (
      <div className="space-y-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="h-24 bg-gray-100 rounded-xl animate-pulse" />
        ))}
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-sm border p-8 text-center">
        <Brain size={32} className="mx-auto text-gray-300 mb-3" />
        <p className="text-sm font-medium text-gray-700 mb-2">No reasoning traces in this session</p>
        <ReasoningProviderNote provider={provider} />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-500">
        {total} LLM call{total !== 1 ? "s" : ""} with reasoning traces captured in this session.
      </p>
      {entries.map((entry) => (
        <ThinkingCard
          key={entry.event_id}
          entry={entry}
          expanded={!!expanded[entry.event_id]}
          onToggle={() =>
            setExpanded((prev) => ({ ...prev, [entry.event_id]: !prev[entry.event_id] }))
          }
        />
      ))}
    </div>
  );
}

const PREVIEW_CHARS = 500;

function ThinkingCard({
  entry,
  expanded,
  onToggle,
}: {
  entry: ReasoningEntry;
  expanded: boolean;
  onToggle: () => void;
}) {
  const ts = new Date(entry.timestamp_ns / 1_000_000).toLocaleTimeString();
  const block = entry.thinking_blocks[0];
  const text = block?.thinking ?? "";
  const isLong = text.length > PREVIEW_CHARS;
  const displayText = expanded || !isLong ? text : text.slice(0, PREVIEW_CHARS) + "…";

  return (
    <div className="bg-white rounded-xl shadow-sm border overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b bg-gray-50">
        <div className="flex items-center gap-2 text-sm">
          <Brain size={14} className="text-purple-500" />
          <span className="font-medium text-gray-700">{entry.model}</span>
          <span className="text-gray-400">·</span>
          <span className="text-gray-500">{ts}</span>
        </div>
        <span className="text-xs text-gray-400 font-mono">{entry.event_id}</span>
      </div>
      <div className="p-5">
        <pre className="text-xs font-mono text-gray-700 whitespace-pre-wrap break-words leading-relaxed">
          {displayText}
        </pre>
        {isLong && (
          <button
            onClick={onToggle}
            className="mt-2 text-xs text-blue-600 hover:text-blue-700 font-medium"
          >
            {expanded ? "Show less" : "Show all"}
          </button>
        )}
      </div>
    </div>
  );
}
