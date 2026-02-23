import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ArrowLeft, ShieldCheck, FileText, BarChart2, AlertTriangle } from "lucide-react";
import { getSession, getSessionEvents, verifySession } from "../api/client";
import { EventTimeline } from "../components/EventTimeline";
import { HashBadge } from "../components/HashBadge";
import clsx from "clsx";

type Tab = "timeline" | "forensics" | "compliance";

export function SessionDetail() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [tab, setTab] = useState<Tab>("timeline");
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

  const events = eventsData?.events ?? [];

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Back nav */}
      <Link
        to="/"
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
        {(["timeline", "forensics", "compliance"] as Tab[]).map((t) => (
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
            {t.charAt(0).toUpperCase() + t.slice(1)}
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
