import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Loader2, Wrench, AlertCircle } from "lucide-react";
import { getForensicReplay } from "../api/client";
import { HashBadge } from "../components/HashBadge";
import { RiskPanel } from "../components/RiskPanel";
import clsx from "clsx";

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}


export function Forensics() {
  const { sessionId } = useParams<{ sessionId: string }>();

  if (!sessionId) return null;

  const { data, isLoading, error } = useQuery({
    queryKey: ["forensic-replay", sessionId],
    queryFn: () => getForensicReplay(sessionId),
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64 gap-2 text-gray-500">
        <Loader2 size={20} className="animate-spin" />
        Building forensic report…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="p-6">
        <div className="flex items-center gap-2 text-red-600 bg-red-50 border border-red-200 rounded-lg p-4">
          <AlertCircle size={16} />
          Failed to load forensic report. Is the backend running?
        </div>
      </div>
    );
  }

  const { report, anomalies } = data;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Back */}
      <Link
        to={`/sessions/${sessionId}`}
        className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
      >
        <ArrowLeft size={14} />
        Back to Session
      </Link>

      {/* Summary card */}
      <div className="bg-white rounded-xl shadow-sm border p-5">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Forensic Replay</h1>
            <p className="font-mono text-xs text-gray-500 mt-1">{sessionId}</p>
          </div>
          <HashBadge
            valid={data.chain_valid}
            totalEvents={report.total_events}
            failedAt={data.chain_check.first_failed_sequence}
          />
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4 mt-4 pt-4 border-t">
          <StatCard label="Agent" value={report.agent_id} small />
          <StatCard label="Model" value={report.model} small />
          <StatCard label="Duration" value={formatMs(report.duration_ms)} />
          <StatCard label="LLM Calls" value={report.llm_call_count} />
          <StatCard label="Tool Calls" value={report.tool_call_count} />
          <StatCard label="Tokens" value={report.total_tokens.toLocaleString()} />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Timeline */}
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border p-5">
          <h2 className="font-semibold text-gray-800 mb-4">Decision Timeline</h2>
          <div className="space-y-1 max-h-[600px] overflow-y-auto pr-1">
            {report.timeline.map((entry) => (
              <TimelineRow key={entry.event_id} entry={entry} />
            ))}
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-4">
          {/* Risk flags */}
          <div className="bg-white rounded-xl shadow-sm border p-5">
            <h2 className="font-semibold text-gray-800 mb-4">
              Risk Flags
              {(report.risk_flags.length + anomalies.length) > 0 && (
                <span className="ml-2 px-1.5 py-0.5 bg-red-100 text-red-700 text-xs rounded-full">
                  {report.risk_flags.length + anomalies.length}
                </span>
              )}
            </h2>
            <RiskPanel riskFlags={report.risk_flags} anomalies={anomalies} />
          </div>

          {/* Data access map */}
          {report.data_access_map.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border p-5">
              <h2 className="font-semibold text-gray-800 mb-4">
                <span className="flex items-center gap-2">
                  <Wrench size={16} />
                  Data Access Map
                </span>
              </h2>
              <div className="space-y-2">
                {report.data_access_map.map((entry) => (
                  <div
                    key={entry.tool_name}
                    className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0"
                  >
                    <div className="flex items-center gap-2">
                      <Wrench size={12} className="text-gray-400" />
                      <span className="text-sm font-medium text-gray-700">{entry.tool_name}</span>
                    </div>
                    <span className="text-sm text-gray-500">
                      {entry.call_count}×
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, small }: { label: string; value: string | number; small?: boolean }) {
  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className={clsx("font-bold text-gray-800 mt-0.5", small ? "text-sm" : "text-xl")}>
        {value}
      </div>
    </div>
  );
}

const SEVERITY_COLORS: Record<string, string> = {
  error: "border-red-400 bg-red-50",
  success: "border-green-400 bg-green-50",
  info: "border-blue-300 bg-blue-50",
  debug: "border-gray-200 bg-gray-50",
};

function TimelineRow({ entry }: { entry: { event_type: string; label: string; severity: string; sequence_number: number; summary: string; pii_detected: string[]; timestamp_ns: number } }) {
  const color = SEVERITY_COLORS[entry.severity] ?? SEVERITY_COLORS.debug;

  return (
    <div className={clsx("border-l-4 rounded-r px-3 py-2", color)}>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-400 font-mono w-6">#{entry.sequence_number}</span>
        <span className="text-xs font-semibold text-gray-600 w-24 shrink-0">{entry.label}</span>
        <span className="text-xs text-gray-600 flex-1 min-w-0 truncate">{entry.summary}</span>
        {entry.pii_detected.length > 0 && (
          <span className="text-xs px-1.5 bg-orange-100 text-orange-700 rounded">PII</span>
        )}
      </div>
    </div>
  );
}
