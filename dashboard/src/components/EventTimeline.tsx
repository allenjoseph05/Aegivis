import { useState } from "react";
import clsx from "clsx";
import { ChevronDown, ChevronRight, AlertCircle, CheckCircle, Info, Wrench, Brain, Flag } from "lucide-react";
import type { AuditEvent } from "../types/events";

const EVENT_COLORS: Record<string, string> = {
  LLM_CALL_START: "border-blue-400 bg-blue-50",
  LLM_CALL_END: "border-blue-600 bg-blue-100",
  TOOL_CALL_START: "border-amber-400 bg-amber-50",
  TOOL_CALL_END: "border-amber-600 bg-amber-100",
  AGENT_FINISH: "border-green-500 bg-green-50",
  SYSTEM_ERROR: "border-red-500 bg-red-50",
  CHECKPOINT: "border-purple-400 bg-purple-50",
};

const EVENT_ICONS: Record<string, React.ReactNode> = {
  LLM_CALL_START: <Brain size={14} className="text-blue-500" />,
  LLM_CALL_END: <Brain size={14} className="text-blue-700" />,
  TOOL_CALL_START: <Wrench size={14} className="text-amber-500" />,
  TOOL_CALL_END: <Wrench size={14} className="text-amber-700" />,
  AGENT_FINISH: <CheckCircle size={14} className="text-green-500" />,
  SYSTEM_ERROR: <AlertCircle size={14} className="text-red-500" />,
  CHECKPOINT: <Flag size={14} className="text-purple-500" />,
};

const EVENT_LABELS: Record<string, string> = {
  LLM_CALL_START: "LLM Request",
  LLM_CALL_END: "LLM Response",
  TOOL_CALL_START: "Tool Invoked",
  TOOL_CALL_END: "Tool Result",
  AGENT_FINISH: "Finished",
  SYSTEM_ERROR: "Error",
  CHECKPOINT: "Checkpoint",
};

function formatNs(ns: number): string {
  const d = new Date(ns / 1_000_000);
  const base = d.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${base}.${ms}`;
}

function EventRow({ event }: { event: AuditEvent }) {
  const [expanded, setExpanded] = useState(false);
  const et = event.event_type;
  const colorClass = EVENT_COLORS[et] ?? "border-gray-300 bg-gray-50";
  const icon = EVENT_ICONS[et] ?? <Info size={14} />;
  const label = EVENT_LABELS[et] ?? et;

  const hasPii = event.pii_detected && event.pii_detected.length > 0;

  return (
    <div className={clsx("border-l-4 rounded-r mb-1", colorClass)}>
      <button
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-black/5 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        {/* Sequence number */}
        <span className="text-xs font-mono text-gray-400 w-8 shrink-0">
          #{event.sequence_number}
        </span>

        {/* Icon */}
        <span className="shrink-0">{icon}</span>

        {/* Event type label */}
        <span className="text-xs font-semibold text-gray-700 w-28 shrink-0">{label}</span>

        {/* Summary */}
        <span className="text-xs text-gray-600 flex-1 truncate">
          {getSummary(et, event.payload as Record<string, unknown>)}
        </span>

        {/* PII badge */}
        {hasPii && (
          <span className="text-xs px-1.5 py-0.5 rounded bg-orange-100 text-orange-700 shrink-0">
            PII
          </span>
        )}

        {/* Timestamp */}
        <span className="text-xs text-gray-400 shrink-0 font-mono">
          {formatNs(event.timestamp_ns)}
        </span>

        {/* Expand chevron */}
        <span className="text-gray-400 shrink-0">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </button>

      {expanded && (
        <div className="px-4 pb-3 space-y-2">
          {/* Hash info */}
          <div className="font-mono text-xs text-gray-500">
            <span className="text-gray-400">hash: </span>
            {event.current_hash.slice(0, 16)}…
          </div>

          {/* PII types */}
          {hasPii && (
            <div className="text-xs">
              <span className="text-orange-600 font-medium">PII detected: </span>
              {event.pii_detected.join(", ")}
            </div>
          )}

          {/* Payload */}
          <details className="text-xs">
            <summary className="cursor-pointer text-gray-500 hover:text-gray-700">
              Payload (masked)
            </summary>
            <pre className="mt-1 p-2 bg-white rounded border text-gray-700 overflow-auto max-h-48 text-xs">
              {JSON.stringify(event.payload, null, 2)}
            </pre>
          </details>

          {/* Original hash proof */}
          {event.payload_hash && (
            <div className="text-xs text-gray-500">
              <span className="font-medium">Original payload SHA-256: </span>
              <span className="font-mono">{event.payload_hash.slice(0, 32)}…</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function getSummary(et: string, payload: Record<string, unknown>): string {
  if (et === "LLM_CALL_START") {
    const msgs = (payload.messages as unknown[]) ?? [];
    return `${msgs.length} messages`;
  }
  if (et === "LLM_CALL_END") {
    const tcs = (payload.tool_calls as unknown[]) ?? [];
    if (tcs.length > 0) {
      const names = tcs.map((tc: unknown) => (tc as Record<string, string>).name).join(", ");
      return `→ call: ${names}`;
    }
    const text = ((payload.response_text as string) ?? "").slice(0, 80);
    return text || `finish: ${payload.finish_reason}`;
  }
  if (et === "TOOL_CALL_START") return `${payload.tool_name}`;
  if (et === "TOOL_CALL_END") {
    const out = ((payload.tool_output_masked as string) ?? "").slice(0, 80);
    return out || `${payload.tool_name}`;
  }
  if (et === "AGENT_FINISH") {
    return ((payload.final_output as string) ?? "").slice(0, 80) || "done";
  }
  if (et === "SYSTEM_ERROR") return (payload.error_message as string) ?? "error";
  if (et === "CHECKPOINT") {
    return `merkle root: ${((payload.merkle_root as string) ?? "").slice(0, 16)}…`;
  }
  return "";
}

interface EventTimelineProps {
  events: AuditEvent[];
  loading?: boolean;
}

export function EventTimeline({ events, loading }: EventTimelineProps) {
  if (loading) {
    return (
      <div className="space-y-1">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />
        ))}
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="text-center py-12 text-gray-400">
        No events found for this session.
      </div>
    );
  }

  return (
    <div className="space-y-0.5">
      {events.map((event) => (
        <EventRow key={event.event_id} event={event} />
      ))}
    </div>
  );
}
