/**
 * LiveAlertFeed — real-time violation and anomaly alert panel.
 *
 * Connects to /ws/alerts via useWebSocket and displays a live feed.
 * Hidden when there are no messages. Collapsible with a clear button.
 */
import { useState } from "react";
import { Link } from "react-router-dom";
import { formatDistanceToNow } from "date-fns";
import { ChevronDown, ChevronUp, X, Wifi, WifiOff } from "lucide-react";
import clsx from "clsx";
import { useWebSocket, type AlertMessage } from "../hooks/useWebSocket";

function SeverityBadge({ severity }: { severity?: string }) {
  const colors: Record<string, string> = {
    critical: "bg-red-100 text-red-700 border border-red-300",
    high: "bg-orange-100 text-orange-700 border border-orange-300",
    medium: "bg-yellow-100 text-yellow-700 border border-yellow-300",
    low: "bg-blue-100 text-blue-700 border border-blue-300",
  };
  const s = (severity ?? "").toLowerCase();
  return (
    <span className={clsx("px-1.5 py-0.5 rounded text-xs font-medium", colors[s] ?? "bg-gray-100 text-gray-600")}>
      {s || "—"}
    </span>
  );
}

function TypeBadge({ type }: { type: string }) {
  const isViolation = type === "violation";
  return (
    <span
      className={clsx(
        "px-1.5 py-0.5 rounded text-xs font-bold uppercase tracking-wide",
        isViolation
          ? "bg-red-500 text-white"
          : "bg-orange-400 text-white"
      )}
    >
      {isViolation ? "VIOLATION" : "ANOMALY"}
    </span>
  );
}

function AlertRow({ msg }: { msg: AlertMessage }) {
  const ruleLabel = msg.rule_id ?? msg.rule_name ?? "unknown";
  const detail = msg.description ?? msg.reason ?? "";

  return (
    <div className="flex items-start gap-2 px-3 py-2 hover:bg-gray-50 border-b border-gray-100 last:border-0">
      <div className="flex items-center gap-1.5 min-w-0 flex-1">
        <TypeBadge type={msg.type} />
        <SeverityBadge severity={msg.severity} />
        <span className="font-mono text-xs text-gray-700 font-medium truncate">{ruleLabel}</span>
        {detail && (
          <span className="text-xs text-gray-500 truncate hidden sm:block">{detail}</span>
        )}
      </div>
      <div className="flex items-center gap-2 shrink-0 text-xs text-gray-400">
        {msg.session_id && (
          <Link
            to={`/sessions/${msg.session_id}`}
            className="font-mono text-blue-500 hover:text-blue-700 hover:underline"
            title={msg.session_id}
          >
            {msg.session_id.slice(0, 10)}...
          </Link>
        )}
        <span>{formatDistanceToNow(msg.receivedAt, { addSuffix: true })}</span>
      </div>
    </div>
  );
}

export function LiveAlertFeed() {
  const { messages, connected, clear } = useWebSocket();
  const [collapsed, setCollapsed] = useState(false);

  // Hide entirely when no messages
  if (messages.length === 0 && !connected) {
    return null;
  }

  return (
    <div className="mb-4 border rounded-xl shadow-sm overflow-hidden bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 border-b">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCollapsed((c) => !c)}
            className="flex items-center gap-1.5 text-sm font-semibold text-gray-700 hover:text-gray-900"
          >
            {collapsed ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
            Live Alerts
            {messages.length > 0 && (
              <span className="ml-1 px-1.5 py-0.5 rounded-full text-xs font-bold bg-red-100 text-red-700">
                {messages.length}
              </span>
            )}
          </button>
          <div
            className={clsx(
              "w-2 h-2 rounded-full",
              connected ? "bg-green-500" : "bg-gray-300"
            )}
            title={connected ? "WebSocket connected" : "WebSocket disconnected (reconnecting...)"}
          />
          {!connected && (
            <span className="text-xs text-gray-400 flex items-center gap-1">
              <WifiOff size={12} /> reconnecting...
            </span>
          )}
        </div>
        {messages.length > 0 && (
          <button
            onClick={clear}
            className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600"
          >
            <X size={13} /> Clear
          </button>
        )}
      </div>

      {/* Feed */}
      {!collapsed && (
        <div className="max-h-56 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="py-6 text-center text-xs text-gray-400 flex items-center justify-center gap-1.5">
              <Wifi size={14} className="text-green-500" />
              Connected — waiting for alerts...
            </div>
          ) : (
            messages.map((msg) => <AlertRow key={msg.id} msg={msg} />)
          )}
        </div>
      )}
    </div>
  );
}
