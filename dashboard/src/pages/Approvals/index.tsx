import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle, XCircle, Clock, ShieldAlert, Terminal,
  Network, List, ChevronDown, ChevronRight,
} from "lucide-react";
import {
  listApprovals, approveRequest, denyRequest,
  type Approval,
} from "../../api/client";

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatCountdown(expiresAt: string): string {
  const diff = Math.max(0, Math.floor((new Date(expiresAt).getTime() - Date.now()) / 1000));
  if (diff === 0) return "Expired";
  const m = Math.floor(diff / 60);
  const s = diff % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const TRIGGER_LABELS: Record<string, { label: string; color: string; Icon: typeof ShieldAlert }> = {
  hitl_tool_list:       { label: "Required Tool",    color: "blue",   Icon: List },
  hitl_score_threshold: { label: "High Risk Score",  color: "red",    Icon: ShieldAlert },
  hitl_network_sink:    { label: "Network Sink",     color: "orange", Icon: Network },
};

const STATUS_BADGE: Record<string, string> = {
  pending:  "bg-yellow-50 text-yellow-800 border border-yellow-200",
  approved: "bg-green-50  text-green-800  border border-green-200",
  denied:   "bg-red-50    text-red-800    border border-red-200",
  expired:  "bg-gray-100  text-gray-500   border border-gray-200",
};

// ─── Approval Card ────────────────────────────────────────────────────────────

function ApprovalCard({ approval, onDecision }: {
  approval: Approval;
  onDecision: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [denyNote, setDenyNote] = useState("");
  const [showDenyForm, setShowDenyForm] = useState(false);
  const [countdown, setCountdown] = useState(formatCountdown(approval.expires_at));
  const qc = useQueryClient();

  // Live countdown
  useState(() => {
    const iv = setInterval(() => setCountdown(formatCountdown(approval.expires_at)), 1000);
    return () => clearInterval(iv);
  });

  const approveMut = useMutation({
    mutationFn: () => approveRequest(approval.id),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["approvals"] }); onDecision(); },
  });

  const denyMut = useMutation({
    mutationFn: () => denyRequest(approval.id, denyNote || undefined),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["approvals"] }); onDecision(); },
  });

  const trigger = TRIGGER_LABELS[approval.trigger] ?? {
    label: approval.trigger, color: "gray", Icon: ShieldAlert,
  };
  const TriggerIcon = trigger.Icon;

  const hasArgs = approval.tool_args && Object.keys(approval.tool_args).length > 0;
  const isExpired = countdown === "Expired";

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 flex items-start gap-4">
        <div className="p-2 bg-gray-100 rounded-lg flex-shrink-0 mt-0.5">
          <Terminal size={16} className="text-gray-600" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-mono font-semibold text-gray-900 text-sm">{approval.tool_name}</span>
            <span className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full font-medium bg-${trigger.color}-50 text-${trigger.color}-700 border border-${trigger.color}-200`}>
              <TriggerIcon size={11} />
              {trigger.label}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-0.5">
            Agent: <span className="font-mono">{approval.agent_id}</span>
            {" · "}Session: <span className="font-mono">{approval.session_id.slice(0, 16)}…</span>
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs text-gray-500 flex-shrink-0">
          <Clock size={12} />
          <span className={countdown === "Expired" ? "text-red-500 font-medium" : ""}>
            {countdown}
          </span>
        </div>
      </div>

      {/* Collapsible args */}
      {hasArgs && (
        <div className="border-t border-gray-100">
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full px-5 py-2 flex items-center gap-2 text-xs text-gray-500 hover:bg-gray-50 transition-colors"
          >
            {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            Tool arguments
          </button>
          {expanded && (
            <pre className="px-5 pb-3 text-xs font-mono text-gray-700 bg-gray-50 overflow-x-auto">
              {JSON.stringify(approval.tool_args, null, 2)}
            </pre>
          )}
        </div>
      )}

      {/* Deny form */}
      {showDenyForm && (
        <div className="border-t border-gray-100 px-5 py-3 space-y-2">
          <label className="text-xs text-gray-600 font-medium">Reason for denial (optional)</label>
          <textarea
            value={denyNote}
            onChange={e => setDenyNote(e.target.value)}
            rows={2}
            className="w-full text-sm border border-gray-200 rounded-lg px-3 py-2 resize-none focus:outline-none focus:ring-2 focus:ring-red-200"
            placeholder="Explain why this tool call was denied…"
          />
        </div>
      )}

      {/* Action buttons */}
      <div className="border-t border-gray-100 px-5 py-3 flex items-center gap-3">
        {isExpired ? (
          <p className="text-xs text-gray-400 italic">
            This request expired — the agent was automatically denied.
          </p>
        ) : (
          <>
            <button
              onClick={() => approveMut.mutate()}
              disabled={approveMut.isPending || denyMut.isPending}
              className="flex items-center gap-1.5 px-4 py-1.5 bg-green-600 hover:bg-green-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-60"
            >
              <CheckCircle size={14} />
              {approveMut.isPending ? "Approving…" : "Approve"}
            </button>

            {!showDenyForm ? (
              <button
                onClick={() => setShowDenyForm(true)}
                disabled={approveMut.isPending}
                className="flex items-center gap-1.5 px-4 py-1.5 border border-red-300 text-red-600 hover:bg-red-50 text-sm font-medium rounded-lg transition-colors disabled:opacity-60"
              >
                <XCircle size={14} />
                Deny
              </button>
            ) : (
              <>
                <button
                  onClick={() => denyMut.mutate()}
                  disabled={denyMut.isPending}
                  className="flex items-center gap-1.5 px-4 py-1.5 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-60"
                >
                  <XCircle size={14} />
                  {denyMut.isPending ? "Denying…" : "Confirm Deny"}
                </button>
                <button
                  onClick={() => { setShowDenyForm(false); setDenyNote(""); }}
                  className="text-sm text-gray-500 hover:text-gray-700"
                >
                  Cancel
                </button>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ─── History Row ─────────────────────────────────────────────────────────────

function HistoryRow({ approval }: { approval: Approval }) {
  return (
    <div className="flex items-center gap-4 py-3 border-b border-gray-50 last:border-0">
      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${STATUS_BADGE[approval.status] ?? STATUS_BADGE.expired}`}>
        {approval.status}
      </span>
      <span className="font-mono text-sm text-gray-800 flex-1 truncate">{approval.tool_name}</span>
      <span className="text-xs text-gray-400 font-mono">{approval.agent_id}</span>
      <span className="text-xs text-gray-400">
        {new Date(approval.created_at).toLocaleString()}
      </span>
      {approval.decided_by && (
        <span className="text-xs text-gray-400">by {approval.decided_by}</span>
      )}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ApprovalsPage() {
  const [tab, setTab] = useState<"pending" | "history">("pending");
  const qc = useQueryClient();

  const pendingQ = useQuery({
    queryKey: ["approvals", "pending"],
    queryFn: () => listApprovals({ status: "pending" }),
    refetchInterval: 5_000,
    enabled: tab === "pending",
  });

  const historyQ = useQuery({
    queryKey: ["approvals", "history"],
    queryFn: () => listApprovals({ status: "all", limit: 50 }),
    enabled: tab === "history",
  });

  const pending = pendingQ.data?.approvals ?? [];
  const history = historyQ.data?.approvals ?? [];

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">HITL Approvals</h1>
        <p className="text-sm text-gray-500 mt-1">
          Review and approve or deny tool calls that require human authorisation before the agent can proceed.
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-gray-200">
        {(["pending", "history"] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium capitalize border-b-2 transition-colors ${
              tab === t
                ? "border-blue-600 text-blue-700"
                : "border-transparent text-gray-500 hover:text-gray-700"
            }`}
          >
            {t}
            {t === "pending" && (pendingQ.data?.count ?? 0) > 0 && (
              <span className="ml-2 bg-red-500 text-white text-xs font-bold rounded-full px-1.5 py-0.5 leading-none">
                {pendingQ.data?.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Pending tab */}
      {tab === "pending" && (
        <div className="space-y-4">
          {pendingQ.isLoading && (
            <p className="text-sm text-gray-400 text-center py-8">Loading…</p>
          )}
          {!pendingQ.isLoading && pending.length === 0 && (
            <div className="text-center py-12 text-gray-400">
              <CheckCircle size={36} className="mx-auto mb-3 text-green-400" />
              <p className="font-medium text-gray-600">No approvals pending</p>
              <p className="text-sm mt-1">All tool calls are proceeding normally.</p>
            </div>
          )}
          {pending.map(a => (
            <ApprovalCard
              key={a.id}
              approval={a}
              onDecision={() => qc.invalidateQueries({ queryKey: ["approvals"] })}
            />
          ))}
        </div>
      )}

      {/* History tab */}
      {tab === "history" && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          {historyQ.isLoading && (
            <p className="text-sm text-gray-400 text-center py-8">Loading…</p>
          )}
          {!historyQ.isLoading && history.length === 0 && (
            <p className="text-sm text-gray-400 text-center py-8">No approval history yet.</p>
          )}
          <div className="px-5">
            {history.map(a => (
              <HistoryRow key={a.id} approval={a} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
