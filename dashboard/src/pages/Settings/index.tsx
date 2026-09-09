import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, Key, AlertTriangle, CheckCircle, type LucideIcon } from "lucide-react";
import clsx from "clsx";
import {
  purgeViolations, purgeSessions, purgeBaselines,
  getMetricsOverview,
  BASE_URL, API_KEY,
} from "../../api/client";

// ─── Shared components ──────────────────────────────────────────────────────────

function Section({
  icon: Icon, title, children,
}: {
  icon: LucideIcon; title: string; children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-2">
        <Icon size={16} className="text-gray-500" />
        <h2 className="font-semibold text-gray-800">{title}</h2>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-mono text-gray-700">{value}</span>
    </div>
  );
}

// ─── Danger Zone ────────────────────────────────────────────────────────────────

function DangerAction({
  title, description, buttonLabel, onConfirm, isPending, result, requireTyped,
}: {
  title: string; description: string; buttonLabel: string;
  onConfirm: () => void; isPending: boolean; result?: string;
  requireTyped?: string;
}) {
  const [confirming, setConfirming] = useState(false);
  const [typedValue, setTypedValue] = useState("");
  const canConfirm = !requireTyped || typedValue === requireTyped;

  function handleClick() {
    if (!confirming) { setConfirming(true); return; }
    if (!canConfirm) return;
    setConfirming(false);
    setTypedValue("");
    onConfirm();
  }

  function handleCancel() {
    setConfirming(false);
    setTypedValue("");
  }

  return (
    <div className="py-4 border-b border-red-50 last:border-0">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-800">{title}</p>
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
          {result && (
            <p className="text-xs text-green-600 mt-1 flex items-center gap-1">
              <CheckCircle size={11} /> {result}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {confirming && !requireTyped && (
            <span className="text-xs text-red-600 font-medium">Are you sure?</span>
          )}
          {confirming && (
            <button onClick={handleCancel}
              className="px-3 py-1.5 text-xs text-gray-600 hover:text-gray-900 border border-gray-200 rounded">
              Cancel
            </button>
          )}
          <button onClick={handleClick} disabled={isPending || (confirming && !canConfirm)}
            className={clsx("px-3 py-1.5 text-xs font-medium rounded transition-colors disabled:opacity-60",
              confirming
                ? "bg-red-600 hover:bg-red-700 text-white"
                : "border border-red-300 text-red-600 hover:bg-red-50")}>
            {isPending ? "Running…" : confirming ? "Confirm" : buttonLabel}
          </button>
        </div>
      </div>
      {confirming && requireTyped && (
        <div className="mt-3">
          <p className="text-xs text-red-600 mb-1">
            Type <strong>{requireTyped}</strong> to confirm this irreversible action:
          </p>
          <input
            type="text"
            value={typedValue}
            onChange={(e) => setTypedValue(e.target.value)}
            placeholder={requireTyped}
            autoFocus
            className="border border-red-300 rounded px-2 py-1 text-xs w-32 focus:outline-none focus:ring-2 focus:ring-red-300"
          />
        </div>
      )}
    </div>
  );
}

// ─── Page ───────────────────────────────────────────────────────────────────────

export function SettingsPage() {
  const apiKey = API_KEY;
  const backendUrl = BASE_URL || window.location.origin;
  const qc = useQueryClient();

  const overviewQ = useQuery({
    queryKey: ["metricsOverview"],
    queryFn: getMetricsOverview,
    staleTime: 30_000,
  });

  const [purgeResults, setPurgeResults] = useState<Record<string, string>>({});

  const purgeVioMut = useMutation({
    mutationFn: purgeViolations,
    onSuccess: (d) => {
      setPurgeResults((r) => ({ ...r, violations: `${d.deleted} violations deleted` }));
      qc.invalidateQueries({ queryKey: ["violations"] });
      qc.invalidateQueries({ queryKey: ["metricsOverview"] });
    },
  });

  const purgeSesMut = useMutation({
    mutationFn: purgeSessions,
    onSuccess: (d) => {
      setPurgeResults((r) => ({
        ...r,
        sessions: `${d.sessions_deleted} sessions + ${d.events_deleted} events deleted`,
      }));
      qc.invalidateQueries({ queryKey: ["sessions"] });
      qc.invalidateQueries({ queryKey: ["metricsOverview"] });
    },
  });

  const purgeBaselinesMut = useMutation({
    mutationFn: purgeBaselines,
    onSuccess: (d) => {
      setPurgeResults((r) => ({ ...r, baselines: `${d.deleted} baseline entries deleted` }));
      qc.invalidateQueries({ queryKey: ["agent-tools"] });
    },
  });

  const overview = overviewQ.data;

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-500 mt-1">
          Global configuration for your Aegivis deployment.
        </p>
      </div>

      {/* System stats */}
      {overview && (
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Sessions", value: overview.session_count },
            { label: "Violations", value: overview.total_violations },
            { label: "Agents", value: overview.agent_count },
          ].map(({ label, value }) => (
            <div key={label} className="bg-white border border-gray-200 rounded-xl p-4 text-center">
              <div className="text-2xl font-bold text-gray-900">{value.toLocaleString()}</div>
              <div className="text-xs text-gray-500 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Connection */}
      <Section icon={Link} title="Connection">
        <Row label="Backend URL" value={backendUrl} />
        <Row label="API key" value={`${apiKey.slice(0, 8)}${"•".repeat(Math.max(0, apiKey.length - 8))}`} />
        <Row label="Organisation" value="default-org" />
      </Section>

      {/* API Keys */}
      <Section icon={Key} title="API Keys">
        <p className="text-sm text-gray-500 mb-4">
          API keys are provisioned in the database. To add a new organisation key, run:
        </p>
        <pre className="text-xs bg-gray-900 text-green-300 rounded-lg p-4 overflow-x-auto">
          {`docker exec aegivis-postgres psql -U abb -d aegivis -c \\
  "INSERT INTO api_keys (org_id, key_hash, name)
   VALUES ('your-org', encode(sha256('your-secret'::bytea),'hex'), 'Org Name');"`}
        </pre>
      </Section>

      {/* Danger Zone */}
      <div className="bg-white rounded-xl border border-red-200 shadow-sm overflow-hidden">
        <div className="px-5 py-4 border-b border-red-100 flex items-center gap-2 bg-red-50">
          <AlertTriangle size={16} className="text-red-500" />
          <h2 className="font-semibold text-red-800">Danger Zone</h2>
        </div>
        <div className="p-5">
          <p className="text-sm text-gray-500 mb-4">
            These actions are irreversible. All data is permanently deleted from the database.
          </p>
          <DangerAction
            title="Purge all violations"
            description="Deletes every policy violation record for this organisation."
            buttonLabel="Purge violations"
            onConfirm={() => purgeVioMut.mutate()}
            isPending={purgeVioMut.isPending}
            result={purgeResults.violations}
          />
          <DangerAction
            title="Purge all sessions & events"
            description="Deletes all sessions and their associated audit events. The hash chain is permanently lost."
            buttonLabel="Purge sessions"
            requireTyped="DELETE"
            onConfirm={() => purgeSesMut.mutate()}
            isPending={purgeSesMut.isPending}
            result={purgeResults.sessions}
          />
          <DangerAction
            title="Reset tool baselines"
            description="Clears all approved/denied/pending tool records. Agents start fresh."
            buttonLabel="Reset baselines"
            onConfirm={() => purgeBaselinesMut.mutate()}
            isPending={purgeBaselinesMut.isPending}
            result={purgeResults.baselines}
          />
        </div>
      </div>
    </div>
  );
}
