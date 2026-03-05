/**
 * Compliance Audit Report Panel.
 *
 * Generates an org-wide compliance audit report for a chosen date range
 * and framework, renders it inline, and provides a browser Print / PDF
 * export button.
 *
 * Frameworks: OWASP ASI 2026, EU AI Act, HIPAA §164.312, SOC 2 Type II
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getAuditReport,
  type AuditReport,
  type AuditControl,
  type ComplianceFramework,
} from "../../api/client";
import styles from "./ExportPage.module.css";

// ─── Framework options ────────────────────────────────────────────────────────

const FRAMEWORKS: { value: ComplianceFramework; label: string }[] = [
  { value: "soc2", label: "SOC 2 Type II" },
  { value: "owasp_asi_2026", label: "OWASP ASI 2026" },
  { value: "eu_ai_act", label: "EU AI Act" },
  { value: "hipaa", label: "HIPAA §164.312" },
];

// ─── Helper: format ISO date for display ──────────────────────────────────────

function fmtIso(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

// ─── Status badge ─────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: "pass" | "partial" | "fail" }) {
  const cls: Record<string, string> = {
    pass: "bg-green-100 text-green-800 border-green-200",
    partial: "bg-amber-100 text-amber-800 border-amber-200",
    fail: "bg-red-100 text-red-800 border-red-200",
  };
  const label: Record<string, string> = {
    pass: "PASS",
    partial: "PARTIAL",
    fail: "FAIL",
  };
  return (
    <span
      className={`inline-block px-2 py-0.5 rounded border text-xs font-bold tracking-wide ${cls[status]}`}
    >
      {label[status]}
    </span>
  );
}

// ─── Summary cards ────────────────────────────────────────────────────────────

function SummaryCards({ report }: { report: AuditReport }) {
  const s = report.summary;
  const cards = [
    { label: "Sessions", value: s.total_sessions.toLocaleString() },
    { label: "LLM Calls", value: s.llm_calls.toLocaleString() },
    { label: "Tool Calls", value: s.tool_calls.toLocaleString() },
    { label: "Violations", value: s.total_violations.toLocaleString() },
    { label: "Blocked", value: s.blocked_count.toLocaleString(), highlight: s.blocked_count > 0 },
    { label: "Anomalies", value: s.anomalies.toLocaleString() },
    { label: "PII Events", value: s.pii_events.toLocaleString(), highlight: s.pii_events > 0 },
    { label: "Chain Valid", value: `${s.chain_valid_pct}%`, ok: s.chain_valid_pct === 100 },
    { label: "Memory Blocked", value: s.memory_blocked.toLocaleString() },
  ];

  return (
    <div className="grid grid-cols-3 sm:grid-cols-5 xl:grid-cols-9 gap-3 mb-6 print-summary">
      {cards.map(({ label, value, highlight, ok }) => (
        <div
          key={label}
          className={`rounded-lg border p-3 text-center ${
            highlight
              ? "border-red-200 bg-red-50"
              : ok === false
              ? "border-red-200 bg-red-50"
              : ok === true
              ? "border-green-200 bg-green-50"
              : "border-gray-200 bg-gray-50"
          }`}
        >
          <div className="text-xl font-bold text-gray-900">{value}</div>
          <div className="text-xs text-gray-500 mt-0.5">{label}</div>
        </div>
      ))}
    </div>
  );
}

// ─── Controls table ───────────────────────────────────────────────────────────

function ControlsTable({ controls }: { controls: AuditControl[] }) {
  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold text-gray-900 mb-3">Compliance Controls</h2>
      <table className="w-full text-sm border rounded-lg overflow-hidden">
        <thead className="bg-gray-50 border-b">
          <tr>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide w-24">
              Control ID
            </th>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Control Name
            </th>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide w-24">
              Status
            </th>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Evidence
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {controls.map((ctrl) => (
            <tr key={ctrl.id} className="hover:bg-gray-50">
              <td className="px-4 py-3 font-mono text-xs text-gray-700 font-medium">
                {ctrl.id}
              </td>
              <td className="px-4 py-3 text-sm text-gray-800">{ctrl.name}</td>
              <td className="px-4 py-3">
                <StatusBadge status={ctrl.status} />
              </td>
              <td className="px-4 py-3 text-xs text-gray-600">{ctrl.evidence}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Violations table (top 50) ────────────────────────────────────────────────

function ViolationsTable({ report }: { report: AuditReport }) {
  const rows = report.violations.slice(0, 50);
  if (rows.length === 0) return null;

  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold text-gray-900 mb-3">
        Top Violations ({rows.length}
        {report.violations.length > 50 ? ` of ${report.violations.length}` : ""})
      </h2>
      <table className="w-full text-sm border rounded-lg overflow-hidden">
        <thead className="bg-gray-50 border-b">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Rule
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Action
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Agent
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Reason
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {rows.map((v) => (
            <tr key={v.id} className="hover:bg-gray-50">
              <td className="px-4 py-2 font-mono text-xs text-gray-800">{v.rule_name}</td>
              <td className="px-4 py-2">
                <span
                  className={`px-2 py-0.5 rounded text-xs font-semibold ${
                    v.action === "BLOCK"
                      ? "bg-red-100 text-red-700"
                      : "bg-amber-100 text-amber-700"
                  }`}
                >
                  {v.action}
                </span>
              </td>
              <td className="px-4 py-2 text-xs text-gray-600 font-mono">{v.agent_id || "—"}</td>
              <td className="px-4 py-2 text-xs text-gray-500 truncate max-w-xs">{v.reason}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Agent snapshot table ─────────────────────────────────────────────────────

function AgentsTable({ report }: { report: AuditReport }) {
  if (report.agents.length === 0) return null;

  return (
    <div className="mb-6">
      <h2 className="text-base font-semibold text-gray-900 mb-3">
        Agent Registry Snapshot ({report.agents.length})
      </h2>
      <table className="w-full text-sm border rounded-lg overflow-hidden">
        <thead className="bg-gray-50 border-b">
          <tr>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Agent ID
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Name
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Purpose
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Owner
            </th>
            <th className="px-4 py-2 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Registered
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {report.agents.map((a) => (
            <tr key={a.agent_id} className="hover:bg-gray-50">
              <td className="px-4 py-2 font-mono text-xs text-gray-800">{a.agent_id}</td>
              <td className="px-4 py-2 text-xs text-gray-700">{a.name || "—"}</td>
              <td className="px-4 py-2 text-xs text-gray-500">{a.declared_purpose || "—"}</td>
              <td className="px-4 py-2 text-xs text-gray-600">{a.owner || "—"}</td>
              <td className="px-4 py-2 text-xs text-gray-500">
                {a.registered_at ? fmtIso(a.registered_at) : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─── Rendered report ──────────────────────────────────────────────────────────

function ReportView({ report }: { report: AuditReport }) {
  const fwLabel =
    FRAMEWORKS.find((f) => f.value === report.framework)?.label ?? report.framework;

  return (
    <div id="audit-report-content" className="mt-6">
      {/* Report header */}
      <div className="flex items-start justify-between mb-4 pb-4 border-b">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            {fwLabel} Compliance Audit Report
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Organisation: <span className="font-medium">{report.org_id}</span> &nbsp;|&nbsp;
            Period: {fmtIso(report.period.from_iso)} – {fmtIso(report.period.to_iso)}
          </p>
          <p className="text-xs text-gray-400 mt-0.5">
            Generated {fmtIso(report.generated_at)} &nbsp;·&nbsp; Report ID: {report.report_id}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <StatusBadge status={report.summary.overall_status} />
          <span className="text-xs text-gray-500">Overall Status</span>
        </div>
      </div>

      <SummaryCards report={report} />
      <ControlsTable controls={report.controls} />
      <ViolationsTable report={report} />
      <AgentsTable report={report} />
    </div>
  );
}

// ─── Main panel ───────────────────────────────────────────────────────────────

export function AuditReportPanel() {
  // Default: last 30 days
  const today = new Date();
  const thirtyDaysAgo = new Date(today);
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const toStr = today.toISOString().slice(0, 10);
  const fromStr = thirtyDaysAgo.toISOString().slice(0, 10);

  const [fromDate, setFromDate] = useState(fromStr);
  const [toDate, setToDate] = useState(toStr);
  const [framework, setFramework] = useState<ComplianceFramework>("soc2");
  const [agentId, setAgentId] = useState("");
  const [orgId] = useState("default-org");

  // Only fetch when user clicks "Generate"
  const [enabled, setEnabled] = useState(false);
  const [queryKey, setQueryKey] = useState(0);

  const { data: report, isLoading, error } = useQuery({
    queryKey: ["auditReport", queryKey, orgId, fromDate, toDate, framework, agentId],
    queryFn: () =>
      getAuditReport({ from_date: fromDate, to_date: toDate, framework, agent_id: agentId || undefined }),
    enabled,
  });

  function handleGenerate() {
    setEnabled(true);
    setQueryKey((k) => k + 1);
  }

  function handlePrint() {
    window.print();
  }

  return (
    <>
      {/* Print-only styles */}
      <style>{`
        @media print {
          body * { visibility: hidden; }
          #audit-report-content,
          #audit-report-content * { visibility: visible; }
          #audit-report-content { position: absolute; top: 0; left: 0; width: 100%; }
          .print-summary { break-inside: avoid; }
          table { page-break-inside: auto; }
          tr { page-break-inside: avoid; }
        }
      `}</style>

      <div className={styles.jsonLinesPanel}>
        <h2 className={styles.jsonLinesPanelTitle}>Compliance Audit Report</h2>
        <p className={styles.jsonLinesPanelDesc}>
          Generate an org-wide compliance audit report for any date range. Controls are
          mapped to OWASP ASI 2026, EU AI Act, HIPAA §164.312, or SOC 2 Type II. Use the
          Print button to export as a browser PDF.
        </p>

        {/* Controls */}
        <div className={styles.fieldset}>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="ar-from">From date</label>
            <input
              id="ar-from"
              type="date"
              className={styles.input}
              value={fromDate}
              max={toDate}
              onChange={(e) => setFromDate(e.target.value)}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="ar-to">To date</label>
            <input
              id="ar-to"
              type="date"
              className={styles.input}
              value={toDate}
              min={fromDate}
              onChange={(e) => setToDate(e.target.value)}
            />
          </div>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="ar-framework">Framework</label>
            <select
              id="ar-framework"
              className={styles.input}
              value={framework}
              onChange={(e) => setFramework(e.target.value as ComplianceFramework)}
            >
              {FRAMEWORKS.map((f) => (
                <option key={f.value} value={f.value}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
          <div className={styles.field}>
            <label className={styles.label} htmlFor="ar-agent">Agent ID (optional)</label>
            <input
              id="ar-agent"
              className={styles.input}
              placeholder="Filter by agent"
              value={agentId}
              onChange={(e) => setAgentId(e.target.value)}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex gap-3 mt-4">
          <button
            onClick={handleGenerate}
            disabled={isLoading}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg disabled:opacity-50 transition-colors"
          >
            {isLoading ? "Generating…" : "Generate Report"}
          </button>
          {report && (
            <button
              onClick={handlePrint}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-800 text-white text-sm font-medium rounded-lg transition-colors"
            >
              Print / Save as PDF
            </button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
            {error instanceof Error ? error.message : "Failed to generate report"}
          </div>
        )}

        {/* Loading skeleton */}
        {isLoading && (
          <div className="mt-6 space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-10 bg-gray-100 rounded animate-pulse" />
            ))}
          </div>
        )}

        {/* Report */}
        {report && !isLoading && <ReportView report={report} />}
      </div>
    </>
  );
}
