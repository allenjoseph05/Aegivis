import { useState } from "react";
import { FileText, Download, Loader2 } from "lucide-react";
import { generateComplianceReport } from "../api/client";
import type { Regulation, ComplianceReportResponse } from "../api/client";
import clsx from "clsx";

const REGULATIONS: { value: Regulation; label: string; description: string }[] = [
  {
    value: "eu_ai_act",
    label: "EU AI Act — Article 12",
    description: "Technical documentation and logging requirements",
  },
  {
    value: "gdpr",
    label: "GDPR — Art.5/25/30/32",
    description: "Integrity, privacy by design, records of processing, security of processing",
  },
  {
    value: "hipaa",
    label: "HIPAA — §164.312",
    description: "Technical safeguards and audit controls for ePHI",
  },
  {
    value: "soc2",
    label: "SOC 2 Type II",
    description: "Security and availability trust service criteria",
  },
  {
    value: "owasp_asi_2026",
    label: "OWASP ASI 2026",
    description: "AI system injection and tool abuse prevention controls",
  },
];

interface ReportFormProps {
  sessionId: string;
  orgId?: string;
}

export function ReportForm({ sessionId, orgId: _orgId = "default-org" }: ReportFormProps) {
  const [selected, setSelected] = useState<Regulation>("eu_ai_act");
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<ComplianceReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerate() {
    setLoading(true);
    setError(null);
    setReport(null);
    try {
      const result = await generateComplianceReport(sessionId, selected);
      setReport(result);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to generate report");
    } finally {
      setLoading(false);
    }
  }

  function handleDownloadJson() {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aegivis-${selected}-${sessionId}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4">
      {/* Regulation selector */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {REGULATIONS.map((reg) => (
          <label
            key={reg.value}
            className={clsx(
              "flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors",
              selected === reg.value
                ? "border-blue-500 bg-blue-50"
                : "border-gray-200 hover:border-gray-300"
            )}
          >
            <input
              type="radio"
              name="regulation"
              value={reg.value}
              checked={selected === reg.value}
              onChange={() => setSelected(reg.value)}
              className="mt-1"
            />
            <div>
              <div className="font-medium text-sm text-gray-800">{reg.label}</div>
              <div className="text-xs text-gray-500 mt-0.5">{reg.description}</div>
            </div>
          </label>
        ))}
      </div>

      {/* Generate button */}
      <button
        onClick={handleGenerate}
        disabled={loading}
        className={clsx(
          "flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors",
          loading
            ? "bg-gray-100 text-gray-400 cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-700"
        )}
      >
        {loading ? (
          <>
            <Loader2 size={16} className="animate-spin" />
            Generating…
          </>
        ) : (
          <>
            <FileText size={16} />
            Generate Report
          </>
        )}
      </button>

      {/* Error */}
      {error && (
        <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded p-3">
          {error}
        </div>
      )}

      {/* Report result */}
      {report && (
        <div className="border rounded-lg overflow-hidden">
          {/* Legal disclaimer */}
          <div className="px-4 py-3 bg-amber-50 border-b border-amber-200">
            <p className="text-xs text-amber-800">
              <strong>Notice:</strong> This is a technical evidence report documenting observed
              security control activity. It is not a compliance certification. Legal compliance
              determination requires qualified assessment against applicable regulatory requirements.
            </p>
          </div>

          {/* Header */}
          <div
            className={clsx(
              "flex items-center justify-between px-4 py-3",
              report.compliant ? "bg-green-50 border-b border-green-200" : "bg-red-50 border-b border-red-200"
            )}
          >
            <div>
              <div className="font-semibold text-gray-800">{report.regulation}</div>
              <div className="text-xs text-gray-500">Generated {new Date(report.generated_at).toLocaleString()}</div>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={clsx(
                  "px-2 py-1 rounded-full text-xs font-bold",
                  report.compliant
                    ? "bg-green-100 text-green-700"
                    : "bg-red-100 text-red-700"
                )}
              >
                {report.compliant ? "COMPLIANT" : "NON-COMPLIANT"}
              </span>
              <button
                onClick={handleDownloadJson}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs text-gray-600 hover:bg-gray-100"
              >
                <Download size={14} />
                JSON
              </button>
            </div>
          </div>

          {/* Gaps */}
          {report.gaps.length > 0 && (
            <div className="px-4 py-3 bg-red-50 border-b border-red-100">
              <div className="text-xs font-semibold text-red-700 mb-1">Compliance Gaps</div>
              <ul className="space-y-1">
                {report.gaps.map((gap, i) => (
                  <li key={i} className="text-sm text-red-600 flex gap-2">
                    <span>•</span> {gap}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Recommendations */}
          {report.recommendations.length > 0 && (
            <div className="px-4 py-3 border-b border-gray-100">
              <div className="text-xs font-semibold text-gray-600 mb-1">Recommendations</div>
              <ul className="space-y-1">
                {report.recommendations.map((rec, i) => (
                  <li key={i} className="text-sm text-gray-600 flex gap-2">
                    <span>•</span> {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Evidence (collapsed) */}
          <details className="px-4 py-3">
            <summary className="text-xs font-semibold text-gray-500 cursor-pointer hover:text-gray-700">
              Evidence ({Object.keys(report.evidence).length} sections)
            </summary>
            <pre className="mt-2 p-3 bg-gray-50 rounded text-xs text-gray-700 overflow-auto max-h-64">
              {JSON.stringify(report.evidence, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}
