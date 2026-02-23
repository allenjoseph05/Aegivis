import clsx from "clsx";
import { AlertTriangle, AlertCircle, Info, XCircle } from "lucide-react";
import type { RiskFlag, AnomalyFlag } from "../api/client";

const SEVERITY_CONFIG = {
  critical: {
    icon: <XCircle size={16} />,
    bg: "bg-red-50 border-red-300",
    text: "text-red-700",
    badge: "bg-red-100 text-red-700",
    label: "CRITICAL",
  },
  high: {
    icon: <AlertCircle size={16} />,
    bg: "bg-orange-50 border-orange-300",
    text: "text-orange-700",
    badge: "bg-orange-100 text-orange-700",
    label: "HIGH",
  },
  medium: {
    icon: <AlertTriangle size={16} />,
    bg: "bg-yellow-50 border-yellow-300",
    text: "text-yellow-700",
    badge: "bg-yellow-100 text-yellow-700",
    label: "MEDIUM",
  },
  low: {
    icon: <Info size={16} />,
    bg: "bg-blue-50 border-blue-300",
    text: "text-blue-700",
    badge: "bg-blue-100 text-blue-700",
    label: "LOW",
  },
} as const;

type Severity = keyof typeof SEVERITY_CONFIG;

interface FlagItemProps {
  severity: string;
  flag_type: string;
  description: string;
  sequence_number?: number | null;
}

function FlagItem({ severity, flag_type, description, sequence_number }: FlagItemProps) {
  const config = SEVERITY_CONFIG[severity as Severity] ?? SEVERITY_CONFIG.low;

  return (
    <div className={clsx("border rounded-md p-3 flex gap-2", config.bg)}>
      <span className={clsx("shrink-0 mt-0.5", config.text)}>{config.icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={clsx("text-xs font-bold px-1.5 py-0.5 rounded", config.badge)}>
            {config.label}
          </span>
          <span className="text-xs font-mono text-gray-500">{flag_type}</span>
          {sequence_number !== null && sequence_number !== undefined && (
            <span className="text-xs text-gray-400">seq #{sequence_number}</span>
          )}
        </div>
        <p className={clsx("text-sm mt-1", config.text)}>{description}</p>
      </div>
    </div>
  );
}

interface RiskPanelProps {
  riskFlags: RiskFlag[];
  anomalies: AnomalyFlag[];
}

export function RiskPanel({ riskFlags, anomalies }: RiskPanelProps) {
  const allFlags = [
    ...riskFlags.map((f) => ({ ...f, source: "risk" as const })),
    ...anomalies.map((a) => ({
      severity: a.severity,
      flag_type: a.rule_id,
      description: a.description,
      sequence_number: a.sequence_number,
      source: "anomaly" as const,
    })),
  ].sort((a, b) => {
    const order = { critical: 0, high: 1, medium: 2, low: 3 };
    return (order[a.severity as Severity] ?? 4) - (order[b.severity as Severity] ?? 4);
  });

  if (allFlags.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400 text-sm">
        No anomalies or risk flags detected.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-sm text-gray-500 mb-3">
        {allFlags.length} flag{allFlags.length !== 1 ? "s" : ""} detected
      </p>
      {allFlags.map((flag, i) => (
        <FlagItem
          key={i}
          severity={flag.severity}
          flag_type={flag.flag_type}
          description={flag.description}
          sequence_number={flag.sequence_number}
        />
      ))}
    </div>
  );
}
