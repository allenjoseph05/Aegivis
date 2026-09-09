/**
 * Security Benchmark Page -- Phase 3.5
 *
 * Displays scanner effectiveness metrics from the proxy's benchmark engine.
 * Fetches last result on mount; "Run Benchmark" button triggers POST /benchmark/run.
 */
import React, { useEffect, useState } from "react";
import axios from "axios";
import { formatDistanceToNow } from "date-fns";
import clsx from "clsx";
import {
  Shield,
  ShieldAlert,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Clock,
  Play,
  Loader2,
  Info,
  Lock,
  ChevronLeft,
  ChevronRight,
  Settings,
  ChevronDown,
} from "lucide-react";

import {
  runBenchmark,
  fetchLastBenchmark,
  runExternalBenchmark,
  fetchLastExternalBenchmark,
  PROXY_URL,
} from "../../api/client";
import type {
  BenchmarkReport,
  BenchmarkRunOptions,
  CaseResult,
  CategorySummary,
} from "../../api/client";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function fmtPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function fmtMs(v: number): string {
  if (v < 1) return `${v.toFixed(2)}ms`;
  if (v < 1000) return `${v.toFixed(1)}ms`;
  return `${(v / 1000).toFixed(2)}s`;
}

function formatCategory(name: string): string {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Skeleton loader
// ---------------------------------------------------------------------------

function SkeletonCard() {
  return (
    <div className="bg-white rounded-xl border shadow-sm p-5 animate-pulse">
      <div className="h-3 bg-gray-200 rounded w-3/4 mb-3" />
      <div className="h-8 bg-gray-200 rounded w-1/2 mb-2" />
      <div className="h-3 bg-gray-200 rounded w-2/3" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Metric card
// ---------------------------------------------------------------------------

type CardColor = "green" | "yellow" | "red" | "blue" | "gray";

function MetricCard({
  title,
  value,
  subtitle,
  color,
}: {
  title: string;
  value: string;
  subtitle: string;
  color: CardColor;
}) {
  const colorMap: Record<CardColor, string> = {
    green: "text-green-700",
    yellow: "text-amber-600",
    red: "text-red-600",
    blue: "text-blue-700",
    gray: "text-gray-700",
  };
  return (
    <div className="bg-white rounded-xl border shadow-sm p-5">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">
        {title}
      </div>
      <div className={clsx("text-3xl font-bold tabular-nums", colorMap[color])}>
        {value}
      </div>
      <div className="text-xs text-gray-400 mt-1">{subtitle}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Score bar
// ---------------------------------------------------------------------------

function ScoreBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.round(score * 100));
  const color =
    score >= 0.8
      ? "bg-red-500"
      : score >= 0.5
      ? "bg-amber-500"
      : "bg-green-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-gray-100 rounded-full overflow-hidden">
        <div className={clsx("h-full rounded-full", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="tabular-nums text-xs text-gray-500 w-8 text-right">
        {score.toFixed(2)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Rate bar
// ---------------------------------------------------------------------------

function RateBar({ rate }: { rate: number }) {
  const pct = Math.min(100, Math.round(rate * 100));
  const color =
    rate >= 0.9
      ? "bg-green-500"
      : rate >= 0.7
      ? "bg-amber-400"
      : "bg-red-400";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden max-w-[120px]">
        <div className={clsx("h-full rounded-full", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="tabular-nums text-xs font-medium text-gray-600">
        {fmtPct(rate)}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Layer badge
// ---------------------------------------------------------------------------

function LayerBadge({
  name,
  active,
  installHint,
}: {
  name: string;
  active: boolean;
  installHint?: string;
}) {
  return (
    <span
      title={active ? undefined : installHint}
      className={clsx(
        "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border",
        active
          ? "bg-green-50 text-green-700 border-green-200"
          : "bg-gray-50 text-gray-400 border-gray-200 opacity-60"
      )}
    >
      {active ? (
        <CheckCircle2 size={11} className="text-green-500" />
      ) : (
        <XCircle size={11} />
      )}
      {name}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Case badge (TP / FN / FP / TN)
// ---------------------------------------------------------------------------

function CaseBadge({ c }: { c: CaseResult }) {
  if (c.expected === "attack" && c.detected) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-green-100 text-green-700">
        <CheckCircle2 size={11} /> Detected
      </span>
    );
  }
  if (c.expected === "attack" && !c.detected) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-red-100 text-red-700">
        <XCircle size={11} /> Missed
      </span>
    );
  }
  if (c.expected === "clean" && c.detected) {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 text-amber-700">
        <AlertTriangle size={11} /> FP
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-gray-100 text-gray-500">
      <CheckCircle2 size={11} /> Clean
    </span>
  );
}

function caseRowGlow(c: CaseResult): string {
  if (c.expected === "attack" && c.detected) return "bg-green-50/60";
  if (c.expected === "attack" && !c.detected) return "bg-red-50/60";
  if (c.expected === "clean" && c.detected) return "bg-amber-50/60";
  return "";
}

// ---------------------------------------------------------------------------
// Top layer that caught it
// ---------------------------------------------------------------------------

function topLayer(c: CaseResult): string {
  const scores = c.layer_scores ?? {};
  if (!Object.keys(scores).length) {
    return c.detected ? "unknown" : "-";
  }
  const top = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  return top ? top[0] : "-";
}

// ---------------------------------------------------------------------------
// Confusion Matrix
// ---------------------------------------------------------------------------

function ConfusionMatrix({ report }: { report: BenchmarkReport }) {
  return (
    <div className="bg-white rounded-xl border shadow-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <ShieldAlert size={16} className="text-blue-600" />
        <h3 className="font-semibold text-gray-900">Confusion Matrix</h3>
      </div>
      <div className="overflow-x-auto">
        <table className="text-sm border-collapse w-full max-w-md">
          <thead>
            <tr>
              <th className="px-3 py-2 text-left text-xs text-gray-400 font-normal w-32" />
              <th className="px-3 py-2 text-center text-xs font-semibold text-gray-600 bg-gray-50 border rounded-tl">
                Predicted Attack
              </th>
              <th className="px-3 py-2 text-center text-xs font-semibold text-gray-600 bg-gray-50 border rounded-tr">
                Predicted Clean
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="px-3 py-2 text-xs font-semibold text-gray-600 bg-gray-50 border">
                Is Attack
              </td>
              <td className="px-3 py-2 text-center border bg-green-50">
                <div className="text-2xl font-bold text-green-700 tabular-nums">
                  {report.true_positives}
                </div>
                <div className="text-xs text-green-500 font-medium">TP</div>
              </td>
              <td className="px-3 py-2 text-center border bg-red-50">
                <div className="text-2xl font-bold text-red-600 tabular-nums">
                  {report.false_negatives}
                </div>
                <div className="text-xs text-red-400 font-medium">FN</div>
              </td>
            </tr>
            <tr>
              <td className="px-3 py-2 text-xs font-semibold text-gray-600 bg-gray-50 border">
                Is Clean
              </td>
              <td className="px-3 py-2 text-center border bg-amber-50">
                <div className="text-2xl font-bold text-amber-600 tabular-nums">
                  {report.false_positives}
                </div>
                <div className="text-xs text-amber-400 font-medium">FP</div>
              </td>
              <td className="px-3 py-2 text-center border bg-green-50">
                <div className="text-2xl font-bold text-green-700 tabular-nums">
                  {report.true_negatives}
                </div>
                <div className="text-xs text-green-500 font-medium">TN</div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-xs text-gray-400">
        <span className="font-medium text-gray-500">Accuracy: </span>
        {fmtPct(report.accuracy)}
        {"  |  "}
        <span className="font-medium text-gray-500">Precision: </span>
        {fmtPct(report.precision)}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Latency card
// ---------------------------------------------------------------------------

function LatencyCard({ report }: { report: BenchmarkReport }) {
  const items = [
    { label: "p50", value: fmtMs(report.latency_p50_ms), desc: "Median" },
    { label: "p95", value: fmtMs(report.latency_p95_ms), desc: "95th pct" },
    { label: "p99", value: fmtMs(report.latency_p99_ms), desc: "99th pct" },
    { label: "mean", value: fmtMs(report.latency_mean_ms), desc: "Average" },
  ];
  return (
    <div className="bg-white rounded-xl border shadow-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <Clock size={16} className="text-blue-600" />
        <h3 className="font-semibold text-gray-900">Scan Latency Distribution</h3>
        <span className="text-xs text-gray-400 ml-1">per-request overhead</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {items.map(({ label, value, desc }) => (
          <div key={label} className="text-center">
            <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
              {label}
            </div>
            <div className="text-2xl font-bold text-gray-800 tabular-nums mt-1">
              {value}
            </div>
            <div className="text-xs text-gray-400">{desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Category breakdown table
// ---------------------------------------------------------------------------

function CategoryTable({ categories }: { categories: CategorySummary[] }) {
  return (
    <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
      <div className="flex items-center gap-2 px-5 py-4 border-b bg-gray-50">
        <Shield size={16} className="text-blue-600" />
        <h3 className="font-semibold text-gray-900">Category Breakdown</h3>
      </div>
      <table className="w-full text-sm">
        <thead className="bg-gray-50 border-b">
          <tr>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Category
            </th>
            <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Total
            </th>
            <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Detected
            </th>
            <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide min-w-[140px]">
              Detection Rate
            </th>
            <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Blocked
            </th>
            <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Block Rate
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {categories.length === 0 ? (
            <tr>
              <td colSpan={6} className="text-center py-8 text-gray-400 text-sm">
                No category data available.
              </td>
            </tr>
          ) : (
            categories.map((cat) => (
              <tr key={cat.name} className="hover:bg-gray-50 transition-colors">
                <td className="px-4 py-2.5 font-medium text-gray-800">
                  {formatCategory(cat.name)}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-gray-600">
                  {cat.total}
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-gray-600">
                  {cat.detected}
                </td>
                <td className="px-4 py-2.5">
                  <RateBar rate={cat.detection_rate} />
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-gray-600">
                  <span className="inline-flex items-center gap-1">
                    <Lock size={10} className="text-gray-400" />
                    {cat.blocked}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right tabular-nums text-gray-500 text-xs">
                  {fmtPct(cat.block_rate)}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sample results table (paginated)
// ---------------------------------------------------------------------------

const PAGE_SIZE = 20;

function SamplesTable({ samples }: { samples: CaseResult[] }) {
  const [page, setPage] = useState(0);
  const [filterExpected, setFilterExpected] = useState("");
  const [filterCategory, setFilterCategory] = useState("");

  const categories = Array.from(new Set(samples.map((s) => s.category))).sort();

  const filtered = samples.filter((s) => {
    if (filterExpected && s.expected !== filterExpected) return false;
    if (filterCategory && s.category !== filterCategory) return false;
    return true;
  });

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  function handleFilterChange(setter: (v: string) => void) {
    return (e: React.ChangeEvent<HTMLSelectElement>) => {
      setter(e.target.value);
      setPage(0);
    };
  }

  return (
    <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b bg-gray-50">
        <div className="flex items-center gap-2">
          <ShieldAlert size={16} className="text-blue-600" />
          <h3 className="font-semibold text-gray-900">Sample Results</h3>
          <span className="text-xs text-gray-400 ml-1">
            {filtered.length} case{filtered.length !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="flex gap-2">
          <select
            value={filterExpected}
            onChange={handleFilterChange(setFilterExpected)}
            className="border rounded-lg px-3 py-1.5 text-xs text-gray-600 bg-white shadow-sm outline-none"
          >
            <option value="">All types</option>
            <option value="attack">Attacks</option>
            <option value="clean">Clean</option>
          </select>
          <select
            value={filterCategory}
            onChange={handleFilterChange(setFilterCategory)}
            className="border rounded-lg px-3 py-1.5 text-xs text-gray-600 bg-white shadow-sm outline-none"
          >
            <option value="">All categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {formatCategory(c)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b">
            <tr>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide max-w-xs">
                Input
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Category
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Expected
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide min-w-[120px]">
                Score
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Result
              </th>
              <th className="px-4 py-2.5 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Caught by
              </th>
              <th className="px-4 py-2.5 text-right text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Latency
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {paginated.length === 0 ? (
              <tr>
                <td colSpan={7} className="text-center py-10 text-gray-400 text-sm">
                  No cases match the current filter.
                </td>
              </tr>
            ) : (
              paginated.map((c, i) => (
                <tr
                  key={i}
                  className={clsx("hover:bg-gray-50/80 transition-colors", caseRowGlow(c))}
                >
                  <td className="px-4 py-2.5 max-w-xs">
                    <span
                      className="block truncate font-mono text-xs text-gray-700"
                      title={c.text}
                    >
                      {c.text.length > 80 ? c.text.slice(0, 80) + "..." : c.text}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-xs text-gray-600 whitespace-nowrap">
                    {formatCategory(c.category)}
                  </td>
                  <td className="px-4 py-2.5">
                    <span
                      className={clsx(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                        c.expected === "attack"
                          ? "bg-red-50 text-red-600"
                          : "bg-gray-100 text-gray-500"
                      )}
                    >
                      {c.expected === "attack" ? (
                        <ShieldAlert size={10} />
                      ) : (
                        <CheckCircle2 size={10} />
                      )}
                      {c.expected}
                    </span>
                  </td>
                  <td className="px-4 py-2.5">
                    <ScoreBar score={c.score} />
                  </td>
                  <td className="px-4 py-2.5">
                    <CaseBadge c={c} />
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-gray-500">
                    {topLayer(c)}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs text-gray-400">
                    {fmtMs(c.latency_ms)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-5 py-3 border-t bg-gray-50">
          <span className="text-xs text-gray-500">
            Page {page + 1} of {totalPages} &mdash; {filtered.length} cases
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="inline-flex items-center gap-1 px-3 py-1 text-xs border rounded hover:bg-gray-100 disabled:opacity-40"
            >
              <ChevronLeft size={12} /> Previous
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="inline-flex items-center gap-1 px-3 py-1 text-xs border rounded hover:bg-gray-100 disabled:opacity-40"
            >
              Next <ChevronRight size={12} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Active layers card
// ---------------------------------------------------------------------------

const LAYER_DEFS: Array<{
  id: string;
  label: string;
  description: string;
  alwaysActive: boolean;
  installHint?: string;
  installCmd?: string;
}> = [
  {
    id: "normalizer",
    label: "Layer 0: Normalizer",
    description: "Decodes Base64/Hex/ROT13/URL obfuscation. Zero dependencies.",
    alwaysActive: true,
  },
  {
    id: "structural",
    label: "Layer 1: Structural",
    description: "Detects LLM token delimiters, RTL overrides, injection phrases. Zero dependencies.",
    alwaysActive: true,
  },
  {
    id: "semantic",
    label: "Layer 2: Semantic",
    description: "Cosine similarity to 180-sentence attack vector library. Catches paraphrased variants.",
    alwaysActive: false,
    installHint: "Requires sentence-transformers",
    installCmd: "pip install 'aegivis-proxy[security]'",
  },
  {
    id: "coherence",
    label: "Layer 3: Coherence",
    description: "Measures semantic drift from original task intent. Catches goal-hijacking in multi-turn sessions.",
    alwaysActive: false,
    installHint: "Requires sentence-transformers (same as Layer 2)",
    installCmd: "pip install 'aegivis-proxy[security]'",
  },
  {
    id: "classifier",
    label: "Layer 4: Classifier",
    description: "DeBERTa fine-tuned on 30k+ real attacks. 88-99% PINT benchmark accuracy.",
    alwaysActive: false,
    installHint: "Requires transformers + torch (~280 MB)",
    installCmd: "pip install 'aegivis-proxy[classifier]'",
  },
];

function ActiveLayersCard({ activeLayers }: { activeLayers: string[] }) {
  const activeSet = new Set(activeLayers.map((l) => l.toLowerCase()));
  const inactiveLayers = LAYER_DEFS.filter(
    ({ id, alwaysActive }) => !alwaysActive && !activeSet.has(id)
  );
  // Deduplicate install commands
  const missingCmds = [...new Set(inactiveLayers.map((l) => l.installCmd).filter(Boolean))];

  return (
    <div className="bg-white rounded-xl border shadow-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <Shield size={16} className="text-green-600" />
        <h3 className="font-semibold text-gray-900">Active Detection Layers</h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {LAYER_DEFS.map(({ id, label, description, alwaysActive, installHint }) => {
          const isActive = alwaysActive || activeSet.has(id);
          return (
            <LayerBadge
              key={id}
              name={label}
              active={isActive}
              installHint={isActive ? undefined : `${installHint} — ${description}`}
            />
          );
        })}
      </div>
      {missingCmds.length > 0 && (
        <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p className="text-xs font-medium text-amber-800 mb-2">
            {inactiveLayers.length} layer{inactiveLayers.length > 1 ? "s" : ""} inactive — install optional dependencies to unlock higher TPR:
          </p>
          {missingCmds.map((cmd) => (
            <code key={cmd} className="block text-xs bg-amber-100 text-amber-900 px-2 py-1 rounded font-mono mt-1">
              {cmd}
            </code>
          ))}
        </div>
      )}
      <p className="mt-3 text-xs text-gray-400">
        Layers 0–1 run on any machine (no ML). Layers 2–4 unlock semantic + ML detection.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// FPR color helper
// ---------------------------------------------------------------------------

function fprColor(fpr: number): CardColor {
  if (fpr < 0.01) return "green";
  if (fpr <= 0.05) return "yellow";
  return "red";
}

// ---------------------------------------------------------------------------
// Reusable report display
// ---------------------------------------------------------------------------

function ReportDisplay({
  report,
  isExternal,
}: {
  report: BenchmarkReport;
  isExternal: boolean;
}) {
  return (
    <>
      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="True Positive Rate (TPR)"
          value={fmtPct(report.tpr)}
          subtitle="Attacks detected"
          color="green"
        />
        <MetricCard
          title="False Positive Rate (FPR)"
          value={fmtPct(report.fpr)}
          subtitle="Clean msgs flagged"
          color={fprColor(report.fpr)}
        />
        <MetricCard
          title="F1 Score"
          value={report.f1.toFixed(3)}
          subtitle="Precision-recall balance"
          color="blue"
        />
        <MetricCard
          title="Scan Latency (p50)"
          value={fmtMs(report.latency_p50_ms)}
          subtitle="Per-request overhead"
          color="gray"
        />
      </div>

      {/* Benchmark rigor cards (only shown for internal benchmark with rigor data) */}
      {!isExternal && (
        report.contamination_pct !== undefined ||
        report.paraphrase_tpr !== undefined ||
        report.long_context_tpr !== undefined
      ) && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {report.contamination_pct !== undefined && (
            <MetricCard
              title="Self-Test Bias"
              value={fmtPct(report.contamination_pct)}
              subtitle={`${Math.round((report.contamination_pct ?? 0) * (report.total_attacks ?? 0))} of ${report.total_attacks} attack cases use exact scanner phrases`}
              color={(report.contamination_pct ?? 0) > 0.8 ? "yellow" : "green"}
            />
          )}
          {report.paraphrase_tpr !== undefined && (
            <MetricCard
              title="Paraphrase TPR"
              value={fmtPct(report.paraphrase_tpr)}
              subtitle={`${report.paraphrase_detected ?? 0} / ${report.paraphrase_count ?? 0} paraphrase attacks caught`}
              color={(report.paraphrase_tpr ?? 0) >= 0.5 ? "green" : (report.paraphrase_tpr ?? 0) >= 0.25 ? "yellow" : "red"}
            />
          )}
          {report.long_context_tpr !== undefined && (
            <MetricCard
              title="Long-Context TPR"
              value={fmtPct(report.long_context_tpr)}
              subtitle={`${report.long_context_detected ?? 0} / ${report.long_context_count ?? 0} padding-evasion attacks caught — needs WalledGuard 8k`}
              color={(report.long_context_tpr ?? 0) >= 0.8 ? "green" : (report.long_context_tpr ?? 0) >= 0.4 ? "yellow" : "red"}
            />
          )}
        </div>
      )}

      {/* Active layers (internal only — external uses same pipeline) */}
      {!isExternal && <ActiveLayersCard activeLayers={report.active_layers ?? []} />}

      {/* Category table */}
      <CategoryTable categories={report.categories ?? []} />

      {/* Confusion matrix + latency */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <ConfusionMatrix report={report} />
        <LatencyCard report={report} />
      </div>

      {/* Sample results */}
      {report.samples && report.samples.length > 0 && (
        <SamplesTable samples={report.samples} />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Layer settings panel
// ---------------------------------------------------------------------------

interface LayerSettings {
  useClassifier: boolean;
}

const DEFAULT_SETTINGS: LayerSettings = { useClassifier: false };

const LAYER_SETTINGS_INFO = [
  {
    id: "normalizer" as const,
    label: "Layer 0 — Normalizer",
    description: "Decodes Base64, Hex, ROT13, URL-encoded obfuscation before scanning.",
    impact: null,
    alwaysOn: true,
  },
  {
    id: "structural" as const,
    label: "Layer 1 — Structural",
    description: "Detects LLM token delimiters (e.g. <|im_start|>, [INST]) and RTL Unicode overrides.",
    impact: null,
    alwaysOn: true,
  },
  {
    id: "classifier" as const,
    label: "Layer 4 — DeBERTa Classifier",
    description:
      "Fine-tuned on 30k+ real injection attacks. Catches paraphrased and obfuscated variants the structural layer misses. Adds ~50ms per case.",
    impactLabel: "+15–30% TPR on paraphrase attacks",
    alwaysOn: false,
    settingKey: "useClassifier" as keyof LayerSettings,
  },
];

function SettingsPanel({
  settings,
  onChange,
}: {
  settings: LayerSettings;
  onChange: (s: LayerSettings) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-5 py-3.5 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Settings size={15} className="text-gray-500" />
          <span className="text-sm font-semibold text-gray-800">Layer Settings</span>
          {settings.useClassifier && (
            <span className="text-[10px] bg-blue-100 text-blue-700 font-semibold px-2 py-0.5 rounded-full">
              DeBERTa ON
            </span>
          )}
        </div>
        <ChevronDown
          size={15}
          className={clsx(
            "text-gray-400 transition-transform",
            open ? "rotate-180" : ""
          )}
        />
      </button>

      {open && (
        <div className="border-t px-5 py-4 space-y-4">
          <p className="text-xs text-gray-500">
            Choose which detection layers are active during the benchmark run. Disabling slower
            layers lets you isolate and compare their individual contribution to TPR and FPR.
          </p>
          {LAYER_SETTINGS_INFO.map((layer) => (
            <div key={layer.id} className="flex items-start gap-3">
              {layer.alwaysOn ? (
                <span className="mt-0.5 w-9 text-center">
                  <span className="inline-block w-5 h-5 rounded bg-green-100 border border-green-300 flex items-center justify-center">
                    <CheckCircle2 size={11} className="text-green-600" />
                  </span>
                </span>
              ) : (
                <input
                  type="checkbox"
                  id={`layer-${layer.id}`}
                  className="mt-1 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
                  checked={settings[layer.settingKey!] ?? false}
                  onChange={(e) =>
                    onChange({ ...settings, [layer.settingKey!]: e.target.checked })
                  }
                />
              )}
              <label
                htmlFor={layer.alwaysOn ? undefined : `layer-${layer.id}`}
                className={clsx("flex-1 cursor-pointer", layer.alwaysOn && "cursor-default")}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-800">{layer.label}</span>
                  {layer.alwaysOn && (
                    <span className="text-[10px] bg-gray-100 text-gray-500 font-medium px-1.5 py-0.5 rounded">
                      always on
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-0.5">{layer.description}</p>
                {"impactLabel" in layer && layer.impactLabel && (
                  <p className="text-xs text-blue-600 font-medium mt-0.5">
                    Expected impact: {layer.impactLabel}
                  </p>
                )}
              </label>
            </div>
          ))}
          <div className="pt-2 border-t">
            <p className="text-xs text-gray-400">
              Layers 2 (Semantic) and 3 (Coherence) require{" "}
              <code className="bg-gray-100 px-1 rounded">pip install aegivis-proxy[security]</code>{" "}
              and are not yet exposed as benchmark toggles.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type BenchmarkMode = "internal" | "external";

export function BenchmarkPage() {
  const [mode, setMode] = useState<BenchmarkMode>("internal");
  const [layerSettings, setLayerSettings] = useState<LayerSettings>(DEFAULT_SETTINGS);

  // Internal benchmark state
  const [internalReport, setInternalReport] = useState<BenchmarkReport | null>(null);
  const [internalLoading, setInternalLoading] = useState(true);
  const [internalRunning, setInternalRunning] = useState(false);
  const [internalError, setInternalError] = useState<string | null>(null);

  // External benchmark state
  const [externalReport, setExternalReport] = useState<BenchmarkReport | null>(null);
  const [externalLoading, setExternalLoading] = useState(true);
  const [externalRunning, setExternalRunning] = useState(false);
  const [externalError, setExternalError] = useState<string | null>(null);

  // Load last internal benchmark on mount.
  // Endpoint always returns 200 (null when no report exists) — no 404 in console.
  useEffect(() => {
    setInternalLoading(true);
    setInternalError(null);
    fetchLastBenchmark()
      .then((data) => setInternalReport(data))
      .catch((err) => {
        setInternalError(
          axios.isAxiosError(err)
            ? `Could not reach proxy at ${PROXY_URL}. Is it running?`
            : String(err)
        );
      })
      .finally(() => setInternalLoading(false));
  }, []);

  // Load last external benchmark on mount.
  // Endpoint always returns 200 (null when no report exists) — no 404 in console.
  useEffect(() => {
    setExternalLoading(true);
    setExternalError(null);
    fetchLastExternalBenchmark()
      .then((data) => setExternalReport(data))
      .catch((err) => {
        setExternalError(
          axios.isAxiosError(err)
            ? `Could not reach proxy at ${PROXY_URL}. Is it running?`
            : String(err)
        );
      })
      .finally(() => setExternalLoading(false));
  }, []);

  const benchmarkOpts: BenchmarkRunOptions = {
    useClassifier: layerSettings.useClassifier,
  };

  async function handleRunInternal() {
    setInternalRunning(true);
    setInternalError(null);
    try {
      const data = await runBenchmark(benchmarkOpts);
      setInternalReport(data);
    } catch (err) {
      setInternalError(
        axios.isAxiosError(err)
          ? `Benchmark failed: ${(err.response?.data as { detail?: string })?.detail ?? err.message}`
          : String(err)
      );
    } finally {
      setInternalRunning(false);
    }
  }

  async function handleRunExternal() {
    setExternalRunning(true);
    setExternalError(null);
    try {
      const data = await runExternalBenchmark(benchmarkOpts);
      setExternalReport(data);
    } catch (err) {
      setExternalError(
        axios.isAxiosError(err)
          ? `External benchmark failed: ${(err.response?.data as { detail?: string })?.detail ?? err.message}`
          : String(err)
      );
    } finally {
      setExternalRunning(false);
    }
  }

  const isInternal = mode === "internal";
  const report = isInternal ? internalReport : externalReport;
  const loading = isInternal ? internalLoading : externalLoading;
  const running = isInternal ? internalRunning : externalRunning;
  const error = isInternal ? internalError : externalError;
  const handleRun = isInternal ? handleRunInternal : handleRunExternal;

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Security Benchmark</h1>
          <p className="text-sm text-gray-500 mt-1">
            Scanner effectiveness evaluation against labelled attack datasets
          </p>
          {report && (
            <p className="text-xs text-gray-400 mt-0.5">
              Last run{" "}
              {formatDistanceToNow(new Date(report.run_at), { addSuffix: true })}
              {" — "}
              {report.duration_s.toFixed(1)}s total &bull;{" "}
              {report.total_attacks} core attacks + {report.total_clean} clean
              {(report.long_context_count ?? 0) > 0 && (
                <> + {report.long_context_count} long-context (held-out)</>
              )}
            </p>
          )}
        </div>
        <button
          onClick={handleRun}
          disabled={running}
          className={clsx(
            "inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold text-white transition-colors shadow-sm",
            running
              ? "bg-blue-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700 active:bg-blue-800"
          )}
        >
          {running ? (
            <><Loader2 size={15} className="animate-spin" />Running...</>
          ) : (
            <><Play size={15} />Run Benchmark</>
          )}
        </button>
      </div>

      {/* Mode tabs */}
      <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1 w-fit">
        <button
          onClick={() => setMode("internal")}
          className={clsx(
            "px-4 py-1.5 rounded-md text-sm font-medium transition-colors",
            mode === "internal"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-gray-500 hover:text-gray-700"
          )}
        >
          Internal Dataset
          <span className="ml-1.5 text-xs text-gray-400">145 attacks</span>
        </button>
        <button
          onClick={() => setMode("external")}
          className={clsx(
            "px-4 py-1.5 rounded-md text-sm font-medium transition-colors",
            mode === "external"
              ? "bg-white text-gray-900 shadow-sm"
              : "text-gray-500 hover:text-gray-700"
          )}
        >
          Real-World Datasets
          <span className="ml-1.5 text-xs text-gray-400">HuggingFace</span>
        </button>
      </div>

      {/* Layer settings */}
      <SettingsPanel settings={layerSettings} onChange={setLayerSettings} />

      {/* Internal / External content */}
      <>
        {/* External dataset info */}
        {mode === "external" && !externalReport && !externalLoading && !externalError && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3 text-sm text-blue-700 space-y-1">
              <p className="font-semibold">Real-World Dataset Benchmark</p>
              <p className="text-xs text-blue-600">
                Tests the scanner against 10 HuggingFace datasets covering direct injection,
                jailbreaks, in-the-wild attacks, HackAPrompt competition prompts, and adversarial
                machine-generated attacks. Benign FPR stress-tested against Alpaca + ChatGPT prompts.
              </p>
              <p className="text-xs text-blue-500">
                Requires: <code className="bg-blue-100 px-1 rounded">pip install datasets</code> on the proxy.
                First run downloads datasets (~100-500 MB, cached after).
              </p>
            </div>
          )}

          {/* Error banner */}
          {error && (
            <div className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
              <XCircle size={16} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* No-data info banner */}
          {!loading && !error && !report && (
            <div className="flex items-start gap-2 bg-gray-50 border border-gray-200 text-gray-600 rounded-lg px-4 py-3 text-sm">
              <Info size={16} className="mt-0.5 shrink-0 text-gray-400" />
              {isInternal ? (
                <span>
                  No benchmark run yet. Click{" "}
                  <span className="font-semibold text-gray-800">Run Benchmark</span> to
                  evaluate scanner effectiveness against 145 known attacks and 60 clean
                  messages. Takes 1-30s depending on active layers.
                </span>
              ) : (
                <span>
                  No external benchmark run yet. Click{" "}
                  <span className="font-semibold text-gray-800">Run Benchmark</span> to
                  test against real-world HuggingFace datasets. Requires{" "}
                  <code className="bg-gray-100 px-1 rounded text-xs">pip install datasets</code>.
                  First run may take several minutes to download datasets.
                </span>
              )}
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
              <SkeletonCard />
            </div>
          )}

          {/* Results */}
          {!loading && report && (
            <ReportDisplay report={report} isExternal={!isInternal} />
          )}
        </>
    </div>
  );
}
