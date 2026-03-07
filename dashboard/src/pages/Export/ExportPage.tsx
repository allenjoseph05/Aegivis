/**
 * SIEM Export Page.
 *
 * Three tabs: JSON Lines streaming download, Splunk HEC push, and
 * Elasticsearch Bulk API push.
 */
import { useState } from "react";
import { ExportPanel } from "./ExportPanel";
import type { ExportTarget, PushResult } from "./export.types";
import { pushToSplunk, pushToElasticsearch, BASE_URL, API_KEY } from "../../api/client";
import { AuditReportPanel } from "./AuditReportPanel";
import styles from "./ExportPage.module.css";

// ─── Field descriptors ─────────────────────────────────────────────────────────

const FILTER_FIELDS = [
  {
    key: "session_id",
    label: "Session ID",
    placeholder: "Filter by session (optional)",
  },
  {
    key: "agent_id",
    label: "Agent ID",
    placeholder: "Filter by agent (optional)",
  },
  {
    key: "limit",
    label: "Max events",
    placeholder: "5000",
    type: "number" as const,
    defaultValue: "5000",
  },
];

const SPLUNK_FIELDS = [
  {
    key: "hec_url",
    label: "HEC URL",
    placeholder: "https://splunk.example.com:8088/services/collector/event",
    required: true,
    hint: "e.g. https://splunk:8088/services/collector/event",
  },
  {
    key: "hec_token",
    label: "HEC Token",
    placeholder: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    required: true,
    type: "password" as const,
    hint: "without 'Splunk ' prefix",
  },
  {
    key: "index",
    label: "Splunk Index",
    placeholder: "aegivis",
    defaultValue: "aegivis",
  },
  {
    key: "source",
    label: "Source",
    placeholder: "aegivis-proxy",
    defaultValue: "aegivis-proxy",
  },
];

const ELASTIC_FIELDS = [
  {
    key: "es_url",
    label: "Elasticsearch URL",
    placeholder: "https://elasticsearch.example.com:9200",
    required: true,
    hint: "base URL without trailing slash",
  },
  {
    key: "api_key",
    label: "API Key",
    placeholder: "Optional — leave empty for open clusters",
    type: "password" as const,
  },
  {
    key: "index",
    label: "Index Name",
    placeholder: "aegivis",
    defaultValue: "aegivis",
  },
];

// ─── JSON Lines tab ───────────────────────────────────────────────────────────

function JsonLinesTab() {
  const [sessionId, setSessionId] = useState("");
  const [agentId, setAgentId] = useState("");
  const [limit, setLimit] = useState("10000");

  const backendUrl = BASE_URL;
  const apiKey = API_KEY;

  const params = new URLSearchParams();
  if (sessionId) params.set("session_id", sessionId);
  if (agentId) params.set("agent_id", agentId);
  if (limit) params.set("limit", limit);

  const downloadUrl = `${backendUrl}/v1/export/jsonlines?${params.toString()}`;

  const curlCmd =
    `curl -H "X-API-Key: ${apiKey}" \\\n` +
    `     "${downloadUrl}" \\\n` +
    `     > aegivis-events.ndjson`;

  return (
    <div className={styles.jsonLinesPanel}>
      <h2 className={styles.jsonLinesPanelTitle}>JSON Lines (NDJSON) Download</h2>
      <p className={styles.jsonLinesPanelDesc}>
        Stream audit events as newline-delimited JSON. Each line is one normalised
        event with all security sub-fields promoted to top level for easy SIEM mapping.
        Compatible with any log aggregator, data lake, or SIEM that accepts NDJSON.
      </p>

      {/* Filters */}
      <div className={styles.fieldset}>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="jl-session">Session ID (optional)</label>
          <input
            id="jl-session"
            className={styles.input}
            placeholder="Filter by session"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
          />
        </div>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="jl-agent">Agent ID (optional)</label>
          <input
            id="jl-agent"
            className={styles.input}
            placeholder="Filter by agent"
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
          />
        </div>
        <div className={styles.field}>
          <label className={styles.label} htmlFor="jl-limit">Max events</label>
          <input
            id="jl-limit"
            className={styles.input}
            type="number"
            min={1}
            max={100000}
            value={limit}
            onChange={(e) => setLimit(e.target.value)}
          />
        </div>
      </div>

      {/* curl command */}
      <div className={styles.codeBlock}>{curlCmd}</div>

      {/* Direct download link */}
      <a
        href={downloadUrl}
        download="aegivis-events.ndjson"
        className={styles.downloadBtn}
        aria-label="Download events as NDJSON file"
      >
        Download NDJSON
      </a>
    </div>
  );
}

// ─── Splunk tab ───────────────────────────────────────────────────────────────

function SplunkTab() {
  const [values, setValues] = useState<Record<string, string>>({
    index: "aegivis",
    source: "aegivis-proxy",
    limit: "5000",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PushResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setError(null);
    setResult(null);
    setIsLoading(true);
    try {
      const res = await pushToSplunk({
        hec_url: values.hec_url ?? "",
        hec_token: values.hec_token ?? "",
        index: values.index || "aegivis",
        source: values.source || "aegivis-proxy",
        session_id: values.session_id || undefined,
        agent_id: values.agent_id || undefined,
        limit: values.limit ? parseInt(values.limit, 10) : 5000,
      });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Push failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <ExportPanel
      title="Splunk HEC Push"
      description="Push audit events to a Splunk HTTP Event Collector. Events are wrapped in
        the standard HEC envelope and delivered in batches of 500."
      fields={SPLUNK_FIELDS}
      filterFields={FILTER_FIELDS}
      isLoading={isLoading}
      result={result}
      error={error}
      values={values}
      onChange={(k, v) => setValues((p) => ({ ...p, [k]: v }))}
      onSubmit={handleSubmit}
      onClear={() => { setResult(null); setError(null); }}
    />
  );
}

// ─── Elasticsearch tab ────────────────────────────────────────────────────────

function ElasticsearchTab() {
  const [values, setValues] = useState<Record<string, string>>({
    index: "aegivis",
    limit: "5000",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PushResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setError(null);
    setResult(null);
    setIsLoading(true);
    try {
      const res = await pushToElasticsearch({
        es_url: values.es_url ?? "",
        api_key: values.api_key || undefined,
        index: values.index || "aegivis",
        session_id: values.session_id || undefined,
        agent_id: values.agent_id || undefined,
        limit: values.limit ? parseInt(values.limit, 10) : 5000,
      });
      setResult(res);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Push failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <ExportPanel
      title="Elasticsearch Push"
      description="Push audit events to Elasticsearch via the Bulk API. Events use event_id
        as the document _id (re-sends are idempotent). Delivered in batches of 500."
      fields={ELASTIC_FIELDS}
      filterFields={FILTER_FIELDS}
      isLoading={isLoading}
      result={result}
      error={error}
      values={values}
      onChange={(k, v) => setValues((p) => ({ ...p, [k]: v }))}
      onSubmit={handleSubmit}
      onClear={() => { setResult(null); setError(null); }}
    />
  );
}

// ─── Main page ─────────────────────────────────────────────────────────────────

export function ExportPage() {
  const [tab, setTab] = useState<ExportTarget>("jsonlines");

  return (
    <div className={styles.page}>
      {/* Header */}
      <div className={styles.header}>
        <h1 className={styles.title}>Export & Compliance</h1>
        <p className={styles.subtitle}>
          Export audit events to your SIEM, data lake, or log aggregator.
          Generate compliance audit reports for OWASP ASI 2026, EU AI Act, HIPAA, or SOC 2.
        </p>
      </div>

      {/* Tabs */}
      <div className={styles.tabs} role="tablist">
        {(
          [
            { id: "jsonlines" as const, label: "JSON Lines" },
            { id: "splunk" as const, label: "Splunk HEC" },
            { id: "elasticsearch" as const, label: "Elasticsearch" },
            { id: "audit" as const, label: "Audit Report" },
          ] as const
        ).map(({ id, label }) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            className={`${styles.tab} ${tab === id ? styles.tabActive : ""}`}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Tab panels */}
      {tab === "jsonlines" && <JsonLinesTab />}
      {tab === "splunk" && <SplunkTab />}
      {tab === "elasticsearch" && <ElasticsearchTab />}
      {tab === "audit" && <AuditReportPanel />}
    </div>
  );
}
