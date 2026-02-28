/**
 * Generic export panel used by each tab on the Export page.
 *
 * Renders a form for connection parameters + optional event filters,
 * a submit button, and a result/error summary after the push completes.
 */
import { type FormEvent } from "react";
import type { PushResult } from "./export.types";
import styles from "./ExportPanel.module.css";

// ─── Sub-types ────────────────────────────────────────────────────────────────

/** A single form field descriptor. */
export interface FieldSpec {
  key: string;
  label: string;
  placeholder?: string;
  required?: boolean;
  type?: "text" | "password" | "number";
  defaultValue?: string;
  hint?: string;
}

interface ExportPanelProps {
  title: string;
  description: string;
  fields: FieldSpec[];
  filterFields?: FieldSpec[];
  isLoading: boolean;
  result: PushResult | null;
  error: string | null;
  values: Record<string, string>;
  onChange: (key: string, value: string) => void;
  onSubmit: () => void;
  onClear: () => void;
  submitLabel?: string;
}

// ─── Component ────────────────────────────────────────────────────────────────

export function ExportPanel({
  title,
  description,
  fields,
  filterFields = [],
  isLoading,
  result,
  error,
  values,
  onChange,
  onSubmit,
  onClear,
  submitLabel = "Push Events",
}: ExportPanelProps) {
  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit();
  }

  return (
    <div className={styles.panel}>
      <h2 className={styles.panelTitle}>{title}</h2>
      <p className={styles.panelDesc}>{description}</p>

      <form onSubmit={handleSubmit}>
        {/* Connection fields */}
        <div className={styles.fieldset}>
          {fields.map((f) => (
            <div key={f.key} className={f.key === fields[0].key ? styles.fieldFull : styles.field}>
              <label className={styles.label} htmlFor={`field-${f.key}`}>
                {f.label}
                {f.required && <span className={styles.required}>*</span>}
                {f.hint && (
                  <span style={{ fontWeight: 400, color: "#94a3b8", marginLeft: 6 }}>
                    ({f.hint})
                  </span>
                )}
              </label>
              <input
                id={`field-${f.key}`}
                className={styles.input}
                type={f.type ?? "text"}
                placeholder={f.placeholder ?? ""}
                required={f.required}
                value={values[f.key] ?? f.defaultValue ?? ""}
                onChange={(e) => onChange(f.key, e.target.value)}
              />
            </div>
          ))}
        </div>

        {/* Optional event filters */}
        {filterFields.length > 0 && (
          <div className={styles.filterSection}>
            <div className={styles.filterTitle}>Event Filters (optional)</div>
            <div className={styles.fieldset}>
              {filterFields.map((f) => (
                <div key={f.key} className={styles.field}>
                  <label className={styles.label} htmlFor={`filter-${f.key}`}>
                    {f.label}
                  </label>
                  <input
                    id={`filter-${f.key}`}
                    className={styles.input}
                    type={f.type ?? "text"}
                    placeholder={f.placeholder ?? ""}
                    value={values[f.key] ?? ""}
                    onChange={(e) => onChange(f.key, e.target.value)}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        <div className={styles.actions}>
          <button type="submit" className={styles.submitBtn} disabled={isLoading}>
            {isLoading && <span className={styles.spinner} aria-hidden="true" />}
            {isLoading ? "Pushing..." : submitLabel}
          </button>
          {result && (
            <button type="button" className={styles.clearBtn} onClick={onClear}>
              Clear results
            </button>
          )}
        </div>
      </form>

      {/* Result box */}
      {error && (
        <div className={`${styles.resultBox} ${styles.resultError}`} role="alert">
          <div className={styles.resultTitle}>Push failed</div>
          <div>{error}</div>
        </div>
      )}

      {result && !error && (
        <div
          className={`${styles.resultBox} ${result.errors.length > 0 ? styles.resultError : styles.resultSuccess}`}
          role="status"
        >
          <div className={styles.resultTitle}>
            {result.errors.length > 0 ? "Push completed with errors" : "Push completed successfully"}
          </div>
          <div className={styles.resultRow}>
            <span className={styles.resultStat}>
              Events sent: <strong>{result.sent.toLocaleString()}</strong>
            </span>
            <span className={styles.resultStat}>
              Batches: <strong>{result.batches}</strong>
            </span>
            {result.errors.length > 0 && (
              <span className={styles.resultStat} style={{ color: "#991b1b" }}>
                Errors: <strong>{result.errors.length}</strong>
              </span>
            )}
          </div>
          {result.errors.length > 0 && (
            <ul className={styles.errorList}>
              {result.errors.map((e, i) => (
                <li key={i}>{e}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
