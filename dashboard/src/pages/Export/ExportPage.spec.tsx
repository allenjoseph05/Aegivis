/**
 * Unit tests for the SIEM Export page.
 *
 * Tests tab switching, form rendering, and push result display.
 * HTTP calls are mocked via vi.mock.
 *
 * Run: npm run test:run
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { ExportPage } from "./ExportPage";

// ─── Mocks ─────────────────────────────────────────────────────────────────────

vi.mock("../../api/client", () => ({
  pushToSplunk: vi.fn(),
  pushToElasticsearch: vi.fn(),
  // Constants used by JsonLinesTab
  BASE_URL: "http://localhost:8000",
  API_KEY: "dev-dashboard-key",
}));

import { pushToSplunk, pushToElasticsearch } from "../../api/client";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <MemoryRouter>
      <QueryClientProvider client={qc}>
        <ExportPage />
      </QueryClientProvider>
    </MemoryRouter>
  );
}

function makePushResult(overrides = {}) {
  return { sent: 500, batches: 1, errors: [], ...overrides };
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

describe("ExportPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders page title", () => {
    renderPage();
    expect(screen.getByText("Export & Compliance")).toBeInTheDocument();
  });

  it("renders three tabs", () => {
    renderPage();
    expect(screen.getByRole("tab", { name: "JSON Lines" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Splunk HEC" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Elasticsearch" })).toBeInTheDocument();
  });

  it("shows JSON Lines panel by default", () => {
    renderPage();
    expect(screen.getByText(/JSON Lines \(NDJSON\) Download/i)).toBeInTheDocument();
  });

  it("switches to Splunk tab", () => {
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Splunk HEC" }));
    expect(screen.getByText("Splunk HEC Push")).toBeInTheDocument();
  });

  it("switches to Elasticsearch tab", () => {
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Elasticsearch" }));
    expect(screen.getByText("Elasticsearch Push")).toBeInTheDocument();
  });

  it("Splunk tab shows HEC URL and token fields", () => {
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Splunk HEC" }));
    expect(screen.getByLabelText(/HEC URL/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/HEC Token/i)).toBeInTheDocument();
  });

  it("Elasticsearch tab shows ES URL field", () => {
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Elasticsearch" }));
    expect(screen.getByLabelText(/Elasticsearch URL/i)).toBeInTheDocument();
  });

  it("Splunk push shows success result", async () => {
    vi.mocked(pushToSplunk).mockResolvedValueOnce(makePushResult({ sent: 250 }));
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Splunk HEC" }));

    fireEvent.change(screen.getByLabelText(/HEC URL/i), {
      target: { value: "http://splunk.test:8088/services/collector/event" },
    });
    fireEvent.change(screen.getByLabelText(/HEC Token/i), {
      target: { value: "my-token" },
    });

    fireEvent.click(screen.getByText("Push Events"));

    await waitFor(() =>
      expect(screen.getByRole("status")).toBeInTheDocument()
    );
    expect(screen.getByText(/250/)).toBeInTheDocument();
  });

  it("Splunk push shows error result with error messages", async () => {
    vi.mocked(pushToSplunk).mockResolvedValueOnce(
      makePushResult({ sent: 0, errors: ["Batch 1: HTTP 401 — Unauthorized"] })
    );
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Splunk HEC" }));
    fireEvent.change(screen.getByLabelText(/HEC URL/i), {
      target: { value: "http://splunk.test:8088/services/collector/event" },
    });
    fireEvent.change(screen.getByLabelText(/HEC Token/i), {
      target: { value: "bad-token" },
    });
    fireEvent.click(screen.getByText("Push Events"));

    await waitFor(() =>
      expect(screen.getByText(/HTTP 401/i)).toBeInTheDocument()
    );
  });

  it("Elasticsearch push shows success result", async () => {
    vi.mocked(pushToElasticsearch).mockResolvedValueOnce(makePushResult({ sent: 1000 }));
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Elasticsearch" }));
    fireEvent.change(screen.getByLabelText(/Elasticsearch URL/i), {
      target: { value: "http://es.test:9200" },
    });
    fireEvent.click(screen.getByText("Push Events"));

    await waitFor(() =>
      expect(screen.getByRole("status")).toBeInTheDocument()
    );
    expect(screen.getByText(/1,000/)).toBeInTheDocument();
  });

  it("clear results button dismisses result", async () => {
    vi.mocked(pushToSplunk).mockResolvedValueOnce(makePushResult());
    renderPage();
    fireEvent.click(screen.getByRole("tab", { name: "Splunk HEC" }));
    fireEvent.change(screen.getByLabelText(/HEC URL/i), {
      target: { value: "http://splunk.test:8088/services/collector/event" },
    });
    fireEvent.change(screen.getByLabelText(/HEC Token/i), {
      target: { value: "tok" },
    });
    fireEvent.click(screen.getByText("Push Events"));
    await waitFor(() => expect(screen.getByRole("status")).toBeInTheDocument());
    fireEvent.click(screen.getByText("Clear results"));
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("tab selected state is correct", () => {
    renderPage();
    const jsonTab = screen.getByRole("tab", { name: "JSON Lines" });
    const splunkTab = screen.getByRole("tab", { name: "Splunk HEC" });
    expect(jsonTab).toHaveAttribute("aria-selected", "true");
    expect(splunkTab).toHaveAttribute("aria-selected", "false");
    fireEvent.click(splunkTab);
    expect(jsonTab).toHaveAttribute("aria-selected", "false");
    expect(splunkTab).toHaveAttribute("aria-selected", "true");
  });
});
