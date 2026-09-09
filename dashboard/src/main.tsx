import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, NavLink, Navigate, useLocation } from "react-router-dom";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { AegivisLogo } from "./components/AegivisLogo";
import {
  LayoutDashboard, Bot, Activity, ShieldAlert, Bell,
  Layers, TrendingUp, Cog, BarChart2, Download, Zap,
} from "lucide-react";
import "./index.css";

// Pages
import { OverviewPage } from "./pages/Overview";
import { Sessions } from "./pages/Sessions";
import { SessionDetail } from "./pages/SessionDetail";
import { Forensics } from "./pages/Forensics";
import { Compliance } from "./pages/Compliance";
import { SecurityStack } from "./pages/SecurityStack";
import { BenchmarkPage } from "./pages/Benchmark";
import { AgentsPage } from "./pages/Agents";
import { ViolationsPage } from "./pages/Violations";
import { SettingsPage } from "./pages/Settings";

// Enterprise pages — loaded lazily only when VITE_EDITION=enterprise.
// Named exports are wrapped into { default } so React.lazy is satisfied.
const AgentDetail = React.lazy(() =>
  import("./pages/Agents/AgentDetail").then(m => ({ default: m.AgentDetail })));
const ApprovalsPage = React.lazy(() =>
  import("./pages/Approvals").then(m => ({ default: m.ApprovalsPage })));
const SecurityPosturePage = React.lazy(() =>
  import("./pages/SecurityPosture").then(m => ({ default: m.SecurityPosturePage })));
const AnalyticsPage = React.lazy(() =>
  import("./pages/Analytics").then(m => ({ default: m.AnalyticsPage })));
const ExportPage = React.lazy(() =>
  import("./pages/Export/ExportPage").then(m => ({ default: m.ExportPage })));

// ---------------------------------------------------------------------------
// Error reporting — structured JS error capture, self-hosted (no external SaaS)
//
// In production builds errors are POSTed to the backend's /v1/ingest endpoint
// so they appear alongside agent events.  Set VITE_ERROR_REPORTING=false to
// disable.  If you prefer Sentry, install @sentry/react and initialise it
// here instead (using import.meta.env.VITE_SENTRY_DSN as the DSN gate).
// ---------------------------------------------------------------------------

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const REPORT_ERRORS =
  import.meta.env.PROD && import.meta.env.VITE_ERROR_REPORTING !== "false";

function reportError(error: Error, extra?: Record<string, unknown>) {
  // Always log locally.
  console.error("[aegivis] unhandled error", { error, ...extra });

  if (!REPORT_ERRORS) return;

  // Fire-and-forget POST to backend ingest (auth header is optional for this event type).
  const payload = {
    event_type: "JS_ERROR",
    timestamp_ns: Date.now() * 1_000_000,
    payload: {
      message: error.message,
      name: error.name,
      stack: error.stack?.slice(0, 2_000),  // limit stack size
      url: window.location.pathname,
      user_agent: navigator.userAgent,
      ...extra,
    },
  };
  fetch(`${API_BASE}/v1/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    keepalive: true,   // survives page unload
  }).catch(() => {/* silently ignore reporting failures */});
}

// Capture uncaught errors and unhandled promise rejections outside React.
if (typeof window !== "undefined") {
  window.addEventListener("error", (e) => {
    if (e.error instanceof Error) reportError(e.error, { source: "window.onerror" });
  });
  window.addEventListener("unhandledrejection", (e) => {
    const err = e.reason instanceof Error ? e.reason : new Error(String(e.reason));
    reportError(err, { source: "unhandledrejection" });
  });
}

// ---------------------------------------------------------------------------
// Error boundary — prevents a crash in one page from blanking the whole app
// ---------------------------------------------------------------------------

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: React.ErrorInfo) {
    reportError(error, {
      source: "ErrorBoundary",
      component_stack: info.componentStack?.slice(0, 1_000),
    });
  }
  render() {
    if (this.state.error) {
      return (
        <div className="flex flex-col items-center justify-center h-64 text-gray-500 p-8">
          <div className="text-2xl font-bold text-red-400 mb-2">Something went wrong</div>
          <div className="text-sm text-gray-400 mb-4 max-w-md text-center">
            {this.state.error.message}
          </div>
          <button
            onClick={() => this.setState({ error: null })}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
// Page title — updates document.title based on current route
// ---------------------------------------------------------------------------

const ROUTE_TITLES: Record<string, string> = {
  "/overview":   "Overview",
  "/agents":     "Agents",
  "/sessions":   "Sessions",
  "/violations": "Violations",
  "/security":   "Security Stack",
  "/settings":   "Settings",
  "/benchmark":  "Benchmark",
};

function PageTitle() {
  const { pathname } = useLocation();
  useEffect(() => {
    // Exact match first, then prefix match (e.g. /agents/:id → "Agents")
    const exact = ROUTE_TITLES[pathname];
    if (exact) {
      document.title = `${exact} — Aegivis`;
      return;
    }
    const prefix = Object.keys(ROUTE_TITLES).find((p) => pathname.startsWith(p + "/"));
    document.title = prefix ? `${ROUTE_TITLES[prefix]} — Aegivis` : "Aegivis";
  }, [pathname]);
  return null;
}

// ---------------------------------------------------------------------------

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5_000,
    },
  },
});

// ---------------------------------------------------------------------------
// Sidebar components
// ---------------------------------------------------------------------------

interface SidebarLinkProps {
  to: string;
  icon: React.ElementType;
  children: React.ReactNode;
  badge?: number;
}

function SidebarLink({ to, icon: Icon, children, badge }: SidebarLinkProps) {
  return (
    <NavLink to={to}>
      {({ isActive }) => (
        <span
          className={clsx(
            "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
            isActive
              ? "bg-gradient-to-r from-brand-600/30 to-brand-600/5 text-white"
              : "text-white/60 hover:text-white/90 hover:bg-white/[0.06]",
          )}
        >
          <Icon
            size={16}
            className={isActive ? "text-brand-400" : "opacity-70"}
          />
          <span className="flex-1 truncate">{children}</span>
          {badge !== undefined && badge > 0 && (
            <span className="bg-red-500 text-white text-[10px] font-bold rounded-full px-1.5 py-0.5 leading-none">
              {badge > 9 ? "9+" : badge}
            </span>
          )}
        </span>
      )}
    </NavLink>
  );
}

function NavSection({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-2">
      <p className="text-[10px] font-semibold text-white/30 uppercase tracking-widest px-3 mb-1 mt-4">
        {label}
      </p>
      <div className="space-y-0.5">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Edition badge — shown in sidebar footer
// ---------------------------------------------------------------------------

const EDITION = (import.meta.env.VITE_EDITION ?? "community").toLowerCase();
const IS_ENTERPRISE = EDITION === "enterprise";

function EditionBadge() {
  return (
    <div className="px-4 py-3 border-t border-white/[0.06] flex items-center justify-between">
      <span className="text-white/30 text-xs">v1.0.0</span>
      {IS_ENTERPRISE ? (
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-amber-400/20 text-amber-300 border border-amber-400/30 tracking-wide">
          ENTERPRISE
        </span>
      ) : (
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-white/[0.06] text-white/30 border border-white/[0.08] tracking-wide">
          COMMUNITY
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

function Layout({ children }: { children: React.ReactNode }) {
  // Approvals badge — polled only in enterprise mode
  const { data: pendingData } = useQuery({
    queryKey: ["approvals-count"],
    queryFn: () => import("./api/client").then(m => m.listApprovals({ status: "pending", limit: 1 })),
    refetchInterval: 10_000,
    staleTime: 0,
    enabled: IS_ENTERPRISE,
  });
  const pendingCount = IS_ENTERPRISE ? (pendingData?.count ?? 0) : 0;

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* ── Sidebar ────────────────────────────────────────────────── */}
      <aside className="fixed inset-y-0 left-0 w-60 bg-[#09162e] flex flex-col z-30">
        {/* Logo row */}
        <div className="flex items-center gap-2.5 px-4 py-4 border-b border-white/[0.06]">
          <AegivisLogo size={28} />
          <span className="text-white font-bold text-base tracking-tight">Aegivis</span>
        </div>

        {/* Nav items */}
        <nav className="flex-1 overflow-y-auto scrollbar-sidebar px-2 py-2">
          <NavSection label="Monitor">
            <SidebarLink to="/overview" icon={LayoutDashboard}>Overview</SidebarLink>
            <SidebarLink to="/agents" icon={Bot}>Agents</SidebarLink>
            <SidebarLink to="/sessions" icon={Activity}>Sessions</SidebarLink>
            <SidebarLink to="/violations" icon={ShieldAlert}>Violations</SidebarLink>
            {IS_ENTERPRISE && (
              <SidebarLink to="/approvals" icon={Bell} badge={pendingCount}>Approvals</SidebarLink>
            )}
          </NavSection>

          <NavSection label="Security">
            <SidebarLink to="/security" icon={Layers}>Security Stack</SidebarLink>
            {IS_ENTERPRISE && (
              <SidebarLink to="/posture" icon={TrendingUp}>Posture</SidebarLink>
            )}
            <SidebarLink to="/settings" icon={Cog}>Settings</SidebarLink>
          </NavSection>

          <NavSection label="Insights">
            {IS_ENTERPRISE && (
              <SidebarLink to="/analytics" icon={BarChart2}>Analytics</SidebarLink>
            )}
            {IS_ENTERPRISE && (
              <SidebarLink to="/export" icon={Download}>Export</SidebarLink>
            )}
            <SidebarLink to="/benchmark" icon={Zap}>Benchmark</SidebarLink>
          </NavSection>
        </nav>

        {/* Edition footer */}
        <EditionBadge />
      </aside>

      {/* ── Main content ───────────────────────────────────────────── */}
      <main className="flex-1 ml-60 min-h-screen">
        <PageTitle />
        <ErrorBoundary>{children}</ErrorBoundary>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Navigate to="/overview" replace />} />

            {/* Community routes */}
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/sessions" element={<Sessions />} />
            {/* Legacy redirect for old /events URL */}
            <Route path="/events" element={<Navigate to="/sessions" replace />} />
            <Route path="/violations" element={<ViolationsPage />} />
            <Route path="/security" element={<SecurityStack />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />

            {/* Enterprise routes — only active when VITE_EDITION=enterprise */}
            {IS_ENTERPRISE && (
              <Route path="/agents/:agentId" element={
                <React.Suspense fallback={null}><AgentDetail /></React.Suspense>
              } />
            )}
            {IS_ENTERPRISE && (
              <Route path="/approvals" element={
                <React.Suspense fallback={null}><ApprovalsPage /></React.Suspense>
              } />
            )}
            {IS_ENTERPRISE && (
              <Route path="/posture" element={
                <React.Suspense fallback={null}><SecurityPosturePage /></React.Suspense>
              } />
            )}
            {IS_ENTERPRISE && (
              <Route path="/analytics" element={
                <React.Suspense fallback={null}><AnalyticsPage /></React.Suspense>
              } />
            )}
            {IS_ENTERPRISE && (
              <Route path="/export" element={
                <React.Suspense fallback={null}><ExportPage /></React.Suspense>
              } />
            )}

            {/* Session detail routes */}
            <Route path="/sessions/:sessionId" element={<SessionDetail />} />
            <Route path="/sessions/:sessionId/forensics" element={<Forensics />} />
            <Route path="/sessions/:sessionId/compliance" element={<Compliance />} />

            {/* 404 — catch-all */}
            <Route path="*" element={
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <div className="text-4xl font-bold text-gray-300 mb-2">404</div>
                <div className="text-sm">Page not found</div>
                <a href="/overview" className="mt-4 text-blue-500 text-sm hover:underline">Back to Overview</a>
              </div>
            } />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
