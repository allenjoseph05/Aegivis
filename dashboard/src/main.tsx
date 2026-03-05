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
import { ExportPage } from "./pages/Export";
import { BenchmarkPage } from "./pages/Benchmark";
import { AgentsPage } from "./pages/Agents";
import { AgentDetail } from "./pages/Agents/AgentDetail";
import { ViolationsPage } from "./pages/Violations";
import { SettingsPage } from "./pages/Settings";
import { AnalyticsPage } from "./pages/Analytics";
import { ApprovalsPage } from "./pages/Approvals";
import { SecurityPosturePage } from "./pages/SecurityPosture";
import { listApprovals } from "./api/client";

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
  "/approvals":  "Approvals",
  "/security":   "Security Stack",
  "/posture":    "Security Posture",
  "/settings":   "Settings",
  "/analytics":  "Analytics",
  "/export":     "Export",
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

function Layout({ children }: { children: React.ReactNode }) {
  // Pending HITL approvals badge — poll every 10 s
  const { data: pendingData } = useQuery({
    queryKey: ["approvals-count"],
    queryFn: () => listApprovals({ status: "pending", limit: 1 }),
    refetchInterval: 10_000,
    staleTime: 0,
  });
  const pendingCount = pendingData?.count ?? 0;

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
            <SidebarLink to="/approvals" icon={Bell} badge={pendingCount}>Approvals</SidebarLink>
          </NavSection>

          <NavSection label="Security">
            <SidebarLink to="/security" icon={Layers}>Security Stack</SidebarLink>
            <SidebarLink to="/posture" icon={TrendingUp}>Posture</SidebarLink>
            <SidebarLink to="/settings" icon={Cog}>Settings</SidebarLink>
          </NavSection>

          <NavSection label="Insights">
            <SidebarLink to="/analytics" icon={BarChart2}>Analytics</SidebarLink>
            <SidebarLink to="/export" icon={Download}>Export</SidebarLink>
            <SidebarLink to="/benchmark" icon={Zap}>Benchmark</SidebarLink>
          </NavSection>
        </nav>

        {/* Version footer */}
        <div className="px-4 py-3 border-t border-white/[0.06]">
          <span className="text-white/30 text-xs">v1.0.0</span>
        </div>
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

            {/* Primary nav */}
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/agents/:agentId" element={<AgentDetail />} />
            <Route path="/sessions" element={<Sessions />} />
            {/* Legacy redirect for old /events URL */}
            <Route path="/events" element={<Navigate to="/sessions" replace />} />
            <Route path="/violations" element={<ViolationsPage />} />
            <Route path="/approvals" element={<ApprovalsPage />} />
            <Route path="/security" element={<SecurityStack />} />
            <Route path="/posture" element={<SecurityPosturePage />} />
            <Route path="/settings" element={<SettingsPage />} />

            {/* Secondary nav */}
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/export" element={<ExportPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />

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
