import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, NavLink, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";

// Pages
import { OverviewPage } from "./pages/Overview";
import { Sessions } from "./pages/Sessions";
import { SessionDetail } from "./pages/SessionDetail";
import { Forensics } from "./pages/Forensics";
import { Compliance } from "./pages/Compliance";
import { Metrics } from "./pages/Metrics";
import { Security } from "./pages/Security";
import { TopologyPage } from "./pages/Topology";
import { ExportPage } from "./pages/Export";
import { BenchmarkPage } from "./pages/Benchmark";
import { PolicyBuilderPage } from "./pages/PolicyBuilder";
import { AgentsPage } from "./pages/Agents";
import { BaselinesPage } from "./pages/Baselines";
import { ViolationsPage } from "./pages/Violations";
import { AlertsPage } from "./pages/Alerts";
import { SettingsPage } from "./pages/Settings";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5_000,
    },
  },
});

const NAV_LINK = ({ isActive }: { isActive: boolean }) =>
  `px-3 py-1.5 rounded text-sm transition-colors ${
    isActive
      ? "bg-white/10 text-white"
      : "text-gray-400 hover:text-white hover:bg-white/5"
  }`;

const NAV_LINK_SM = ({ isActive }: { isActive: boolean }) =>
  `px-2.5 py-1.5 rounded text-xs transition-colors ${
    isActive
      ? "bg-white/10 text-white"
      : "text-gray-500 hover:text-gray-300 hover:bg-white/5"
  }`;

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-gray-900 text-white px-6 shadow-lg">
        {/* Primary row */}
        <div className="flex items-center gap-6 py-3">
          {/* Logo */}
          <div className="flex items-center gap-2 font-bold text-lg flex-shrink-0">
            <span className="text-2xl">⬛</span>
            <span>AgentBlackBox</span>
          </div>

          {/* Primary nav */}
          <div className="flex items-center gap-1 ml-4 flex-wrap">
            <NavLink to="/overview" className={NAV_LINK}>Overview</NavLink>
            <NavLink to="/agents" className={NAV_LINK}>Agents</NavLink>
            <NavLink to="/events" className={NAV_LINK}>Events</NavLink>
            <NavLink to="/violations" className={NAV_LINK}>Violations</NavLink>
            <NavLink to="/security" className={NAV_LINK}>Security Stack</NavLink>
            <NavLink to="/alerts" className={NAV_LINK}>Alerts</NavLink>
            <NavLink to="/settings" className={NAV_LINK}>Settings</NavLink>
          </div>

          <div className="ml-auto text-xs text-gray-500 flex-shrink-0">v1.0.0</div>
        </div>

        {/* Secondary row — smaller, lighter */}
        <div className="flex items-center gap-1 pb-1.5 border-t border-white/5 pt-1.5">
          <span className="text-xs text-gray-600 mr-1">More:</span>
          <NavLink to="/baselines" className={NAV_LINK_SM}>Baselines</NavLink>
          <NavLink to="/policy" className={NAV_LINK_SM}>Policies</NavLink>
          <NavLink to="/metrics" className={NAV_LINK_SM}>Metrics</NavLink>
          <NavLink to="/export" className={NAV_LINK_SM}>Export</NavLink>
          <NavLink to="/benchmark" className={NAV_LINK_SM}>Benchmark</NavLink>
          <NavLink to="/topology" className={NAV_LINK_SM}>Topology</NavLink>
        </div>
      </nav>

      <main>{children}</main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Layout>
          <Routes>
            {/* Default: redirect / to /overview */}
            <Route path="/" element={<Navigate to="/overview" replace />} />

            {/* Primary nav */}
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/events" element={<Sessions />} />
            <Route path="/violations" element={<ViolationsPage />} />
            <Route path="/security" element={<Security />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/settings" element={<SettingsPage />} />

            {/* Secondary nav */}
            <Route path="/baselines" element={<BaselinesPage />} />
            <Route path="/policy" element={<PolicyBuilderPage />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/export" element={<ExportPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
            <Route path="/topology" element={<TopologyPage />} />

            {/* Session detail routes (linked from Events + Overview) */}
            <Route path="/sessions/:sessionId" element={<SessionDetail />} />
            <Route path="/sessions/:sessionId/forensics" element={<Forensics />} />
            <Route path="/sessions/:sessionId/compliance" element={<Compliance />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
