import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";

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

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 5_000,
    },
  },
});

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top nav */}
      <nav className="bg-gray-900 text-white px-6 py-3 flex items-center gap-6 shadow-lg">
        <div className="flex items-center gap-2 font-bold text-lg">
          <span className="text-2xl">⬛</span>
          <span>AgentBlackBox</span>
        </div>
        <div className="flex items-center gap-1 ml-4">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Sessions
          </NavLink>
          <NavLink
            to="/metrics"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Metrics
          </NavLink>
          <NavLink
            to="/security"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Security
          </NavLink>
          <NavLink
            to="/topology"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Topology
          </NavLink>
          <NavLink
            to="/export"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Export
          </NavLink>
          <NavLink
            to="/policy"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Policy
          </NavLink>
          <NavLink
            to="/agents"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Agents
          </NavLink>
          <NavLink
            to="/benchmark"
            className={({ isActive }) =>
              `px-3 py-1.5 rounded text-sm transition-colors ${
                isActive
                  ? "bg-white/10 text-white"
                  : "text-gray-400 hover:text-white hover:bg-white/5"
              }`
            }
          >
            Benchmark
          </NavLink>
        </div>
        <div className="ml-auto text-xs text-gray-500">v1.0.0</div>
      </nav>

      {/* Content */}
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
            <Route path="/" element={<Sessions />} />
            <Route path="/sessions/:sessionId" element={<SessionDetail />} />
            <Route path="/sessions/:sessionId/forensics" element={<Forensics />} />
            <Route path="/sessions/:sessionId/compliance" element={<Compliance />} />
            <Route path="/metrics" element={<Metrics />} />
            <Route path="/security" element={<Security />} />
            <Route path="/topology" element={<TopologyPage />} />
            <Route path="/export" element={<ExportPage />} />
            <Route path="/policy" element={<PolicyBuilderPage />} />
            <Route path="/agents" element={<AgentsPage />} />
            <Route path="/benchmark" element={<BenchmarkPage />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
