import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import "./index.css";

import { Sessions } from "./pages/Sessions";
import { SessionDetail } from "./pages/SessionDetail";
import { Forensics } from "./pages/Forensics";
import { Compliance } from "./pages/Compliance";

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
          </Routes>
        </Layout>
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);
