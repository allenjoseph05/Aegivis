/**
 * Agent Onboarding Wizard — Phase 7
 * 3-step modal: Identity → Security Preset → Proxy Setup + Connection Verifier
 */
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  X, ChevronRight, ChevronLeft, Check,
  CheckCircle, RefreshCw, Copy,
} from "lucide-react";
import { registerAgent, listSessions } from "../../api/client";
import { PROXY_URL } from "../../api/client";

// ─── Preset data (mirrors SecurityStack) ──────────────────────────────────────

const PRESETS = [
  {
    id: "minimal",
    name: "Minimal",
    subtitle: "Latency-sensitive",
    color: "border-gray-200",
    badge: "bg-gray-100 text-gray-700",
    description: "Structural scanner + Canary + Tool Baseline. <3ms overhead.",
    layers: 4,
  },
  {
    id: "balanced",
    name: "Balanced",
    subtitle: "Recommended",
    color: "border-blue-400 ring-2 ring-blue-200",
    badge: "bg-blue-600 text-white",
    description: "All structural + Credential + Taint + RCE/SSRF + PII. ~5ms overhead.",
    layers: 12,
    recommended: true,
  },
  {
    id: "maximum",
    name: "Maximum",
    subtitle: "High-security",
    color: "border-red-300",
    badge: "bg-red-600 text-white",
    description: "Everything + Sync ML classifier. ~55ms overhead.",
    layers: 15,
  },
];

// ─── Step 1 — Identity ─────────────────────────────────────────────────────────

interface IdentityForm {
  agent_id: string;
  name: string;
  declared_purpose: string;
  owner: string;
  allowed_tools_raw: string;
}

function StepIdentity({
  form,
  setForm,
}: {
  form: IdentityForm;
  setForm: (f: IdentityForm) => void;
}) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-500">
        Tell us about your agent so we can track it in the registry.
      </p>

      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">
          Agent ID <span className="text-red-500">*</span>
        </label>
        <input
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="e.g. research-bot-v1"
          value={form.agent_id}
          onChange={(e) => setForm({ ...form, agent_id: e.target.value })}
        />
        <p className="text-xs text-gray-400 mt-0.5">
          Must match the <code className="bg-gray-100 px-1 rounded">x-aegivis-agent-id</code> header
          your agent sends.
        </p>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">
          Display Name <span className="text-red-500">*</span>
        </label>
        <input
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="e.g. Research Assistant"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
        />
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-700 mb-1">
          Declared Purpose
        </label>
        <textarea
          rows={2}
          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
          placeholder="What is this agent supposed to do?"
          value={form.declared_purpose}
          onChange={(e) => setForm({ ...form, declared_purpose: e.target.value })}
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">Owner</label>
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="team or email"
            value={form.owner}
            onChange={(e) => setForm({ ...form, owner: e.target.value })}
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-700 mb-1">
            Allowed Tools
          </label>
          <input
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="web_search, read_file"
            value={form.allowed_tools_raw}
            onChange={(e) => setForm({ ...form, allowed_tools_raw: e.target.value })}
          />
          <p className="text-xs text-gray-400 mt-0.5">Comma-separated</p>
        </div>
      </div>
    </div>
  );
}

// ─── Step 2 — Security Preset ──────────────────────────────────────────────────

function StepPreset({
  selected,
  setSelected,
}: {
  selected: string;
  setSelected: (p: string) => void;
}) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-500">
        Choose a security profile. You can tune individual layers later in the Security Stack page.
      </p>
      <div className="space-y-3">
        {PRESETS.map((p) => (
          <button
            key={p.id}
            onClick={() => setSelected(p.id)}
            className={`w-full text-left border-2 rounded-xl p-4 transition-all ${
              selected === p.id ? p.color + " bg-gray-50" : "border-gray-200 hover:border-gray-300"
            }`}
          >
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <span className={`text-xs px-2 py-0.5 rounded font-bold ${p.badge}`}>
                  {p.name}
                </span>
                <span className="text-xs text-gray-500">{p.subtitle}</span>
                {p.recommended && (
                  <span className="text-xs text-blue-600 font-medium">★ Recommended</span>
                )}
              </div>
              {selected === p.id && <Check size={16} className="text-blue-600" />}
            </div>
            <p className="text-sm text-gray-600">{p.description}</p>
            <p className="text-xs text-gray-400 mt-1">{p.layers} security layers active</p>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Step 3 — Proxy Setup ──────────────────────────────────────────────────────

function CodeBlock({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);

  function copy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="relative">
      <pre className="text-xs bg-gray-900 text-green-300 rounded-lg p-4 overflow-x-auto">
        {code}
      </pre>
      <button
        onClick={copy}
        className="absolute top-2 right-2 p-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
        title="Copy"
      >
        {copied ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
      </button>
    </div>
  );
}

function StepSetup({ agentId }: { agentId: string }) {
  const proxyBase = PROXY_URL || "http://localhost:8080";

  // Poll for the first session from this agent
  const sessionQ = useQuery({
    queryKey: ["sessions", agentId, "onboarding"],
    queryFn: () => listSessions({ agent_id: agentId, limit: 1 }),
    refetchInterval: 3_000,
    staleTime: 0,
  });

  const hasConnected = (sessionQ.data?.count ?? 0) > 0;

  const openaiCode = `import openai

client = openai.OpenAI(
    base_url="${proxyBase}/openai/v1",
    api_key="your-openai-key",
    default_headers={
        "x-aegivis-agent-id": "${agentId}",
        "x-aegivis-session-id": "session-001",
    }
)`;

  const anthropicCode = `import anthropic

client = anthropic.Anthropic(
    base_url="${proxyBase}/anthropic",
    api_key="your-anthropic-key",
    default_headers={
        "x-aegivis-agent-id": "${agentId}",
        "x-aegivis-session-id": "session-001",
    }
)`;

  return (
    <div className="space-y-5">
      <p className="text-sm text-gray-500">
        Point your agent at the proxy by changing the base URL. No other changes required.
      </p>

      <div>
        <p className="text-xs font-medium text-gray-700 mb-2">OpenAI SDK</p>
        <CodeBlock code={openaiCode} />
      </div>
      <div>
        <p className="text-xs font-medium text-gray-700 mb-2">Anthropic SDK</p>
        <CodeBlock code={anthropicCode} />
      </div>

      {/* Connection verifier */}
      <div
        className={`rounded-xl border px-5 py-4 flex items-center gap-3 transition-colors ${
          hasConnected
            ? "border-green-200 bg-green-50"
            : "border-gray-200 bg-gray-50"
        }`}
      >
        {hasConnected ? (
          <CheckCircle size={20} className="text-green-600 flex-shrink-0" />
        ) : (
          <RefreshCw size={20} className="text-gray-400 animate-spin flex-shrink-0" />
        )}
        <div>
          <p className={`text-sm font-semibold ${hasConnected ? "text-green-800" : "text-gray-600"}`}>
            {hasConnected ? "Connection verified!" : "Waiting for first request…"}
          </p>
          <p className={`text-xs mt-0.5 ${hasConnected ? "text-green-600" : "text-gray-400"}`}>
            {hasConnected
              ? `Agent "${agentId}" has connected to the proxy.`
              : "Make one API call through the proxy to verify the setup."}
          </p>
        </div>
      </div>
    </div>
  );
}

// ─── Wizard shell ──────────────────────────────────────────────────────────────

const STEPS = ["Identity", "Security Profile", "Proxy Setup"];

export function OnboardingWizard({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const [step, setStep] = useState(0);
  const [registeredAgentId, setRegisteredAgentId] = useState("");
  const [preset, setPreset] = useState("balanced");
  const [form, setForm] = useState<IdentityForm>({
    agent_id: "",
    name: "",
    declared_purpose: "",
    owner: "",
    allowed_tools_raw: "",
  });
  const [error, setError] = useState<string | null>(null);

  const registerMut = useMutation({
    mutationFn: registerAgent,
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ["agents"] });
      setRegisteredAgentId(vars.agent_id);
      setStep(2);
    },
    onError: (e: unknown) => {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(msg ?? "Registration failed");
    },
  });

  function canAdvance() {
    if (step === 0) return form.agent_id.trim().length > 0 && form.name.trim().length > 0;
    if (step === 1) return true;
    return true;
  }

  function handleNext() {
    setError(null);
    if (step === 0) {
      // Validate then move to step 1
      setStep(1);
    } else if (step === 1) {
      // Register agent then move to step 2
      const allowed_tools = form.allowed_tools_raw
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      registerMut.mutate({
        agent_id: form.agent_id.trim(),
        name: form.name.trim(),
        declared_purpose: form.declared_purpose.trim() || undefined,
        allowed_tools,
        owner: form.owner.trim() || undefined,
      });
    } else {
      onClose();
    }
  }

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 flex-shrink-0">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Register New Agent</h2>
            <p className="text-xs text-gray-400 mt-0.5">Step {step + 1} of {STEPS.length}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-gray-100 text-gray-400"
          >
            <X size={16} />
          </button>
        </div>

        {/* Step indicator */}
        <div className="flex px-6 py-3 gap-2 flex-shrink-0">
          {STEPS.map((s, i) => (
            <div key={s} className="flex items-center gap-2 flex-1">
              <div
                className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0 transition-colors ${
                  i < step
                    ? "bg-green-500 text-white"
                    : i === step
                    ? "bg-blue-600 text-white"
                    : "bg-gray-100 text-gray-400"
                }`}
              >
                {i < step ? <Check size={12} /> : i + 1}
              </div>
              <span className={`text-xs font-medium ${i === step ? "text-gray-900" : "text-gray-400"}`}>
                {s}
              </span>
              {i < STEPS.length - 1 && (
                <div className={`flex-1 h-px ${i < step ? "bg-green-300" : "bg-gray-100"}`} />
              )}
            </div>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {error && (
            <div className="mb-4 text-sm text-red-600 bg-red-50 rounded-lg p-3">{error}</div>
          )}
          {step === 0 && <StepIdentity form={form} setForm={setForm} />}
          {step === 1 && <StepPreset selected={preset} setSelected={setPreset} />}
          {step === 2 && <StepSetup agentId={registeredAgentId || form.agent_id} />}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-100 flex-shrink-0">
          <button
            onClick={() => step > 0 && step < 2 && setStep((s) => s - 1)}
            disabled={step === 0 || step === 2}
            className="flex items-center gap-1 px-4 py-2 text-sm text-gray-600 hover:text-gray-900 disabled:opacity-30 disabled:cursor-not-allowed"
          >
            <ChevronLeft size={14} /> Back
          </button>
          <button
            onClick={handleNext}
            disabled={!canAdvance() || registerMut.isPending}
            className="flex items-center gap-1 px-5 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg disabled:opacity-60 transition-colors"
          >
            {registerMut.isPending ? (
              <><RefreshCw size={14} className="animate-spin" /> Registering…</>
            ) : step === 2 ? (
              <>Done <Check size={14} /></>
            ) : (
              <>Next <ChevronRight size={14} /></>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
