import { Settings, Database, Shield, Key, Link, type LucideIcon } from "lucide-react";

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: LucideIcon;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100 flex items-center gap-2">
        <Icon size={16} className="text-gray-500" />
        <h2 className="font-semibold text-gray-800">{title}</h2>
      </div>
      <div className="p-5">{children}</div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-50 last:border-0">
      <span className="text-sm text-gray-500">{label}</span>
      <span className="text-sm font-mono text-gray-700">{value}</span>
    </div>
  );
}

export function SettingsPage() {
  const apiKey = import.meta.env.VITE_API_KEY || "dev-dashboard-key";
  const backendUrl = import.meta.env.VITE_BACKEND_URL || window.location.origin;

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-sm text-gray-500 mt-1">
          Global configuration for your AgentBlackBox deployment.
        </p>
      </div>

      {/* Connection */}
      <Section icon={Link} title="Connection">
        <Row label="Backend URL" value={backendUrl} />
        <Row label="API key" value={`${apiKey.slice(0, 8)}${"•".repeat(Math.max(0, apiKey.length - 8))}`} />
        <Row label="Organisation" value="default-org" />
      </Section>

      {/* Security defaults */}
      <Section icon={Shield} title="Security Defaults">
        <p className="text-sm text-gray-500 mb-3">
          Default security profile applied to all new agents. Per-agent overrides coming in Phase 2.
        </p>
        <div className="space-y-2">
          {[
            "Structural injection scanner",
            "Async ML classifier",
            "PII masking",
            "Canary detection",
            "Tool baseline enforcement",
            "Taint tracking",
          ].map((layer) => (
            <div key={layer} className="flex items-center justify-between py-1.5">
              <span className="text-sm text-gray-700">{layer}</span>
              <span className="text-xs px-2 py-0.5 bg-green-50 text-green-700 rounded-full font-medium">
                Enabled
              </span>
            </div>
          ))}
        </div>
      </Section>

      {/* API keys */}
      <Section icon={Key} title="API Keys">
        <p className="text-sm text-gray-500 mb-4">
          API keys are provisioned in the database. To add a new organisation key, run:
        </p>
        <pre className="text-xs bg-gray-900 text-green-300 rounded-lg p-4 overflow-x-auto">
          {`docker exec abb-postgres psql -U abb -d agentblackbox -c \\
  "INSERT INTO api_keys (org_id, key_hash, name)
   VALUES ('your-org', encode(sha256('your-secret'::bytea),'hex'), 'Org Name');"`}
        </pre>
      </Section>

      {/* Integrations */}
      <Section icon={Database} title="Integrations">
        <p className="text-sm text-gray-500 mb-3">
          Configure via environment variables in <code className="font-mono bg-gray-100 px-1 rounded">docker-compose.yml</code>.
        </p>
        <div className="space-y-1 text-sm text-gray-600">
          {[
            ["SIEM — Splunk HEC", "ABB_SPLUNK_HEC_URL"],
            ["SIEM — Elasticsearch", "ABB_ELASTIC_URL"],
            ["Alerts — Slack", "ABB_SLACK_WEBHOOK_URL"],
            ["Alerts — Email (SMTP)", "ABB_SMTP_HOST"],
            ["Alerts — PagerDuty", "ABB_PAGERDUTY_ROUTING_KEY"],
            ["Redis streams", "ABB_REDIS_URL"],
          ].map(([name, envVar]) => (
            <div key={name} className="flex items-center justify-between py-1.5 border-b border-gray-50 last:border-0">
              <span>{name}</span>
              <code className="text-xs text-gray-400 bg-gray-50 px-1.5 py-0.5 rounded">{envVar}</code>
            </div>
          ))}
        </div>
      </Section>

      <p className="text-xs text-gray-400 text-center">
        Full settings configuration UI coming in Phase 6.
      </p>
    </div>
  );
}
