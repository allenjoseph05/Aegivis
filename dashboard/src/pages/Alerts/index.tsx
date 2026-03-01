import { Bell, Slack, Mail, Zap, Webhook, ChevronRight, type LucideIcon } from "lucide-react";

function ChannelCard({
  icon: Icon,
  name,
  description,
  comingSoon,
}: {
  icon: LucideIcon;
  name: string;
  description: string;
  comingSoon?: boolean;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5 shadow-sm flex items-start gap-4">
      <div className="p-2.5 bg-gray-100 rounded-lg flex-shrink-0">
        <Icon size={20} className="text-gray-600" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <p className="font-semibold text-gray-800">{name}</p>
          {comingSoon && (
            <span className="text-xs px-2 py-0.5 bg-blue-50 text-blue-600 rounded-full font-medium">
              Coming soon
            </span>
          )}
        </div>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
      {!comingSoon && (
        <ChevronRight size={16} className="text-gray-400 flex-shrink-0 mt-1" />
      )}
    </div>
  );
}

export function AlertsPage() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Alerts</h1>
        <p className="text-sm text-gray-500 mt-1">
          Configure where violation alerts are delivered and which categories trigger them.
        </p>
      </div>

      {/* Alert channels */}
      <div>
        <h2 className="text-base font-semibold text-gray-700 mb-4">Alert Channels</h2>
        <div className="space-y-3">
          <ChannelCard
            icon={Slack}
            name="Slack"
            description="Send alerts to a Slack channel via webhook URL. Configure per-severity routing."
            comingSoon
          />
          <ChannelCard
            icon={Mail}
            name="Email"
            description="Send alert emails via SMTP. Supports custom recipients per violation category."
            comingSoon
          />
          <ChannelCard
            icon={Zap}
            name="PagerDuty"
            description="Create incidents in PagerDuty for BLOCK events. Requires Events API v2 key."
            comingSoon
          />
          <ChannelCard
            icon={Webhook}
            name="Webhook"
            description="POST violation payloads to any HTTP endpoint. Signed with HMAC-SHA256."
            comingSoon
          />
        </div>
      </div>

      {/* Coming soon note */}
      <div className="bg-blue-50 border border-blue-100 rounded-xl p-5 flex items-start gap-3">
        <Bell size={18} className="text-blue-500 mt-0.5 flex-shrink-0" />
        <div>
          <p className="text-sm font-semibold text-blue-800">Alert configuration is coming in Phase 4</p>
          <p className="text-sm text-blue-600 mt-1">
            You'll be able to configure Slack webhooks, email, PagerDuty, and generic webhooks with
            per-category routing rules, cooldown windows, severity filters, and quiet hours.
          </p>
          <p className="text-sm text-blue-600 mt-2">
            In the meantime, the backend already sends alerts via the channels configured in{" "}
            <code className="font-mono bg-blue-100 px-1 rounded">ABB_SLACK_WEBHOOK_URL</code>,{" "}
            <code className="font-mono bg-blue-100 px-1 rounded">ABB_SMTP_HOST</code>, and{" "}
            <code className="font-mono bg-blue-100 px-1 rounded">ABB_PAGERDUTY_ROUTING_KEY</code>{" "}
            environment variables.
          </p>
        </div>
      </div>
    </div>
  );
}
