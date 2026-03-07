"""Proxy configuration via environment variables."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ProxySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIVIS_", case_sensitive=False)

    # Backend ingestion endpoint
    backend_url: str = "http://localhost:8000"
    backend_api_key: str = "dev-proxy-key"

    # Organization (used in event envelope)
    org_id: str = "default-org"

    # Proxy server
    host: str = "0.0.0.0"
    port: int = 8080
    # CORS — locked to dashboard origins by default.
    # Override in production: AEGIVIS_CORS_ORIGINS=https://yourdomain.com
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Real LLM provider URLs (defaults are the public APIs)
    openai_upstream: str = "https://api.openai.com"
    anthropic_upstream: str = "https://api.anthropic.com"
    google_upstream: str = "https://generativelanguage.googleapis.com"
    azure_upstream: str = ""         # set to https://{resource}.openai.azure.com
    ollama_upstream: str = "http://localhost:11434"

    # Batch transport
    batch_size: int = 10
    batch_flush_interval_s: float = 2.0
    transport_timeout_s: float = 10.0

    # PII settings
    pii_enabled: bool = True
    pii_language: str = "en"

    # Hash chain
    checkpoint_interval: int = 1000   # CHECKPOINT every N events per session

    # Policy engine
    # Path to a custom YAML policy file. If empty, uses proxy/app/policies/default.yaml.
    policy_yaml: str = ""
    # Comma-separated list of rules to disable by name (e.g. "high-latency-alert,pii-in-llm-request")
    policy_disabled_rules: str = ""

    # Tool permissions engine (Phase 3.1 Iteration 3)
    # Path to a tool permissions YAML file.
    # If empty, proxy/app/policies/tool_permissions.yaml is loaded automatically.
    # All rules in the bundled file are disabled by default — enable as needed.
    tool_permissions_yaml: str = ""

    # Agent identity
    # Comma-separated agent_id:key pairs. e.g. "my-agent:aegivis_abc123,other-agent:aegivis_def456"
    agent_keys: str = ""
    # If true, requests without a valid X-Aegivis-Agent-Key are rejected with 401
    require_agent_key: bool = False

    # Violations reporting — send policy violations to backend
    violations_enabled: bool = True

    # Reliability: local disk buffer when backend is unreachable
    # Path to a SQLite file used to persist unsent events. Empty = disabled.
    buffer_db_path: str = "aegivis_buffer.db"   # relative to CWD; set "" to disable
    buffer_retry_interval_s: float = 30.0   # how often to retry flushing buffered events
    buffer_max_events: int = 50_000         # max events to buffer before dropping

    # Reliability: session state persistence across proxy restarts
    # Path to a JSON file for session state. Empty = disabled (in-memory only).
    session_state_path: str = "aegivis_sessions.json"

    # -------------------------------------------------------------------------
    # Security scanning
    # -------------------------------------------------------------------------
    # Prompt-injection score thresholds (enforcement/structural scanner).
    # score >= block_threshold => BLOCK (request rejected before forwarding to LLM)
    # score >= alert_threshold => ALERT (violation recorded, request allowed)
    security_injection_block_threshold: float = 0.80
    security_injection_alert_threshold: float = 0.50

    # Credential confidence threshold. Matches below this value are ignored.
    security_credential_confidence_threshold: float = 0.40

    # -------------------------------------------------------------------------
    # Canary token injection (anti-exfiltration)
    # -------------------------------------------------------------------------
    # Inject a 256-bit random canary into each LLM system prompt.
    # If the canary appears in the response, a prompt injection exfiltration is
    # confirmed with zero false positives.
    security_canary_enabled: bool = True

    # -------------------------------------------------------------------------
    # Spotlighting (Indirect Prompt Injection defense)
    # -------------------------------------------------------------------------
    # Wrap all tool/function result messages in randomized delimiters. Reduces
    # IPI success rates from >50% to <2% (Microsoft Research, 2024).
    security_spotlighting_enabled: bool = True

    # -------------------------------------------------------------------------
    # Output scanning (post-response threat detection)
    # -------------------------------------------------------------------------
    # Scan every LLM response for: canary leakage, relay injection patterns,
    # prompt echo, and credentials in output.
    security_output_scanning_enabled: bool = True

    # -------------------------------------------------------------------------
    # Management API authentication
    # -------------------------------------------------------------------------
    # When non-empty, all management endpoints (/policies/reload,
    # /tool-permissions/reload) require this key in the X-Aegivis-Admin-Key header.
    # Set to a long random string in production.
    # Example: AEGIVIS_ADMIN_API_KEY=$(openssl rand -hex 32)
    # Leave empty ("") to disable admin auth (suitable for local development).
    admin_api_key: str = ""

    # -------------------------------------------------------------------------
    # RCE scanner (Phase 3.1)
    # -------------------------------------------------------------------------
    # Minimum confidence to flag a tool call argument as an RCE attempt.
    security_rce_confidence_threshold: float = 0.70

    # -------------------------------------------------------------------------
    # SSRF scanner (Phase 3.1)
    # -------------------------------------------------------------------------
    # Comma-separated list of allowed domains/hostnames for outbound tool calls.
    # When non-empty, any hostname NOT in this list is flagged as suspicious.
    # Leave empty for audit-only mode (private-IP and cloud-metadata checks
    # still apply even with an empty allowlist).
    # Example: "api.openai.com,api.anthropic.com,httpbin.org"
    security_ssrf_allowed_domains: str = ""

    # Resolve hostnames via DNS during SSRF scanning to catch rebinding attacks.
    # Disable if DNS lookups cause latency issues in your environment.
    security_ssrf_enable_dns_resolution: bool = True

    # -------------------------------------------------------------------------
    # Tool output scanning (Phase 3.3)
    # -------------------------------------------------------------------------
    # Scan tool return values for injection before they re-enter LLM context.
    # Closes indirect-injection gap (EchoLeak CVE-2025-32711).
    security_tool_output_scanning_enabled: bool = True
    # Maximum bytes to scan per tool output (long outputs truncated).
    security_tool_output_max_bytes: int = 8192

    # -------------------------------------------------------------------------
    # MCP tool definition scanning (Phase 3.3)
    # -------------------------------------------------------------------------
    # Scan the tools[] array for malicious tool definitions:
    # name traversal, description injection, shadow overloading.
    security_mcp_scanning_enabled: bool = True

    # -------------------------------------------------------------------------
    # Steganographic injection scanner (Phase 9E)
    # -------------------------------------------------------------------------
    # Detect invisible Unicode characters (tag chars U+E0000-E007F, zero-width
    # chars, RTL direction overrides) used to hide injection payloads.
    # Near-zero FPR: tag chars have no legitimate use; two-gate rule
    # (invisible chars + injection in stripped text) keeps BLOCK FPR < 0.5%.
    security_unicode_stego_enabled: bool = True

    # Scan PDF and DOCX documents for hidden text layers, annotations, metadata
    # injection, and macros. Requires: pip install 'aegivis-proxy[docs]'
    # (pymupdf + python-docx). Gracefully disabled if deps not installed.
    security_document_scan_enabled: bool = True

    # -------------------------------------------------------------------------
    # Behavioral analytics (Phase 3.3)
    # -------------------------------------------------------------------------
    # Enable Markov sequence model + Isolation Forest anomaly detection.
    security_behavioral_enabled: bool = True
    # Minimum sessions before Isolation Forest starts scoring.
    security_isolation_forest_min_samples: int = 10
    # Expected fraction of anomalous sessions (IsolationForest contamination).
    security_isolation_forest_contamination: float = 0.05
    # Markov: P(to|from) below this threshold triggers an ALERT violation.
    security_markov_alert_threshold: float = 0.05

    # -------------------------------------------------------------------------
    # ML injection classifier (Phase 4)
    # -------------------------------------------------------------------------
    # Two operating modes (independently configurable):
    #
    # ASYNC mode (analysis_classifier_enabled):
    #   Runs in a background thread AFTER forwarding — zero latency on current call.
    #   Sets session.ml_injection_flag so the NEXT call in the session is blocked.
    #   Catches multi-turn injection chains. Always safe to enable.
    #
    # SYNC mode (analysis_classifier_sync_enabled):
    #   Runs INLINE before forwarding — adds ~50ms per LLM call (CPU inference).
    #   Blocks the CURRENT call if the classifier fires. Catches single-turn attacks.
    #   Recommended for production when latency budget allows.
    #   With sync enabled, TPR jumps from ~28% (structural-only) to ~85%+.
    #
    # Both modes can be active simultaneously — the async run is skipped when the
    # sync run already flagged the message (no double work).
    #
    # Requires: pip install 'aegivis-proxy[classifier]'  (transformers + torch)
    analysis_classifier_enabled: bool = False       # async mode (background, next-call)
    analysis_classifier_sync_enabled: bool = False  # sync mode (inline, current call)
    # HuggingFace model ID for the injection classifier.
    # protectai/deberta-v3-base-prompt-injection-v2: 88–99% PINT accuracy, ~280 MB.
    # Not gated — no HuggingFace account required.
    analysis_classifier_model: str = "protectai/deberta-v3-base-prompt-injection-v2"
    # Minimum injection probability to flag a session (set ml_injection_flag = True).
    # Scores >= threshold => "malicious" => ml_injection_flag set.
    # Scores >= threshold*0.6 => "suspicious" => ALERT violation only, flag NOT set.
    analysis_classifier_threshold: float = 0.85

    # ── Fast sync classifier (hot-path, PG-2-22M) ────────────────────────────
    # ── Ensemble second classifier ────────────────────────────────────────────
    # When enabled, a second model runs alongside the primary classifier.
    # The ensemble rule is max(primary_score, secondary_score) — if either
    # fires, the injection is flagged.  Reduces FN from different blind spots.
    # meta-llama/Prompt-Guard-86M: RoBERTa-based, ~86MB, ~30ms CPU.
    # Different architecture + training data → catches what DeBERTa misses.
    # Requires: pip install 'aegivis-proxy[classifier]' (same extra)
    analysis_classifier_second_enabled: bool = False
    analysis_classifier_second_model: str = "meta-llama/Prompt-Guard-86M"

    # -------------------------------------------------------------------------
    # Context-aware injection detection
    # -------------------------------------------------------------------------
    # Track injection scores across the last N turns of a session.
    # If the rolling average rises above the alert/block threshold, fire even
    # if no single turn exceeds the per-turn threshold.
    # This catches Crescendo / multi-turn gradual injection attacks.
    security_context_injection_window: int = 5           # number of turns to average
    security_context_injection_alert_avg: float = 0.35   # rolling avg to alert
    security_context_injection_block_avg: float = 0.60   # rolling avg to block

    # -------------------------------------------------------------------------
    # Tool output ML scan
    # -------------------------------------------------------------------------
    # When enabled, tool outputs also pass through the full ML classifier
    # pipeline (same as user messages) in addition to the pattern scanner.
    # Closes the indirect injection gap for web-fetched content, emails, etc.
    # Adds ~50ms per tool call when sync classifier is active.
    security_tool_output_ml_scan_enabled: bool = False

    # -------------------------------------------------------------------------
    # Rate limiting / token budget
    # -------------------------------------------------------------------------
    # Per-session token cap.  0 = disabled.
    # Blocks the next LLM call once cumulative token usage exceeds the cap.
    # Prevents runaway agents and data-exfiltration via many small calls.
    rate_limit_max_tokens_per_session: int = 0
    # Per-session LLM call cap.  0 = disabled (tool_call_loop_protection in
    # default.yaml handles extreme loops; this is a softer budget cap).
    rate_limit_max_llm_calls_per_session: int = 0

    # -------------------------------------------------------------------------
    # Capability Manifests (Phase 12)
    # -------------------------------------------------------------------------
    # HMAC-SHA256 signing key for capability manifests.
    # Must match AEGIVIS_MANIFEST_SIGNING_KEY on the backend.
    # If empty, signature verification is skipped (development mode).
    manifest_signing_key: str = ""
    # When true: BLOCK tool calls that violate the active manifest.
    # When false: ALERT only (observe mode — for rollout testing).
    manifest_enforce: bool = True
    # Cache TTL (seconds) for manifest lookups. Manifests are re-fetched
    # from the backend after this interval. Default: 5 minutes.
    manifest_cache_ttl_s: float = 300.0

    # -------------------------------------------------------------------------
    # Hard Egress Enforcement (Phase 13)
    # -------------------------------------------------------------------------
    # Deterministic allowlist-based network destination checking.
    # Replaces the probabilistic SSRF heuristic with a strict allowlist.
    # Mode "allowlist": BLOCK any tool call whose network destination arg
    #   (url, endpoint, webhook_url, host, etc.) is not in the allowlist.
    # Mode "audit": log violations only, never block (for learning period).
    # Mode "off": disabled entirely (fall back to SSRF heuristic only).
    #
    # Allowlist format (comma-separated):
    #   api.openai.com          — exact hostname
    #   *.anthropic.com         — wildcard subdomains
    #   db.internal:5432        — hostname with port
    #   10.0.0.0/8              — CIDR range
    #
    # Leave empty to run in audit mode (observes but does not block).
    # Example: AEGIVIS_SECURITY_EGRESS_ALLOWLIST=api.openai.com,*.anthropic.com
    security_egress_mode: str = "audit"   # "allowlist" | "audit" | "off"
    security_egress_allowlist: str = ""   # comma-separated allowed destinations

    # -------------------------------------------------------------------------
    # IFC Label System (Phase 11)
    # -------------------------------------------------------------------------
    # Tracks data provenance labels (TRUSTED / EXTERNAL) across the session.
    # When EXTERNAL-labeled content (from web search, email, fetch, etc.)
    # flows into a privileged sink argument (url, to, command, path, etc.),
    # the tool call is blocked. Deterministic — no ML, no heuristics.
    security_ifc_enabled: bool = True
    # When True: block the tool call on IFC violation. When False: alert only.
    security_ifc_block: bool = True
    # Comma-separated additional sink argument keys to treat as privileged.
    # Default set covers url/to/command/path/webhook etc. Add custom keys here.
    # Example: "target_email,output_path,sink_url"
    security_ifc_extra_sink_keys: str = ""

    # -------------------------------------------------------------------------
    # Data flow taint tracking (Phase 8)
    # -------------------------------------------------------------------------
    # Tags credentials the moment they enter agent context (system prompt or
    # tool result). Blocks tool calls that would send a tainted credential to
    # a network-sink tool (http_request, send_email, webhook, etc.).
    # Near-zero false positive rate — exact credential substring match.
    security_taint_tracking_enabled: bool = True

    # -------------------------------------------------------------------------
    # Multi-agent spawn chain enforcement (Phase 6)
    # -------------------------------------------------------------------------
    # Maximum allowed agent delegation depth.  0 = disabled.
    # Root agent = depth 0.  Agent spawned by root = depth 1.  etc.
    # Set X-Aegivis-Parent-Agent-Id and X-Aegivis-Parent-Session-Id headers when
    # spawning sub-agents so the proxy can track the chain.
    max_spawn_depth: int = 10

    # -------------------------------------------------------------------------
    # Prometheus metrics (Phase 3.4)
    # -------------------------------------------------------------------------
    # Expose /metrics endpoint for Prometheus scraping.
    metrics_enabled: bool = True

    # -------------------------------------------------------------------------
    # OpenTelemetry distributed tracing (Phase 3.4)
    # -------------------------------------------------------------------------
    # Enable OTLP/gRPC trace export (requires opentelemetry packages).
    otel_enabled: bool = False
    # OTLP gRPC endpoint URL (Jaeger, Tempo, etc.)
    otel_endpoint: str = "http://localhost:4317"
    # Service name shown in trace UIs.
    otel_service_name: str = "aegivis-proxy"

    # -------------------------------------------------------------------------
    # Redis (optional — enables distributed session state + stream pipeline)
    # -------------------------------------------------------------------------
    # Set to a Redis URL to enable:
    #   - RedisSessionTracker: session state shared across multiple proxy instances
    #   - RedisTransport: events published to Redis Streams (consumed by backend)
    # Empty = use in-memory session state + direct HTTP transport (default).
    # Example: "redis://redis:6379/0"
    redis_url: str = ""

    # -------------------------------------------------------------------------
    # Tool baseline enforcement
    # -------------------------------------------------------------------------
    # When enabled the proxy:
    #   1. Hashes the tools[] array on the first call of each session and
    #      blocks any mid-session change (tool-set-mutation violation).
    #   2. Reports newly observed tools to the backend so operators can
    #      review them in the Baselines dashboard.
    #   3. At TOOL_CALL_START, blocks any tool whose name is not in the
    #      operator-approved set for this agent (once a baseline exists).
    #
    # Agents with no approved baseline remain in *audit mode*: tools are
    # observed and reported but never blocked on scope grounds.
    tool_baseline_enabled: bool = True
    # How long (seconds) to cache the approved tool list per agent before
    # re-fetching from the backend.  5 minutes is a good default.
    tool_baseline_cache_ttl_s: float = 300.0

    # -------------------------------------------------------------------------
    # Human-in-the-Loop (HITL) approvals (Phase 9)
    # -------------------------------------------------------------------------
    # When enabled, certain tool calls are paused and require a human reviewer
    # to approve or deny before the agent may continue.
    hitl_enabled: bool = False
    # Comma-separated tool names that always require approval (exact match).
    # Example: "bash,exec,shell,http_request"
    hitl_required_tools: str = ""
    # If enforcement score >= this threshold, the tool call requires approval.
    hitl_score_threshold: float = 0.95
    # Max seconds the proxy will wait for a decision before treating as denied.
    hitl_timeout_seconds: int = 300
    # Require approval when a network-sink tool receives tainted credentials.
    hitl_on_network_sink: bool = True

    # -------------------------------------------------------------------------
    # Trust propagation (Phase 9)
    # -------------------------------------------------------------------------
    # When a child session's parent was compromised, automatically lower the
    # child's ML classifier threshold so it is more sensitive to injection.
    trust_propagation_enabled: bool = True
    # Fraction to subtract from the classifier threshold for untrusted children.
    # e.g. threshold=0.85, discount=0.20 → effective threshold=0.68
    trust_propagation_discount: float = 0.20

    # -------------------------------------------------------------------------
    # Request body size limit
    # -------------------------------------------------------------------------
    # Reject requests whose body exceeds this size before any processing.
    # Prevents memory exhaustion from very large payloads.
    # Default: 10 MB. Set 0 to disable.
    max_request_body_bytes: int = 10 * 1024 * 1024  # 10 MB

    # -------------------------------------------------------------------------
    # Reasoning trace (extended thinking capture)
    # -------------------------------------------------------------------------
    # Max characters to store per thinking block. Truncates before DB write.
    reasoning_trace_max_chars: int = 4000

    # Logging
    log_level: str = "INFO"
    log_request_bodies: bool = False  # set True only in dev, never in prod

    @property
    def hitl_required_tools_set(self) -> frozenset:
        return frozenset(t.strip() for t in self.hitl_required_tools.split(",") if t.strip())


settings = ProxySettings()
