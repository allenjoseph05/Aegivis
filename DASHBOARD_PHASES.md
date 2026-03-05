# AgentBlackBox Dashboard Redesign — Implementation Phases

> **Status**: Planning complete. Implement one phase at a time in order.
> **Principle**: Progressive disclosure — simple overview first, power-user features behind tabs/modals.

---

## Phase 0 — Navigation Restructure *(prerequisite for all others)*

**What**: Rename + reorder the nav so it matches how operators actually work.

| Before | After |
|--------|-------|
| Sessions | Events |
| Benchmark | *(moved to `/dev` route, removed from primary nav)* |
| Topology | *(removed from primary nav, link stays in Settings)* |
| *(new)* | Agents |
| *(new)* | Security Stack |
| *(new)* | Alerts |

**Primary nav order**: Overview → Agents → Events → Violations → Security Stack → Alerts → Settings
**Secondary (under Settings)**: Audit & Export, Policies, API Keys, Integrations

**Files**:
- `dashboard/src/main.tsx` — reorder routes + nav links, rename Sessions→Events

---

## Phase 1 — Overview Page (Agent Health + Incident Feed)

**Goal**: Operators land here and immediately see "are my agents safe right now?"

**Components to build**:
1. **Agent Health Cards** — one card per agent showing:
   - Name + status badge (Healthy / Alerting / Blocked / Offline)
   - Last seen timestamp
   - Today's violation count with severity breakdown
   - Enforcement mode (Audit / Active)
   - Quick link → Agent detail

2. **Live Incident Feed** — last 20 violations across all agents
   - Severity icon + rule name + agent + time ago
   - Click → Violations page filtered to that event

3. **Security Posture Score** (0–100) — weighted aggregate:
   - 40pts: no BLOCK events in last 24h
   - 30pts: tool baseline approved for all agents
   - 20pts: all key security layers enabled
   - 10pts: no pending tool reviews

4. **Quick Action Bar**:
   - "Review pending tools" (badge with count)
   - "Export audit report" → opens Export page

**API needed**: `/v1/agents` (existing), `/v1/violations?limit=20` (existing), `/v1/metrics/overview` (existing)
**New API needed**: None (use existing data)

**Files**:
- `dashboard/src/pages/Overview/index.tsx` *(new)*
- `dashboard/src/pages/Overview/AgentHealthCard.tsx` *(new)*
- `dashboard/src/pages/Overview/IncidentFeed.tsx` *(new)*
- `dashboard/src/pages/Overview/PostureScore.tsx` *(new)*
- `dashboard/src/api/client.ts` — no new API calls needed

---

## Phase 2 — Agent Detail Page (Tabbed)

**Goal**: Operators can configure each agent independently from one place.

**Tab structure**:
```
[Health]  [Security Profile]  [Tool Baseline]  [Rate Limits]  [Policies]
```

### Tab: Health
- Session timeline chart (LLM calls per hour, last 24h)
- Violation trend (7 days)
- Top violation rules table
- Last 5 sessions with status

### Tab: Security Profile
- Per-agent toggles for each security layer (enabled / alert-only / disabled)
- Layers: Injection Scanner | PII Masking | Canary Detection | Tool Baseline | Taint Tracking | System Prompt Guard | Spawn Chain Limit | Unicode Stego | Rate Limiting
- Each toggle shows: estimated FPR, latency impact, description
- "Save Profile" button → POST to agent config endpoint

### Tab: Tool Baseline
- Current status: Audit Mode / Enforcement Active
- Tool table: name | description | first seen | status (pending/approved/denied) | approve/deny buttons
- "Approve All" button
- "What is this?" info box (first-time explanation)

### Tab: Rate Limits
- Token limit per session (slider, 0 = disabled, recommendations shown)
- LLM call limit per session (slider)
- Live example: "At 10,000 tokens/session, a GPT-4 agent summarizing docs would hit this after ~40 pages"
- Recommendations shown based on agent's observed usage

### Tab: Policies
- List of active policy rules for this agent
- Toggle each rule on/off
- Threshold sliders for injection/credential score rules

**New API needed**:
- `GET /v1/agents/{agent_id}/config` — per-agent security profile settings
- `PUT /v1/agents/{agent_id}/config` — save profile

**Files**:
- `dashboard/src/pages/Agents/AgentDetail.tsx` *(new)*
- `dashboard/src/pages/Agents/tabs/HealthTab.tsx` *(new)*
- `dashboard/src/pages/Agents/tabs/SecurityProfileTab.tsx` *(new)*
- `dashboard/src/pages/Agents/tabs/ToolBaselineTab.tsx` *(new, extracts from Baselines page)*
- `dashboard/src/pages/Agents/tabs/RateLimitsTab.tsx` *(new)*
- `dashboard/src/pages/Agents/tabs/PoliciesTab.tsx` *(new)*
- `dashboard/src/pages/Agents/index.tsx` — updated to link to detail

---

## Phase 3 — Security Stack Page (Layer Explorer + Playground)

**Goal**: Let operators understand what each security layer does and tune their stack.

### Layer Explorer
Grid of layer cards, each showing:
- Layer name + icon
- Description (1–2 sentences)
- Estimated false-positive rate
- Hot-path latency impact
- Whether it requires ML model (badge: "Requires GPU/CPU model")
- Toggle: Enabled / Alert-only / Disabled (applies globally as default)

**Layers to document**:
| Layer | FPR | Latency | Notes |
|-------|-----|---------|-------|
| Structural Injection Scanner | ~1.2% | <1ms | Always on, no ML |
| Async ML Classifier (DeBERTa) | ~3% | 0ms hot-path | Blocks next call |
| Sync ML Classifier | ~3% | ~50ms | Blocks current call |
| PII Masking (Presidio) | ~2% | ~15ms | Requires spaCy model |
| Canary Detection | <0.1% | <1ms | Zero FP on real traffic |
| Tool Output Scanner | ~1% | <1ms | Structural only |
| System Prompt Guard | ~0.5% | <1ms | Hash-based |
| Taint Tracking | <0.1% | <1ms | Exact match |
| Tool Baseline | 0% | <1ms | Operator-approved list |
| Unicode Steganography | <0.1% | <1ms | |
| Spawn Chain Limiter | 0% | <1ms | Configurable depth |

### Stack Presets
Three preset buttons:
- **Minimal** (latency-sensitive): Structural + Canary + Tool Baseline only
- **Balanced** (recommended): All structural + Async ML + PII + Taint
- **Maximum** (high-security): Everything including Sync ML + second classifier

### Security Sandbox / Playground
- Text input: paste a prompt, tool call JSON, or tool output
- Run selected layers against it
- See per-layer scores, which would trigger alert/block
- Helps operators calibrate thresholds with their own data

**New API needed**:
- `POST /v1/playground/scan` — scan arbitrary text/tool-call through proxy security stack

**Files**:
- `dashboard/src/pages/SecurityStack/index.tsx` *(new)*
- `dashboard/src/pages/SecurityStack/LayerCard.tsx` *(new)*
- `dashboard/src/pages/SecurityStack/StackPresets.tsx` *(new)*
- `dashboard/src/pages/SecurityStack/Playground.tsx` *(new)*
- `proxy/app/api/playground.py` *(new — thin wrapper over enforcement + security modules)*
- `backend/app/api/v1/playground.py` *(or proxy-side endpoint)*

---

## Phase 4 — Alerts Page (Channel Setup + Routing)

**Goal**: Operators configure where violations go (Slack, email, PagerDuty, webhooks) and for which categories.

### Alert Channels
Four channel types, each as a collapsible card:
1. **Slack** — webhook URL + test button
2. **Email** — SMTP host/port/user/pass + "to" address + test button
3. **PagerDuty** — routing key + test button
4. **Generic Webhook** — URL + HMAC signing secret + test button

### Per-Category Routing
Table: for each violation category, choose which channels receive it

| Category | Slack | Email | PagerDuty | Webhook |
|----------|-------|-------|-----------|---------|
| BLOCK events | ✓ | ✓ | ✓ | ✓ |
| ALERT events | ✓ | — | — | ✓ |
| Tool not in baseline | ✓ | — | — | — |
| Data exfiltration | ✓ | ✓ | ✓ | ✓ |
| System prompt mutation | ✓ | ✓ | ✓ | — |
| New session started | — | — | — | — |
| Spawn depth exceeded | ✓ | — | ✓ | — |

### Anti-Fatigue Settings
- **Cooldown window**: suppress duplicate alerts within N seconds (default 600)
- **Minimum severity**: LOW / MEDIUM / HIGH / CRITICAL threshold
- **Quiet hours**: time range where only CRITICAL alerts fire

**New API needed**:
- `GET /v1/alert-config` — current channel + routing config
- `PUT /v1/alert-config` — save
- `POST /v1/alert-config/test/{channel}` — send test notification

**Files**:
- `dashboard/src/pages/Alerts/index.tsx` *(new)*
- `dashboard/src/pages/Alerts/ChannelCard.tsx` *(new)*
- `dashboard/src/pages/Alerts/RoutingTable.tsx` *(new)*
- `dashboard/src/pages/Alerts/AntiFatigueSettings.tsx` *(new)*
- `backend/app/api/v1/alert_config.py` *(new)*

---

## Phase 5 — Violations Page Enhancement

**Goal**: Make violations actionable, not just a log dump.

**Improvements**:
1. **Group-by selector**: by Agent | by Rule | by Severity | by Session
2. **FP Marking**: thumbs-down button → marks violation as false positive + optionally auto-tunes threshold
3. **Detail Drawer**: click violation → right-side drawer shows:
   - Full request body (collapsed by default)
   - Matched patterns / scores per layer
   - Session context (call number, prior violations this session)
   - "Block this session" button
   - "Export this event" button
4. **Threshold Recommendation**: if >30% of a rule's violations are marked FP → banner appears: "Rule X has high FP rate. Suggested threshold: 0.72 (current: 0.50)"
5. **Saved Filters**: save commonly-used filter combos with a name

**New API needed**:
- `POST /v1/violations/{id}/mark-fp` — mark as false positive
- `GET /v1/violations/{id}/context` — full event context for drawer

**Files**:
- `dashboard/src/pages/Violations/index.tsx` *(update existing)*
- `dashboard/src/pages/Violations/ViolationDrawer.tsx` *(new)*
- `dashboard/src/pages/Violations/GroupBySelector.tsx` *(new)*
- `backend/app/api/v1/violations.py` *(add mark-fp + context endpoints)*

---

## Phase 6 — Settings Page Consolidation

**Goal**: One place for all global configuration — no more hunting across pages.

**Sections**:
1. **Organisation** — name, API keys list (show/revoke)
2. **Default Security Profile** — global defaults for new agents (used if no per-agent override)
3. **Rate Limits (Global)** — token/call limits as fallback defaults
4. **Integrations** — SIEM export (Splunk/Elasticsearch), links to alert channel config
5. **Proxy Config** — backend URL, ML model settings, cache TTLs
6. **Danger Zone** — purge all sessions, clear violation history, reset baselines

**Files**:
- `dashboard/src/pages/Settings/index.tsx` *(new, replaces current sparse settings)*
- `dashboard/src/pages/Settings/OrgSection.tsx` *(new)*
- `dashboard/src/pages/Settings/SecurityDefaults.tsx` *(new)*
- `dashboard/src/pages/Settings/IntegrationsSection.tsx` *(new)*

---

## Phase 7 — Agent Onboarding Wizard

**Goal**: Make it easy for a new customer to register their first agent and configure it in 3 steps.

**Wizard steps**:
1. **Agent Identity** — name, description, expected LLM provider (OpenAI/Anthropic/etc.)
2. **Security Profile** — pick a preset (Minimal/Balanced/Maximum) or custom-configure layers
3. **Proxy Setup** — show the 3 lines of code to point their agent at the proxy + verify connection

**"Verify Connection" button**: polls `/v1/agents/{id}/sessions?limit=1` — shows green checkmark when first call arrives.

**Files**:
- `dashboard/src/pages/Agents/OnboardingWizard.tsx` *(new)*
- `dashboard/src/pages/Agents/OnboardingWizard/StepIdentity.tsx` *(new)*
- `dashboard/src/pages/Agents/OnboardingWizard/StepProfile.tsx` *(new)*
- `dashboard/src/pages/Agents/OnboardingWizard/StepSetup.tsx` *(new)*

---

## Phase 8 — Analytics + Trends

**Goal**: "Is my security posture improving over time?"

**Additions**:
1. **Posture Trend Chart** — 30-day line chart of security posture score
2. **Agent Comparison Table** — side-by-side: which agent has most violations? highest FPR?
3. **Config Change History** — log of "who changed what and when" (useful for compliance)
4. **Usage Cost Estimation** — based on observed tokens/session × agent count × avg sessions/day

**Files**:
- `dashboard/src/pages/Analytics/index.tsx` *(new)*
- `backend/app/api/v1/analytics.py` *(new — trend queries)*

---

## Implementation Order

```
Phase 0 (nav)  →  Phase 1 (overview)  →  Phase 2 (agent detail)
    →  Phase 3 (security stack)  →  Phase 4 (alerts)
    →  Phase 5 (violations)  →  Phase 6 (settings)
    →  Phase 7 (onboarding)  →  Phase 8 (analytics)
```

Phases 0–2 unblock the core operator workflow.
Phases 3–4 add configurability and alerting.
Phases 5–6 reduce noise and centralise config.
Phases 7–8 are growth / retention features.

---

## What to Remove / Simplify

| Current element | Action |
|-----------------|--------|
| Topology graph | Remove from primary nav; keep data available via API |
| Benchmark page | Move to `/dev/benchmark` (hidden route); not shown to customers |
| Raw session JSON dump | Replace with readable Event timeline |
| Compliance tab (standalone) | Merge into Audit & Export under Settings |
| Multiple "Reload" buttons | Replace with auto-refresh toggle (30s interval) |
