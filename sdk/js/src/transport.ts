/**
 * Fire-and-forget event shipping to the Aegivis backend.
 *
 * Uses the ORIGINAL fetch reference saved before any patching to avoid
 * recursion. Events are sent as POST /v1/ingest and errors are silently
 * swallowed — observability must never break agent execution.
 */
import { getBackendUrl, getApiKey } from './config.js';

// Saved before intercept.ts patches globalThis.fetch.
let _originalFetch: typeof fetch | null = null;

export function initTransport(originalFetch: typeof fetch): void {
  _originalFetch = originalFetch;
}

export function fire(
  eventType: string,
  payload: Record<string, unknown>,
  sessionId: string,
  agentId: string,
): void {
  const backendUrl = getBackendUrl();
  if (!backendUrl || !_originalFetch) return;

  const event = {
    event_type:          eventType,
    agent_id:            agentId,
    session_id:          sessionId,
    // JS Date.now() is millisecond-precision; convert to approximate nanoseconds.
    timestamp_ns:        Date.now() * 1_000_000,
    interception_layer:  'js-sdk-intercept',
    provider:            payload['provider'] ?? 'unknown',
    model:               payload['model'] ?? '',
    payload,
  };

  void _originalFetch(`${backendUrl}/v1/ingest`, {
    method:   'POST',
    headers:  { 'Content-Type': 'application/json', 'X-API-Key': getApiKey() },
    body:     JSON.stringify(event),
    // keepalive ensures delivery even if the page/worker is unloading.
    keepalive: true,
  }).catch(() => { /* fire-and-forget — never throw */ });
}
