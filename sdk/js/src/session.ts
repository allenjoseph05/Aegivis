/**
 * Session and agent ID resolution.
 *
 * Priority:
 *   1. Explicit value set via setSessionContext()
 *   2. AEGIVIS_SESSION_ID / AEGIVIS_AGENT_ID env var (Node.js) or sessionStorage (browser)
 *   3. Auto-generated stable ID for the process lifetime
 */
import { getDefaultAgentId } from './config.js';

function getEnv(key: string): string | undefined {
  if (typeof process !== 'undefined' && typeof process.env === 'object') {
    return process.env[key] ?? undefined;
  }
  if (typeof window !== 'undefined' && window.sessionStorage) {
    return window.sessionStorage.getItem(key) ?? undefined;
  }
  return undefined;
}

function randomHex(len: number): string {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID().replace(/-/g, '').slice(0, len);
  }
  // Fallback
  return Math.random().toString(16).slice(2, 2 + len).padEnd(len, '0');
}

let _sessionId: string | null = null;
let _agentId: string | null = null;

export function getSessionId(): string {
  return _sessionId ?? getEnv('AEGIVIS_SESSION_ID') ?? (_sessionId = `intercept-${randomHex(12)}`);
}

export function getAgentId(): string {
  return _agentId ?? getEnv('AEGIVIS_AGENT_ID') ?? getDefaultAgentId();
}

/**
 * Set session context programmatically — use this inside a "session" wrapper
 * so that all intercepted calls in that scope share the same session/agent IDs.
 */
export function setSessionContext(sessionId: string, agentId?: string): void {
  _sessionId = sessionId;
  if (agentId) _agentId = agentId;
}

export function clearSessionContext(): void {
  _sessionId = null;
  _agentId = null;
}
