/**
 * Environment variable access — works in Node.js and browser.
 */

function getEnv(key: string): string | undefined {
  // Node.js
  if (typeof process !== 'undefined' && typeof process.env === 'object') {
    return process.env[key] ?? undefined;
  }
  // Browser — check sessionStorage as a fallback for config
  if (typeof window !== 'undefined' && window.sessionStorage) {
    return window.sessionStorage.getItem(key) ?? undefined;
  }
  return undefined;
}

export function getBackendUrl(): string {
  return (getEnv('AEGIVIS_BACKEND_URL') ?? 'http://localhost:8000').replace(/\/$/, '');
}

export function getApiKey(): string {
  return getEnv('AEGIVIS_API_KEY') ?? 'dev-dashboard-key';
}

export function getDefaultAgentId(): string {
  return getEnv('AEGIVIS_AGENT_ID') ?? 'intercepted-agent';
}

export function isEnabled(): boolean {
  const val = (getEnv('AEGIVIS_INTERCEPT') ?? 'true').toLowerCase();
  return val !== 'false' && val !== '0' && val !== 'no';
}

export function isDebug(): boolean {
  return getEnv('AEGIVIS_DEBUG') === '1';
}
