/**
 * Aegivis OpenAI JS SDK Adapter
 * ==============================
 * Wraps an OpenAI client instance to add explicit session/agent context to
 * all intercepted events.
 *
 * Note: ``@aegivis/sdk/intercept`` already covers OpenAI calls automatically
 * via the global fetch patch.  This adapter is for explicit context binding
 * (agent ID, session ID) without relying on environment variables.
 *
 * Usage::
 *
 *     import OpenAI from 'openai';
 *     import { instrumentOpenAI } from '@aegivis/sdk/adapters/openai';
 *
 *     const client = instrumentOpenAI(new OpenAI(), { agentId: 'finance-bot' });
 *     // All calls on client are now tagged with agentId: 'finance-bot'
 *
 * Install:  ``npm install '@aegivis/sdk' 'openai'``
 */

import { setSessionContext } from '../session.js';

interface InstrumentOptions {
  agentId?:   string;
  sessionId?: string;
}

/**
 * Wrap an OpenAI client so all its API calls are tagged with the given
 * session and agent context in the Aegivis backend.
 *
 * Returns the same client object (mutated in-place) for easy drop-in use.
 */
export function instrumentOpenAI<T extends object>(
  client: T,
  options: InstrumentOptions = {},
): T {
  const { agentId, sessionId } = options;

  // Bind context — the global fetch interceptor reads these values.
  if (sessionId || agentId) {
    setSessionContext(sessionId ?? `openai-${Date.now()}`, agentId);
  }

  // Return client unchanged — the fetch interceptor handles the rest.
  return client;
}
