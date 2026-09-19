/**
 * Aegivis JavaScript / TypeScript SDK
 * =====================================
 * Universal AI agent observability and security for the JS ecosystem.
 *
 * Quick start — zero-config (covers all fetch-based frameworks)::
 *
 *     import '@aegivis/sdk/intercept';   // one import, all LLM calls captured
 *
 * With explicit session context::
 *
 *     import { session } from '@aegivis/sdk';
 *
 *     await session({ agentId: 'my-agent' }, async (s) => {
 *       s.annotate('Starting task');
 *       await runMyAgent();
 *     });
 *
 * Framework adapters::
 *
 *     // LangChain.js
 *     import { AegivisLangChain } from '@aegivis/sdk/adapters/langchain';
 *     chain.withConfig({ callbacks: [new AegivisLangChain({ agentId: 'bot' })] });
 *
 *     // OpenAI JS SDK
 *     import { instrumentOpenAI } from '@aegivis/sdk/adapters/openai';
 *     const client = instrumentOpenAI(new OpenAI(), { agentId: 'finance-bot' });
 *
 *     // Vercel AI SDK
 *     import { aegivisMiddleware } from '@aegivis/sdk/adapters/vercel-ai';
 *     const wrapped = wrapLanguageModel({ model, middleware: aegivisMiddleware() });
 */

export { install as installIntercept, uninstall as uninstallIntercept } from './intercept.js';
export { getSessionId, getAgentId, setSessionContext, clearSessionContext } from './session.js';
export { fire } from './transport.js';
export { detectProvider } from './providers.js';

// Re-export adapters for convenience (tree-shaken if unused).
export { AegivisLangChain } from './adapters/langchain.js';
export { instrumentOpenAI } from './adapters/openai.js';
export { aegivisMiddleware } from './adapters/vercel-ai.js';

// ---------------------------------------------------------------------------
// Session context helper
// ---------------------------------------------------------------------------

import { setSessionContext, clearSessionContext, getSessionId, getAgentId } from './session.js';
import { fire } from './transport.js';

interface SessionOptions {
  agentId?:   string;
  sessionId?: string;
}

interface SessionHandle {
  id:       string;
  agentId:  string;
  annotate: (note: string, metadata?: Record<string, unknown>) => void;
}

/**
 * Run a block with a specific session/agent context.
 * All intercepted LLM calls within the callback are attributed to this session.
 *
 *     await session({ agentId: 'finance-bot' }, async (s) => {
 *       s.annotate('Starting analysis');
 *       const result = await llm.invoke('...');
 *     });
 */
export async function session<T>(
  options: SessionOptions,
  fn: (handle: SessionHandle) => Promise<T>,
): Promise<T> {
  const sessionId = options.sessionId ?? `js-session-${Date.now().toString(36)}`;
  const agentId   = options.agentId   ?? getAgentId();

  setSessionContext(sessionId, agentId);

  const handle: SessionHandle = {
    id:      sessionId,
    agentId,
    annotate(note, metadata = {}) {
      fire(
        'ANNOTATION',
        { note, ...metadata, source: 'js-sdk-session' },
        sessionId,
        agentId,
      );
    },
  };

  try {
    return await fn(handle);
  } finally {
    clearSessionContext();
  }
}

export const version = '2.0.0';
