/**
 * Aegivis Vercel AI SDK Adapter
 * ==============================
 * Middleware for the Vercel AI SDK (``ai`` package) that tags all LLM calls
 * with Aegivis session and agent context.
 *
 * Note: ``@aegivis/sdk/intercept`` already covers Vercel AI SDK calls via the
 * global fetch patch.  This adapter provides explicit session/agent binding
 * and adds chain-level metadata (step counts, finish reason).
 *
 * Usage::
 *
 *     import { streamText } from 'ai';
 *     import { aegivisMiddleware } from '@aegivis/sdk/adapters/vercel-ai';
 *
 *     const result = await streamText({
 *       model: openai('gpt-4o'),
 *       prompt: '...',
 *       experimental_telemetry: {
 *         isEnabled: true,
 *         functionId: 'my-agent',
 *       },
 *       experimental_providerMetadata: {
 *         aegivis: { agentId: 'my-agent' },
 *       },
 *     });
 *
 * Or as a language model middleware::
 *
 *     import { wrapLanguageModel } from 'ai';
 *     const wrapped = wrapLanguageModel({
 *       model: openai('gpt-4o'),
 *       middleware: aegivisMiddleware({ agentId: 'my-agent' }),
 *     });
 *
 * Install:  ``npm install '@aegivis/sdk' 'ai'``
 */

import { fire } from '../transport.js';
import { getSessionId, getAgentId, setSessionContext } from '../session.js';

interface AegivisMiddlewareOptions {
  agentId?:   string;
  sessionId?: string;
}

/**
 * Vercel AI SDK language model middleware.
 * Wraps ``doGenerate`` and ``doStream`` to emit LLM_CALL_START/END events.
 */
export function aegivisMiddleware(options: AegivisMiddlewareOptions = {}) {
  const sessionId = options.sessionId ?? getSessionId();
  const agentId   = options.agentId   ?? getAgentId();

  if (options.sessionId || options.agentId) {
    setSessionContext(sessionId, agentId);
  }

  return {
    wrapGenerate: async (opts: {
      doGenerate: () => Promise<Record<string, unknown>>;
      params:     Record<string, unknown>;
    }) => {
      const t0 = Date.now();
      const model = String(
        (opts.params['model'] as Record<string, unknown>)?.['modelId'] ?? '',
      );

      fire(
        'LLM_CALL_START',
        { provider: 'vercel-ai', model, source: 'vercel-ai-middleware' },
        sessionId,
        agentId,
      );

      let result: Record<string, unknown>;
      try {
        result = await opts.doGenerate();
      } catch (err) {
        fire(
          'LLM_CALL_ERROR',
          { provider: 'vercel-ai', model, error: String(err).slice(0, 300), source: 'vercel-ai-middleware' },
          sessionId,
          agentId,
        );
        throw err;
      }

      fire(
        'LLM_CALL_END',
        {
          provider:         'vercel-ai',
          model,
          latency_ms:       Date.now() - t0,
          finish_reason:    result['finishReason'],
          input_tokens:     (result['usage'] as Record<string, number> | undefined)?.['promptTokens'],
          output_tokens:    (result['usage'] as Record<string, number> | undefined)?.['completionTokens'],
          response_preview: String(
            (result['text'] ?? (result['toolCalls'] as unknown[] | undefined)?.[0] ?? ''),
          ).slice(0, 500),
          source:           'vercel-ai-middleware',
        },
        sessionId,
        agentId,
      );

      return result;
    },

    wrapStream: async (opts: {
      doStream: () => Promise<Record<string, unknown>>;
      params:   Record<string, unknown>;
    }) => {
      const t0 = Date.now();
      const model = String(
        (opts.params['model'] as Record<string, unknown>)?.['modelId'] ?? '',
      );

      fire(
        'LLM_CALL_START',
        { provider: 'vercel-ai', model, streamed: true, source: 'vercel-ai-middleware' },
        sessionId,
        agentId,
      );

      let result: Record<string, unknown>;
      try {
        result = await opts.doStream();
      } catch (err) {
        fire(
          'LLM_CALL_ERROR',
          { provider: 'vercel-ai', model, error: String(err).slice(0, 300), source: 'vercel-ai-middleware' },
          sessionId,
          agentId,
        );
        throw err;
      }

      fire(
        'LLM_CALL_END',
        { provider: 'vercel-ai', model, streamed: true, latency_ms: Date.now() - t0, source: 'vercel-ai-middleware' },
        sessionId,
        agentId,
      );

      return result;
    },
  };
}
