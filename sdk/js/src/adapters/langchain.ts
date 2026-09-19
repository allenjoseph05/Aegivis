/**
 * Aegivis LangChain.js Adapter
 * ============================
 * A callback handler that captures LLM calls, tool executions, and chain/agent
 * events from LangChain.js and LangGraph.js.
 *
 * Usage (LangChain.js)::
 *
 *     import { AegivisLangChain } from '@aegivis/sdk/adapters/langchain';
 *
 *     const chain = RunnableSequence.from([...]).withConfig({
 *       callbacks: [new AegivisLangChain({ agentId: 'my-agent' })],
 *     });
 *
 * Usage (global callbacks)::
 *
 *     import { CallbackManager } from '@langchain/core/callbacks/manager';
 *     CallbackManager.configure(undefined, [new AegivisLangChain()]);
 *
 * Install:  ``npm install '@aegivis/sdk' '@langchain/core'``
 *
 * Note: This adapter records LangChain-level tool *invocations* as seen by the
 * framework.  For actual tool execution events, use ``instrument()`` from
 * ``@aegivis/sdk`` to wrap individual tool functions.
 */

import { fire } from '../transport.js';
import { getSessionId, getAgentId } from '../session.js';

interface AegivisLangChainOptions {
  agentId?:   string;
  sessionId?: string;
}

/**
 * LangChain.js callback handler.
 *
 * Implements the LangChain BaseCallbackHandler interface via duck-typing so
 * ``@langchain/core`` is a peer dependency and not required at import time.
 */
export class AegivisLangChain {
  readonly name = 'aegivis';

  private readonly _sessionId: string;
  private readonly _agentId:   string;
  private readonly _timings    = new Map<string, number>();

  constructor(options: AegivisLangChainOptions = {}) {
    this._sessionId = options.sessionId ?? getSessionId();
    this._agentId   = options.agentId   ?? getAgentId();
  }

  // ── LLM events ────────────────────────────────────────────────────────────

  async handleLLMStart(
    llm: Record<string, unknown>,
    prompts: string[],
    runId: string,
    _parentRunId?: string,
    extraParams?: Record<string, unknown>,
  ): Promise<void> {
    this._timings.set(runId, Date.now());
    fire(
      'LLM_CALL_START',
      {
        provider:             'langchain',
        model:                String(extraParams?.['invocation_params']?.['model_name'] ?? llm['id']?.[llm['id']?.length - 1] ?? ''),
        user_message_preview: prompts[0]?.slice(0, 500) ?? '',
        message_count:        prompts.length,
        source:               'langchain-callback',
      },
      this._sessionId,
      this._agentId,
    );
  }

  async handleLLMEnd(
    output: Record<string, unknown>,
    runId: string,
  ): Promise<void> {
    const latency_ms = Date.now() - (this._timings.get(runId) ?? Date.now());
    this._timings.delete(runId);

    const generations = (output['generations'] as Array<Array<Record<string, unknown>>>) ?? [];
    const firstGen    = generations[0]?.[0];
    const text        = String(firstGen?.['text'] ?? firstGen?.['message']?.['content'] ?? '');

    const llmOutput = (output['llmOutput'] as Record<string, unknown>) ?? {};
    const usage     = (llmOutput['tokenUsage'] as Record<string, unknown>) ?? {};

    fire(
      'LLM_CALL_END',
      {
        provider:         'langchain',
        response_preview: text.slice(0, 500),
        input_tokens:     usage['promptTokens'] as number | undefined,
        output_tokens:    usage['completionTokens'] as number | undefined,
        latency_ms,
        source:           'langchain-callback',
      },
      this._sessionId,
      this._agentId,
    );
  }

  async handleLLMError(err: unknown, runId: string): Promise<void> {
    this._timings.delete(runId);
    fire(
      'LLM_CALL_ERROR',
      { provider: 'langchain', error: String(err).slice(0, 300), source: 'langchain-callback' },
      this._sessionId,
      this._agentId,
    );
  }

  // ── Tool events ────────────────────────────────────────────────────────────

  async handleToolStart(
    tool: Record<string, unknown>,
    input: string,
    runId: string,
  ): Promise<void> {
    this._timings.set(`tool:${runId}`, Date.now());
    fire(
      'TOOL_CALL_START',
      {
        tool_name:   String(tool['name'] ?? ''),
        tool_input:  input.slice(0, 500),
        source:      'langchain-callback',
      },
      this._sessionId,
      this._agentId,
    );
  }

  async handleToolEnd(output: string, runId: string): Promise<void> {
    const latency_ms = Date.now() - (this._timings.get(`tool:${runId}`) ?? Date.now());
    this._timings.delete(`tool:${runId}`);
    fire(
      'TOOL_CALL_END',
      {
        tool_output: output.slice(0, 500),
        latency_ms,
        source:      'langchain-callback',
      },
      this._sessionId,
      this._agentId,
    );
  }

  async handleToolError(err: unknown, runId: string): Promise<void> {
    this._timings.delete(`tool:${runId}`);
    fire(
      'TOOL_CALL_ERROR',
      { error: String(err).slice(0, 300), source: 'langchain-callback' },
      this._sessionId,
      this._agentId,
    );
  }

  // ── Agent events ───────────────────────────────────────────────────────────

  async handleAgentAction(
    action: Record<string, unknown>,
  ): Promise<void> {
    fire(
      'AGENT_THOUGHT',
      {
        tool:    String(action['tool'] ?? ''),
        input:   String(action['toolInput'] ?? '').slice(0, 500),
        log:     String(action['log'] ?? '').slice(0, 300),
        source:  'langchain-callback',
      },
      this._sessionId,
      this._agentId,
    );
  }

  async handleAgentEnd(
    finish: Record<string, unknown>,
  ): Promise<void> {
    fire(
      'AGENT_THOUGHT',
      {
        finish_output: String(
          (finish['returnValues'] as Record<string, unknown>)?.['output'] ?? '',
        ).slice(0, 500),
        log:    String(finish['log'] ?? '').slice(0, 300),
        source: 'langchain-callback',
      },
      this._sessionId,
      this._agentId,
    );
  }
}
