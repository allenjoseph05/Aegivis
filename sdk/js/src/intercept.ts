/**
 * Aegivis Zero-Config Interceptor — JavaScript / TypeScript
 * ==========================================================
 * Patches ``globalThis.fetch`` so every outbound LLM API call from every
 * JavaScript/TypeScript AI framework is captured automatically — no proxy URL
 * required and no SDK-specific configuration needed.
 *
 * Import once in your agent entry point::
 *
 *     import '@aegivis/sdk/intercept';      // ESM
 *     require('@aegivis/sdk/intercept');     // CJS
 *
 * Covered automatically:
 *     Anthropic JS SDK, OpenAI JS SDK, LangChain.js, LangGraph.js,
 *     Vercel AI SDK, Google Generative AI SDK, Cohere JS SDK,
 *     Mistral JS SDK — anything that calls globalThis.fetch internally.
 *
 * Supported providers (auto-detected by hostname):
 *     anthropic · openai · azure-openai · google-gemini · google-vertex
 *     cohere · mistral · groq · together · perplexity · deepseek · xai-grok
 *     aws-bedrock · fireworks · sambanova · cerebras · openrouter · nvidia
 *
 * Environment variables::
 *
 *     AEGIVIS_BACKEND_URL   Where to ship events (default: http://localhost:8000)
 *     AEGIVIS_API_KEY       API key (default: dev-dashboard-key)
 *     AEGIVIS_AGENT_ID      Agent label attached to events
 *     AEGIVIS_INTERCEPT     Set "false" to disable without removing the import
 *     AEGIVIS_DEBUG         Set "1" to log intercept activity to console
 */

import { detectProvider } from './providers.js';
import { getSessionId, getAgentId } from './session.js';
import { fire, initTransport } from './transport.js';
import { isEnabled, getBackendUrl, isDebug } from './config.js';

// ---------------------------------------------------------------------------
// Request body extraction
// ---------------------------------------------------------------------------

function readRequestBody(body: BodyInit | null | undefined): Record<string, unknown> {
  if (body == null) return {};

  let text: string;
  if (typeof body === 'string') {
    text = body;
  } else if (body instanceof Uint8Array || body instanceof ArrayBuffer) {
    text = new TextDecoder().decode(body);
  } else {
    // FormData, ReadableStream, Blob — not readable synchronously.
    return {};
  }

  try {
    const data = JSON.parse(text) as Record<string, unknown>;
    const p: Record<string, unknown> = {};

    if (data['model']) p['model'] = data['model'];

    const messages = (data['messages'] as unknown[]) ?? [];
    if (messages.length) {
      p['message_count'] = messages.length;

      // Last user message preview
      for (let i = messages.length - 1; i >= 0; i--) {
        const m = messages[i] as Record<string, unknown>;
        if (m['role'] === 'user') {
          const content = m['content'];
          const preview =
            typeof content === 'string'
              ? content
              : Array.isArray(content)
                ? (content as Array<Record<string, unknown>>)
                    .filter(b => b['type'] === 'text')
                    .map(b => String(b['text'] ?? ''))
                    .join(' ')
                : '';
          p['user_message_preview'] = preview.slice(0, 500);
          break;
        }
      }

      // System message (OpenAI style)
      const sysMsg = messages.find(
        m => (m as Record<string, unknown>)['role'] === 'system',
      ) as Record<string, unknown> | undefined;
      if (sysMsg) {
        p['system_prompt_preview'] = String(sysMsg['content'] ?? '').slice(0, 300);
      }
    }

    // Anthropic top-level system field
    if (data['system']) {
      p['system_prompt_preview'] = String(data['system']).slice(0, 300);
    }

    const tools =
      (data['tools'] as unknown[]) ??
      (data['functions'] as unknown[]) ??
      [];
    if (tools.length) {
      p['tool_count'] = tools.length;
      p['tool_names'] = tools.slice(0, 10).map(t => {
        const tool = t as Record<string, unknown>;
        return tool['name'] ?? (tool['function'] as Record<string, unknown>)?.['name'] ?? '?';
      });
    }

    for (const k of ['max_tokens', 'max_completion_tokens', 'temperature', 'stream']) {
      if (k in data) p[k] = data[k];
    }

    return p;
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Response body extraction — clones to avoid consuming the stream
// ---------------------------------------------------------------------------

async function readResponseBody(response: Response): Promise<Record<string, unknown>> {
  const contentType = response.headers.get('content-type') ?? '';
  if (contentType.includes('text/event-stream')) {
    return { streamed: true };
  }

  try {
    const data = (await response.clone().json()) as Record<string, unknown>;
    const p: Record<string, unknown> = {};

    if (data['model']) p['model'] = data['model'];

    const usage = (data['usage'] as Record<string, unknown>) ?? {};
    const inputTok = (usage['input_tokens'] ?? usage['prompt_tokens']) as number | undefined;
    const outputTok = (usage['output_tokens'] ?? usage['completion_tokens']) as number | undefined;
    if (inputTok  != null) p['input_tokens']  = inputTok;
    if (outputTok != null) p['output_tokens'] = outputTok;

    const stopReason =
      data['stop_reason'] ??
      (data['choices'] as Array<Record<string, unknown>>)?.[0]?.['finish_reason'];
    if (stopReason) p['stop_reason'] = String(stopReason);

    // Response text — Anthropic content blocks or OpenAI choices
    let text = '';
    for (const block of (data['content'] as Array<Record<string, unknown>>) ?? []) {
      if (block['type'] === 'text') {
        text = String(block['text'] ?? '');
        break;
      }
    }
    if (!text) {
      const choices = (data['choices'] as Array<Record<string, unknown>>) ?? [];
      text = String((choices[0]?.['message'] as Record<string, unknown>)?.['content'] ?? '');
    }
    if (text) p['response_preview'] = text.slice(0, 500);

    return p;
  } catch {
    return {};
  }
}

// ---------------------------------------------------------------------------
// Install / uninstall
// ---------------------------------------------------------------------------

/**
 * Patch globalThis.fetch. Idempotent — safe to call multiple times.
 * Returns true if patched, false if fetch is unavailable or disabled.
 */
export function install(): boolean {
  if (!isEnabled()) return false;

  const g = globalThis as Record<string, unknown>;
  if (g['_aegivis_fetch_patched']) return true;
  if (typeof globalThis.fetch !== 'function') return false;

  // Save original BEFORE patching — transport.ts uses this to post events
  // without re-entering our interceptor.
  const originalFetch = globalThis.fetch.bind(globalThis);
  initTransport(originalFetch);

  // Compute skip-hosts set (localhost + Aegivis backend host).
  const skipHosts = new Set(['localhost', '127.0.0.1', '::1']);
  try {
    const backendHost = new URL(getBackendUrl()).hostname;
    if (backendHost) skipHosts.add(backendHost);
  } catch { /* ignore bad URL */ }

  globalThis.fetch = async function aegivisFetch(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> {
    const urlStr = input instanceof Request ? input.url : String(input);

    // Fast host check before URL parsing.
    let host: string;
    try {
      host = new URL(urlStr).hostname.toLowerCase();
    } catch {
      return originalFetch(input, init);
    }
    if (skipHosts.has(host) || host.endsWith('.local')) {
      return originalFetch(input, init);
    }

    const provider = detectProvider(urlStr);
    if (!provider) return originalFetch(input, init);

    const sessionId = getSessionId();
    const agentId   = getAgentId();
    const t0        = Date.now();

    // Read body — works for string/Buffer bodies (all standard LLM SDK calls).
    const body = init?.body ?? (input instanceof Request ? null : null);
    const reqPayload = readRequestBody(body as BodyInit | null);
    reqPayload['provider'] = provider;

    if (isDebug()) {
      console.debug(`[aegivis] intercept ${provider} ${urlStr.slice(0, 80)} session=${sessionId}`);
    }

    fire('LLM_CALL_START', reqPayload, sessionId, agentId);

    let response: Response;
    try {
      response = await originalFetch(input, init);
    } catch (err) {
      fire(
        'LLM_CALL_ERROR',
        { provider, error: String(err).slice(0, 300), model: reqPayload['model'] ?? '' },
        sessionId,
        agentId,
      );
      throw err;
    }

    const latency_ms  = Date.now() - t0;
    const respPayload = await readResponseBody(response);
    respPayload['provider']    = provider;
    respPayload['latency_ms']  = latency_ms;
    respPayload['status_code'] = response.status;

    fire('LLM_CALL_END', { ...reqPayload, ...respPayload }, sessionId, agentId);
    return response;
  };

  g['_aegivis_fetch_patched'] = true;

  if (isDebug()) console.debug('[aegivis] globalThis.fetch patched');

  return true;
}

/**
 * Remove the fetch patch.  The original fetch reference is lost after patching,
 * so the module-level originalFetch closure is still used internally but the
 * public-facing globalThis.fetch is reset to a pass-through after this call.
 */
export function uninstall(): void {
  (globalThis as Record<string, unknown>)['_aegivis_fetch_patched'] = false;
}

// Auto-install on import.
install();
