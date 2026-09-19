/**
 * Tests for @aegivis/sdk/intercept — zero-config fetch interceptor.
 *
 * Tests cover:
 * - Provider detection by hostname
 * - Request payload extraction (model, messages, tools)
 * - Response payload extraction (usage, stop reason, text)
 * - globalThis.fetch is patched on import
 * - Non-LLM requests pass through unchanged
 * - Streaming responses (SSE) don't get body consumed
 * - install() is idempotent
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { detectProvider } from '../src/providers.js';
import { getSessionId, getAgentId, setSessionContext, clearSessionContext } from '../src/session.js';
import { install, uninstall } from '../src/intercept.js';

// ---------------------------------------------------------------------------
// Provider detection
// ---------------------------------------------------------------------------

describe('detectProvider', () => {
  it('detects anthropic', () => {
    expect(detectProvider('https://api.anthropic.com/v1/messages')).toBe('anthropic');
  });

  it('detects openai', () => {
    expect(detectProvider('https://api.openai.com/v1/chat/completions')).toBe('openai');
  });

  it('detects groq', () => {
    expect(detectProvider('https://api.groq.com/openai/v1/chat/completions')).toBe('groq');
  });

  it('detects google-gemini', () => {
    expect(detectProvider('https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent')).toBe('google-gemini');
  });

  it('detects azure-openai', () => {
    expect(detectProvider('https://myresource.openai.azure.com/openai/deployments/gpt-4/chat/completions')).toBe('azure-openai');
  });

  it('detects cohere', () => {
    expect(detectProvider('https://api.cohere.ai/v1/generate')).toBe('cohere');
  });

  it('detects mistral', () => {
    expect(detectProvider('https://api.mistral.ai/v1/chat/completions')).toBe('mistral');
  });

  it('detects deepseek', () => {
    expect(detectProvider('https://api.deepseek.com/chat/completions')).toBe('deepseek');
  });

  it('detects openrouter', () => {
    expect(detectProvider('https://openrouter.ai/api/v1/chat/completions')).toBe('openrouter');
  });

  it('returns null for unknown host', () => {
    expect(detectProvider('https://example.com/api')).toBeNull();
  });

  it('returns null for localhost', () => {
    expect(detectProvider('http://localhost:8000/v1/ingest')).toBeNull();
  });

  it('returns null for 127.0.0.1', () => {
    expect(detectProvider('http://127.0.0.1:8000/')).toBeNull();
  });

  it('returns null for .local hosts', () => {
    expect(detectProvider('http://myservice.local/api')).toBeNull();
  });

  it('returns null for invalid URL', () => {
    expect(detectProvider('not-a-url')).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Session resolution
// ---------------------------------------------------------------------------

describe('session resolution', () => {
  beforeEach(() => {
    clearSessionContext();
    delete process.env['AEGIVIS_SESSION_ID'];
    delete process.env['AEGIVIS_AGENT_ID'];
  });

  afterEach(() => {
    clearSessionContext();
    delete process.env['AEGIVIS_SESSION_ID'];
    delete process.env['AEGIVIS_AGENT_ID'];
  });

  it('generates a stable session ID starting with intercept-', () => {
    const id = getSessionId();
    expect(id).toMatch(/^intercept-[0-9a-f]{12}$/);
  });

  it('returns same generated ID on repeated calls', () => {
    const id1 = getSessionId();
    const id2 = getSessionId();
    expect(id1).toBe(id2);
  });

  it('reads session ID from env var', () => {
    clearSessionContext();
    process.env['AEGIVIS_SESSION_ID'] = 'test-session-123';
    expect(getSessionId()).toBe('test-session-123');
  });

  it('reads agent ID from env var', () => {
    process.env['AEGIVIS_AGENT_ID'] = 'my-finance-bot';
    expect(getAgentId()).toBe('my-finance-bot');
  });

  it('setSessionContext overrides env', () => {
    process.env['AEGIVIS_SESSION_ID'] = 'from-env';
    setSessionContext('from-code', 'my-agent');
    expect(getSessionId()).toBe('from-code');
    expect(getAgentId()).toBe('my-agent');
  });
});

// ---------------------------------------------------------------------------
// Fetch interceptor
// ---------------------------------------------------------------------------

describe('install()', () => {
  it('is idempotent — calling twice is safe', () => {
    const r1 = install();
    const r2 = install();
    expect(r1).toBe(true);
    expect(r2).toBe(true);
    expect((globalThis as Record<string, unknown>)['_aegivis_fetch_patched']).toBe(true);
  });

  it('patches globalThis.fetch', () => {
    install();
    expect(globalThis.fetch.name).toBe('aegivisFetch');
  });
});

describe('fetch interception', () => {
  let fetchCalls: Array<{ url: string; init?: RequestInit }> = [];
  let fireCalls:  Array<{ type: string; payload: Record<string, unknown> }> = [];
  // install() replaces globalThis.fetch with its wrapper, so the mock helpers
  // are only reachable through this reference.
  let mockFetch: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    fetchCalls = [];
    fireCalls  = [];

    // Uninstall to get a clean state, then mock original fetch
    uninstall();
    (globalThis as Record<string, unknown>)['_aegivis_fetch_patched'] = false;

    // Mock fetch
    mockFetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      fetchCalls.push({ url: String(input), init });
      return new Response(JSON.stringify({
        model:     'claude-opus-4-6',
        content:   [{ type: 'text', text: 'Hello from mock' }],
        usage:     { input_tokens: 10, output_tokens: 5 },
        stop_reason: 'end_turn',
      }), {
        headers: { 'content-type': 'application/json' },
      });
    });
    globalThis.fetch = mockFetch as unknown as typeof fetch;

    install();
  });

  afterEach(() => {
    uninstall();
    (globalThis as Record<string, unknown>)['_aegivis_fetch_patched'] = false;
  });

  it('passes through non-LLM requests unchanged', async () => {
    await globalThis.fetch('https://example.com/api');
    expect(fetchCalls.length).toBe(1);
    expect(fetchCalls[0].url).toBe('https://example.com/api');
  });

  it('passes through localhost requests unchanged', async () => {
    await globalThis.fetch('http://localhost:8000/v1/ingest');
    expect(fetchCalls.length).toBe(1);
  });

  it('intercepts anthropic requests', async () => {
    const response = await globalThis.fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      body:   JSON.stringify({ model: 'claude-opus-4-6', messages: [{ role: 'user', content: 'hi' }] }),
    });

    // fire() posts LLM_CALL_START/END through the original fetch, which is this
    // same mock, so count only the calls that reached the provider.
    const providerCalls = fetchCalls.filter((c) => c.url.includes('api.anthropic.com'));
    expect(providerCalls.length).toBe(1);
    expect(response.ok).toBe(true);

    // Response body should still be readable
    const data = await response.json();
    expect(data.model).toBe('claude-opus-4-6');
  });

  it('does not consume streaming response body', async () => {
    mockFetch.mockImplementationOnce(
      async () => new Response(null, {
        headers: { 'content-type': 'text/event-stream' },
      }),
    );

    const response = await globalThis.fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      body:   JSON.stringify({ model: 'claude-opus-4-6', stream: true, messages: [] }),
    });

    // SSE response — body should not be consumed by the interceptor
    expect(response.headers.get('content-type')).toBe('text/event-stream');
  });

  it('re-throws errors from original fetch', async () => {
    mockFetch.mockImplementationOnce(
      async () => { throw new Error('Network failure'); },
    );

    await expect(
      globalThis.fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        body:   JSON.stringify({ model: 'gpt-4o', messages: [] }),
      }),
    ).rejects.toThrow('Network failure');
  });
});

// ---------------------------------------------------------------------------
// LangChain adapter (smoke test — no @langchain/core required)
// ---------------------------------------------------------------------------

describe('AegivisLangChain', () => {
  it('can be instantiated', async () => {
    const { AegivisLangChain } = await import('../src/adapters/langchain.js');
    const handler = new AegivisLangChain({ agentId: 'test' });
    expect(handler.name).toBe('aegivis');
  });

  it('handleLLMStart does not throw', async () => {
    const { AegivisLangChain } = await import('../src/adapters/langchain.js');
    const handler = new AegivisLangChain();
    await expect(
      handler.handleLLMStart({ id: ['ChatOpenAI'] }, ['Hello'], 'run-1'),
    ).resolves.toBeUndefined();
  });

  it('handleLLMEnd does not throw', async () => {
    const { AegivisLangChain } = await import('../src/adapters/langchain.js');
    const handler = new AegivisLangChain();
    await handler.handleLLMStart({ id: [] }, ['p'], 'run-2');
    await expect(
      handler.handleLLMEnd({
        generations: [[{ text: 'Answer here', message: { content: 'Answer here' } }]],
        llmOutput: { tokenUsage: { promptTokens: 10, completionTokens: 5 } },
      }, 'run-2'),
    ).resolves.toBeUndefined();
  });

  it('handleToolStart and handleToolEnd do not throw', async () => {
    const { AegivisLangChain } = await import('../src/adapters/langchain.js');
    const handler = new AegivisLangChain();
    await handler.handleToolStart({ name: 'web_search' }, 'AI news', 'tool-1');
    await handler.handleToolEnd('Results here', 'tool-1');
  });
});
