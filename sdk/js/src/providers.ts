/**
 * LLM provider detection by URL hostname.
 * Mirrors the Python interceptor's _PROVIDER_MAP exactly.
 */

const PROVIDER_MAP: [string, string][] = [
  ['api.anthropic.com',                 'anthropic'],
  ['api.openai.com',                    'openai'],
  ['.openai.azure.com',                 'azure-openai'],
  ['inference.ai.azure.com',            'azure-inference'],
  ['generativelanguage.googleapis.com', 'google-gemini'],
  ['aiplatform.googleapis.com',         'google-vertex'],
  ['api.cohere.ai',                     'cohere'],
  ['api.cohere.com',                    'cohere'],
  ['api.mistral.ai',                    'mistral'],
  ['api.groq.com',                      'groq'],
  ['api.together.xyz',                  'together'],
  ['api.perplexity.ai',                 'perplexity'],
  ['api.deepseek.com',                  'deepseek'],
  ['api.x.ai',                          'xai-grok'],
  ['bedrock-runtime.amazonaws.com',     'aws-bedrock'],
  ['api.fireworks.ai',                  'fireworks'],
  ['api.sambanova.ai',                  'sambanova'],
  ['api.cerebras.ai',                   'cerebras'],
  ['openrouter.ai',                     'openrouter'],
  ['api.replicate.com',                 'replicate'],
  ['api.nvidia.com',                    'nvidia'],
  ['integrate.api.nvidia.com',          'nvidia'],
];

/**
 * Detect LLM provider from a URL string or Request object.
 * Returns null for non-LLM URLs and localhost (Aegivis backend traffic).
 */
export function detectProvider(urlOrRequest: RequestInfo | URL | string): string | null {
  try {
    const urlStr =
      typeof urlOrRequest === 'string'
        ? urlOrRequest
        : urlOrRequest instanceof URL
          ? urlOrRequest.href
          : (urlOrRequest as Request).url;

    const host = new URL(urlStr).hostname.toLowerCase();

    // Never intercept localhost / private hosts — that's the Aegivis backend.
    if (
      host === 'localhost' ||
      host === '127.0.0.1' ||
      host === '::1' ||
      host.endsWith('.local')
    ) {
      return null;
    }

    for (const [fragment, provider] of PROVIDER_MAP) {
      if (host.includes(fragment)) return provider;
    }
    return null;
  } catch {
    return null;
  }
}
