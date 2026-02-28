/**
 * Vitest global test setup.
 *
 * Imported once before every test file via vitest.config.ts `setupFiles`.
 *
 * - Imports jest-dom matchers (toBeInTheDocument, toHaveTextContent, etc.)
 *   so they are available in every test without explicit imports.
 * - Suppresses noisy console.error output from React's act() warnings during
 *   async state updates in tests (optional, can be removed).
 */
import "@testing-library/jest-dom";

// Suppress ResizeObserver not implemented errors from jsdom
window.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

// Suppress matchMedia not implemented errors from jsdom
Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});
