/**
 * Vitest configuration for the AgentBlackBox dashboard.
 *
 * Merges with the existing Vite config so plugins (react, path aliases, etc.)
 * are automatically available in tests.
 */
import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      /** Use dedicated tsconfig.test.json so @testing-library/jest-dom types
       *  are available in tests without polluting the production build tsconfig. */
      typecheck: { tsconfig: "./tsconfig.test.json" },
      /** Use jsdom to simulate a browser DOM for React component tests. */
      environment: "jsdom",

      /** Import jest-dom matchers and any global test utilities before each file. */
      setupFiles: ["./src/test/setup.ts"],

      /**
       * Expose `describe`, `it`, `expect`, `vi`, etc. as globals so individual
       * test files don't need to import them.
       */
      globals: true,

      /** Only pick up files ending in .spec.ts / .spec.tsx. */
      include: ["src/**/*.spec.{ts,tsx}"],

      coverage: {
        provider: "v8",
        reporter: ["text", "lcov"],
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/test/**", "src/main.tsx", "src/**/*.module.css"],
      },
    },
  })
);
