import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    intercept: 'src/intercept.ts',
    'adapters/langchain': 'src/adapters/langchain.ts',
    'adapters/openai': 'src/adapters/openai.ts',
    'adapters/vercel-ai': 'src/adapters/vercel-ai.ts',
  },
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  splitting: false,
  sourcemap: true,
  treeshake: true,
});
