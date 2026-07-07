import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  // relative base so the built bundle works from any static path
  base: './',
  optimizeDeps: {
    // pre-bundling rewrites brotli-wasm's `new URL(..., import.meta.url)`
    // and the dev server then serves HTML instead of the .wasm binary
    exclude: ['brotli-wasm'],
  },
  build: {
    target: 'es2022',
  },
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
    alias: {
      // brotli-wasm's ESM entry fetch()es its wasm (browser-only); use the
      // synchronous node entry under vitest
      'brotli-wasm': fileURLToPath(new URL('./node_modules/brotli-wasm/index.node.js', import.meta.url)),
    },
  },
})
