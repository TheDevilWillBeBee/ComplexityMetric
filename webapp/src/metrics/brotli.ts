/** Minimal structural type for what we use from brotli-wasm. */
export interface BrotliModule {
  compress(buf: Uint8Array, options?: { quality?: number }): Uint8Array
}

let instance: BrotliModule | null = null

/**
 * Lazy singleton. Dynamic import keeps the ~1 MB wasm out of the critical
 * path — it loads on first CompressedRatio use. In vitest the import is
 * aliased to brotli-wasm's synchronous node entry (see vite.config.ts).
 */
export async function getBrotli(): Promise<BrotliModule> {
  if (!instance) {
    const mod = await import('brotli-wasm')
    instance = (await mod.default) as BrotliModule
  }
  return instance
}
