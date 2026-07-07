/**
 * Seeded, deterministic PRNG streams.
 *
 * splitmix32 expands a 32-bit seed into stream state; sfc32 generates the
 * stream. Numeric parity with PyTorch RNG is a non-goal — determinism per
 * seed within this app is the contract.
 */

export interface Rng {
  /** Uniform float in [0, 1). */
  float(): number
  /** Uniform uint32. */
  uint32(): number
  /** Uniform integer in [0, maxExclusive). */
  int(maxExclusive: number): number
}

function splitmix32(seed: number): () => number {
  let s = seed >>> 0
  return () => {
    s = (s + 0x9e3779b9) >>> 0
    let z = s
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0
    return (z ^ (z >>> 15)) >>> 0
  }
}

function sfc32(a: number, b: number, c: number, d: number): () => number {
  return () => {
    a >>>= 0; b >>>= 0; c >>>= 0; d >>>= 0
    const t = (a + b) >>> 0
    a = b ^ (b >>> 9)
    b = (c + (c << 3)) >>> 0
    c = ((c << 21) | (c >>> 11)) >>> 0
    d = (d + 1) >>> 0
    const r = (t + d) >>> 0
    c = (c + r) >>> 0
    return r
  }
}

export function makeRng(seed: number): Rng {
  const mix = splitmix32(seed)
  const next = sfc32(mix(), mix(), mix(), mix())
  // warm up: sfc32 needs a few rounds to decorrelate from weak seeds
  for (let i = 0; i < 12; i++) next()
  return {
    uint32: next,
    float: () => next() / 4294967296,
    int: (maxExclusive: number) => next() % maxExclusive,
  }
}

/** Standard normal sample via Box-Muller. */
export function gaussian(rng: Rng): number {
  // avoid log(0)
  const u1 = 1 - rng.float()
  const u2 = rng.float()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/** Derive an independent stream from a base seed and a purpose label. */
export function forkRng(seed: number, label: string): Rng {
  // FNV-1a over the label, mixed into the seed
  let h = 0x811c9dc5
  for (let i = 0; i < label.length; i++) {
    h ^= label.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return makeRng((seed ^ h) >>> 0)
}
