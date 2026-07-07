/**
 * Langton CA implicit-rule hash pipeline — exact port of
 * DynamicalSystems/systems.py (LangtonCABase, lines 1922-2333).
 *
 * All arithmetic is mod 2^32. CPU code must use `Math.imul(...) >>> 0` and
 * logical shifts (`>>>`) only; the WGSL port uses native u32 ops. This file
 * is the single source of truth shared by the CPU stepper, the GPU
 * precompute, and the tests.
 */

const MASK31 = 0x7fffffff
const MOD31 = 2147483648 // 2^31
const MOD32 = 4294967296 // 2^32
const SEED_MUL = 0x9e3779b1
const ACC_OFFSET = 0x85ebca6b
const FMIX_MUL1 = 0x7feb352d
const FMIX_MUL2 = 0x846ca68b

/** (a*b) mod 2^32, unsigned. */
export function mul32(a: number, b: number): number {
  return Math.imul(a >>> 0, b >>> 0) >>> 0
}

/** MurmurHash3-style finalizer (lowbias32 constants), mod 2^32. */
export function fmix32(x: number): number {
  let h = x >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  h = mul32(h, FMIX_MUL1)
  h = (h ^ (h >>> 15)) >>> 0
  h = mul32(h, FMIX_MUL2)
  h = (h ^ (h >>> 16)) >>> 0
  return h
}

/**
 * L neighborhood coefficients from a SplitMix64 stream seeded with the
 * fixed constant 0x123456789ABCDEF0 (systems.py:2092-2108). Python stores
 * them as *signed* int32; we return the unsigned mod-2^32 representation,
 * which is equivalent under mod-2^32 accumulation.
 */
export function makeCoeffU32(L: number): Uint32Array {
  if (L <= 0) throw new Error('L must be >= 1')
  const M64 = (1n << 64n) - 1n
  let x = 0x123456789abcdef0n
  const out = new Uint32Array(L)
  for (let i = 0; i < L; i++) {
    x = (x + 0x9e3779b97f4a7c15n) & M64
    let z = x
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & M64
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & M64
    z = z ^ (z >> 31n)
    let c = Number(z & 0xffffffffn)
    if (c === 0) c = 1
    out[i] = c >>> 0
  }
  return out
}

/** seed_term = (seed * SEED_MUL) mod 2^32 (systems.py:2301). */
export function seedTerm32(seed: number): number {
  return mul32(seed >>> 0, SEED_MUL)
}

/**
 * Quiescent threshold q = floor(float32(1 - lambda) * 2^31), clamped to
 * [0, 2^31]. Torch does the subtraction and multiplication in float32
 * (systems.py:2321) — replicate with Math.fround so CPU and GPU paths use
 * bit-identical thresholds.
 */
export function qThreshold(lambda: number): number {
  const q = Math.floor(Math.fround(Math.fround(1 - Math.fround(lambda)) * MOD31))
  return Math.min(Math.max(q, 0), MOD31)
}

/**
 * 31-bit hash of an accumulated neighborhood.
 * `accMod32` must already be reduced mod 2^32 (non-negative).
 */
export function hash31(accMod32: number, seedTerm: number): number {
  const x = (accMod32 + seedTerm + ACC_OFFSET) % MOD32
  return (fmix32(x) & MASK31) >>> 0
}

/** Decode a state in [0, K-1] from a 31-bit hash (systems.py:2308-2333). */
export function stateFromHash(h31: number, q: number, K: number): number {
  if (K <= 1) return 0
  if (h31 < q) return 0
  return 1 + ((h31 - q) % (K - 1))
}

export { MOD32 }
