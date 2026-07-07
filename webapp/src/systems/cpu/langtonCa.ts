import type { Dims, Stepper } from '../types'
import { MOD32, hash31, makeCoeffU32, qThreshold, seedTerm32, stateFromHash } from '../langtonHash'

/**
 * Langton-style implicit-rule CA with K states (systems.py LangtonCA1D/2D).
 * The rule table is never materialized: each neighborhood is hashed and the
 * hash decoded into a state, with lambda controlling the quiescent fraction.
 */
export interface LangtonParams {
  numStates: number
  /** Neighborhood size: K for 1D, K (side length) for 2D. */
  kernelSize: number
  /** Per-batch Langton lambda in [0,1]. */
  lambda: Float32Array
  /** Per-batch rule seed (uint32). */
  seed: Uint32Array
}

interface Precomputed {
  coeff: Uint32Array
  seedTerm: Uint32Array
  /** Float64: q can be 2^31 (lambda=0), which overflows int32. */
  q: Float64Array
}

function precompute(params: LangtonParams, L: number): Precomputed {
  const B = params.lambda.length
  const coeff = makeCoeffU32(L)
  const seedTerm = new Uint32Array(B)
  const q = new Float64Array(B)
  for (let b = 0; b < B; b++) {
    seedTerm[b] = seedTerm32(params.seed[b])
    q[b] = qThreshold(params.lambda[b])
  }
  return { coeff, seedTerm, q }
}

/** Per-cell decode: acc (mod 2^32) -> state in [0, K-1]. */
function decode(accMod32: number, seedTerm: number, q: number, K: number): number {
  return stateFromHash(hash31(accMod32, seedTerm), q, K)
}

export function makeLangtonCa1dStepper(
  params: LangtonParams,
  x0: Int32Array,
  dims: Dims,
): Stepper {
  const { B, W } = dims
  const K = params.numStates
  const kernel = params.kernelSize
  const pad = (kernel - 1) >> 1
  const { coeff, seedTerm, q } = precompute(params, kernel)
  let cur = Int32Array.from(x0)
  let next = new Int32Array(cur.length)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * W
          const st = seedTerm[b]
          const qb = q[b]
          for (let x = 0; x < W; x++) {
            // exact in doubles: sum of L terms each < 2^36
            let acc = 0
            for (let p = 0; p < kernel; p++) {
              acc += cur[off + ((x - pad + p + W) % W)] * coeff[p]
            }
            next[off + x] = K <= 1 ? 0 : decode(acc % MOD32, st, qb, K)
          }
        }
        const tmp = cur
        cur = next
        next = tmp
      }
    },
    readState: async () => cur,
    dispose() {},
  }
}

export function makeLangtonCa2dStepper(
  params: LangtonParams,
  x0: Int32Array,
  dims: Dims,
): Stepper {
  const { B, H, W } = dims
  const K = params.numStates
  const kernel = params.kernelSize
  const pad = (kernel - 1) >> 1
  const L = kernel * kernel
  const { coeff, seedTerm, q } = precompute(params, L)
  let cur = Int32Array.from(x0)
  let next = new Int32Array(cur.length)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * H * W
          const st = seedTerm[b]
          const qb = q[b]
          for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
              // coeff index is row-major over the window (matches unfold order)
              let acc = 0
              let j = 0
              for (let ky = 0; ky < kernel; ky++) {
                const yy = (y - pad + ky + H) % H
                for (let kx = 0; kx < kernel; kx++) {
                  acc += cur[off + yy * W + ((x - pad + kx + W) % W)] * coeff[j]
                  j++
                }
              }
              next[off + y * W + x] = K <= 1 ? 0 : decode(acc % MOD32, st, qb, K)
            }
          }
        }
        const tmp = cur
        cur = next
        next = tmp
      }
    },
    readState: async () => cur,
    dispose() {},
  }
}
