import type { Dims, Stepper } from '../types'
import type { Rng } from '../../core/prng'

/**
 * Elementary / general binary 1D CA (systems.py BinaryCA1D).
 *
 * Neighborhood index: leftmost neighbor is the most significant bit —
 * the Python code cross-correlates with kernel [2^(K-1), ..., 2^0] under
 * circular padding ((K-1)/2, K/2). For K=3 this is Wolfram numbering:
 * idx = 4*left + 2*center + 1*right, next = rule[idx].
 */
export interface BinaryCa1dParams {
  K: number
  /** (B, 2^K) rule table, values in {0,1}. */
  rule: Uint8Array
}

/**
 * One rule table of size 2^K. `ruleInt` uses Wolfram numbering
 * (bit i of ruleInt = output for neighborhood index i); null samples a
 * random table with density p.
 */
export function sampleBinaryCa1dRule(
  K: number,
  ruleInt: number | null,
  rng: Rng,
  p = 0.5,
): Uint8Array {
  const size = 1 << K
  const rule = new Uint8Array(size)
  if (ruleInt !== null) {
    // avoid JS 32-bit shift wrap for large i: use float division
    for (let i = 0; i < size; i++) rule[i] = Math.floor(ruleInt / 2 ** i) % 2
  } else {
    for (let i = 0; i < size; i++) rule[i] = rng.float() < p ? 1 : 0
  }
  return rule
}

/** Expand a single rule table to all B batch elements. */
export function expandRule(rule: Uint8Array, B: number): Uint8Array {
  const out = new Uint8Array(B * rule.length)
  for (let b = 0; b < B; b++) out.set(rule, b * rule.length)
  return out
}

export function makeBinaryCa1dStepper(
  params: BinaryCa1dParams,
  x0: Int32Array,
  dims: Dims,
): Stepper {
  const { B, W } = dims
  const { K } = params
  const tableSize = 1 << K
  const pad = (K - 1) >> 1
  let cur = Int32Array.from(x0)
  let next = new Int32Array(cur.length)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * W
          const ruleOff = b * tableSize
          for (let x = 0; x < W; x++) {
            let idx = 0
            for (let p = 0; p < K; p++) {
              const xi = (x - pad + p + W) % W
              idx = (idx << 1) | cur[off + xi]
            }
            next[off + x] = params.rule[ruleOff + idx]
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
