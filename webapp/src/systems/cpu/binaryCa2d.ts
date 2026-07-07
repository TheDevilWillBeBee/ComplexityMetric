import type { Dims, Stepper } from '../types'
import type { Rng } from '../../core/prng'

/**
 * Random-rule binary 2D CA over the 3x3 neighborhood (systems.py BinaryCA2D).
 *
 * Neighborhood-to-index bit weights: the Python code stores
 * [[256,128,64],[32,16,8],[4,2,1]] flipped along dim 1 and cross-correlates
 * (conv2d), so the EFFECTIVE weights by neighbor offset are:
 *   (y-1): 64 128 256
 *   (y  ):  8  16  32     (center = 16)
 *   (y+1):  1   2   4
 * Verified against systems.py:449-453; bs_rule (line 476-484) depends on
 * this exact bit order.
 */
const OFFSETS: ReadonlyArray<readonly [number, number, number]> = [
  [-1, -1, 64],
  [-1, 0, 128],
  [-1, 1, 256],
  [0, -1, 8],
  [0, 0, 16],
  [0, 1, 32],
  [1, -1, 1],
  [1, 0, 2],
  [1, 1, 4],
]

export interface BinaryCa2dParams {
  /** (B, 512) rule table, values in {0,1}. */
  rule: Uint8Array
}

/** One random 512-entry table with density p (sample_params, systems.py:457-459). */
export function sampleBinaryCa2dRule(rng: Rng, p = 0.5): Uint8Array {
  const rule = new Uint8Array(512)
  for (let i = 0; i < 512; i++) rule[i] = rng.float() < p ? 1 : 0
  return rule
}

export function makeBinaryCa2dStepper(
  params: BinaryCa2dParams,
  x0: Int32Array,
  dims: Dims,
): Stepper {
  const { B, H, W } = dims
  let cur = Int32Array.from(x0)
  let next = new Int32Array(cur.length)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * H * W
          const ruleOff = b * 512
          for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
              let idx = 0
              for (const [dy, dx, w] of OFFSETS) {
                const yy = (y + dy + H) % H
                const xx = (x + dx + W) % W
                if (cur[off + yy * W + xx] > 0) idx += w
              }
              next[off + y * W + x] = params.rule[ruleOff + idx]
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
