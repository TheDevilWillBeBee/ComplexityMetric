import type { Dims, Stepper } from '../types'

/**
 * Outer-totalistic binary CA, 1D and 2D (systems.py OuterTotalisticCA1D/2D).
 *
 * Neighbor sum over the K (or K x K) window EXCLUDING the center, periodic
 * boundaries. Dead cells become alive when birth[nsum] == 1; live cells
 * stay alive when survive[nsum] == 1. Table length L = K for 1D, K*K for 2D
 * (nsum can reach L-1).
 */
export interface OuterTotalisticParams {
  K: number
  /** (B, L) tables, L = K (1D) or K*K (2D). */
  birth: Uint8Array
  survive: Uint8Array
}

export function expandTable(table: Uint8Array, B: number): Uint8Array {
  const out = new Uint8Array(B * table.length)
  for (let b = 0; b < B; b++) out.set(table, b * table.length)
  return out
}

export function makeOuterTotalistic1dStepper(
  params: OuterTotalisticParams,
  x0: Int32Array,
  dims: Dims,
): Stepper {
  const { B, W } = dims
  const { K, birth, survive } = params
  const L = K
  const pad = (K - 1) >> 1
  let cur = Int32Array.from(x0)
  let next = new Int32Array(cur.length)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * W
          const tOff = b * L
          for (let x = 0; x < W; x++) {
            let nsum = 0
            for (let p = 0; p < K; p++) nsum += cur[off + ((x - pad + p + W) % W)]
            const c = cur[off + x]
            nsum -= c
            next[off + x] = c > 0 ? survive[tOff + nsum] : birth[tOff + nsum]
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

export function makeOuterTotalistic2dStepper(
  params: OuterTotalisticParams,
  x0: Int32Array,
  dims: Dims,
): Stepper {
  const { B, H, W } = dims
  const { K, birth, survive } = params
  const L = K * K
  const pad = (K - 1) >> 1
  let cur = Int32Array.from(x0)
  let next = new Int32Array(cur.length)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * H * W
          const tOff = b * L
          for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
              let nsum = 0
              for (let ky = 0; ky < K; ky++) {
                const yy = (y - pad + ky + H) % H
                for (let kx = 0; kx < K; kx++) {
                  nsum += cur[off + yy * W + ((x - pad + kx + W) % W)]
                }
              }
              const c = cur[off + y * W + x]
              nsum -= c
              next[off + y * W + x] = c > 0 ? survive[tOff + nsum] : birth[tOff + nsum]
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
