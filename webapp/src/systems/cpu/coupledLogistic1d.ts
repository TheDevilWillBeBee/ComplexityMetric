import type { Dims, Stepper } from '../types'
import type { Rng } from '../../core/prng'

/**
 * Coupled logistic map lattice (systems.py CoupledLogistic1D):
 *   f(x) = r*x*(1-x)
 *   x_i' = clamp((1-eps)*f(x_i) + eps/2*(f(x_{i-1}) + f(x_{i+1})), 0, 1)
 * Periodic boundaries; state is continuous in [0,1].
 */
export interface CoupledLogisticParams {
  /** Per-batch r. */
  r: Float32Array
  /** Per-batch eps. */
  eps: Float32Array
}

export function coupledLogisticSeed(dims: Dims, rng: Rng): Float32Array {
  const x = new Float32Array(dims.B * dims.C * dims.H * dims.W)
  for (let i = 0; i < x.length; i++) x[i] = rng.float()
  return x
}

export function makeCoupledLogistic1dStepper(
  params: CoupledLogisticParams,
  x0: Float32Array,
  dims: Dims,
): Stepper {
  const { B, W } = dims
  let cur = Float32Array.from(x0)
  let next = new Float32Array(cur.length)
  const f = new Float32Array(W)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const off = b * W
          const r = params.r[b]
          const eps = params.eps[b]
          for (let x = 0; x < W; x++) {
            const v = cur[off + x]
            f[x] = r * v * (1 - v)
          }
          for (let x = 0; x < W; x++) {
            const fl = f[(x - 1 + W) % W]
            const fr = f[(x + 1) % W]
            const v = (1 - eps) * f[x] + 0.5 * eps * (fl + fr)
            next[off + x] = v < 0 ? 0 : v > 1 ? 1 : v
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
