import type { Dims, Stepper } from '../types'
import type { Rng } from '../../core/prng'
import { gaussian } from '../../core/prng'

/**
 * Gray-Scott reaction-diffusion (systems.py GrayScott2D), forward Euler
 * with dt=1 and the 5-point Laplacian [[0,1,0],[1,-4,1],[0,1,0]] under
 * periodic boundaries. State is (B, 2, H, W) with channels (u, v).
 */
export interface GrayScottParams {
  Du: Float32Array
  Dv: Float32Array
  F: Float32Array
  k: Float32Array
}

/**
 * seed() (systems.py:1793-1814): u=1, v=0 everywhere; central square of
 * half-side min(H,W)//10 flipped to u=0, v=1; plus Gaussian noise
 * (default 0.02), clamped to [0,1].
 */
export function grayScottSeed(dims: Dims, rng: Rng, noise = 0.02): Float32Array {
  const { B, H, W } = dims
  const x = new Float32Array(B * 2 * H * W)
  const r = Math.floor(Math.min(H, W) / 10)
  const cy = Math.floor(H / 2)
  const cx = Math.floor(W / 2)
  for (let b = 0; b < B; b++) {
    const uOff = b * 2 * H * W
    const vOff = uOff + H * W
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < W; xx++) {
        const inSquare = y >= cy - r && y < cy + r && xx >= cx - r && xx < cx + r
        let u = inSquare ? 0 : 1
        let v = inSquare ? 1 : 0
        if (noise > 0) {
          u += noise * gaussian(rng)
          v += noise * gaussian(rng)
        }
        x[uOff + y * W + xx] = u < 0 ? 0 : u > 1 ? 1 : u
        x[vOff + y * W + xx] = v < 0 ? 0 : v > 1 ? 1 : v
      }
    }
  }
  return x
}

export function makeGrayScott2dStepper(
  params: GrayScottParams,
  x0: Float32Array,
  dims: Dims,
  dt = 1.0,
): Stepper {
  const { B, H, W } = dims
  let cur = Float32Array.from(x0)
  let next = new Float32Array(cur.length)

  const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)

  return {
    step(count: number) {
      for (let s = 0; s < count; s++) {
        for (let b = 0; b < B; b++) {
          const uOff = b * 2 * H * W
          const vOff = uOff + H * W
          const Du = params.Du[b]
          const Dv = params.Dv[b]
          const F = params.F[b]
          const k = params.k[b]
          for (let y = 0; y < H; y++) {
            const yu = ((y - 1 + H) % H) * W
            const yd = ((y + 1) % H) * W
            const yc = y * W
            for (let x = 0; x < W; x++) {
              const xl = (x - 1 + W) % W
              const xr = (x + 1) % W
              const u = cur[uOff + yc + x]
              const v = cur[vOff + yc + x]
              const lapU =
                cur[uOff + yu + x] + cur[uOff + yd + x] + cur[uOff + yc + xl] + cur[uOff + yc + xr] - 4 * u
              const lapV =
                cur[vOff + yu + x] + cur[vOff + yd + x] + cur[vOff + yc + xl] + cur[vOff + yc + xr] - 4 * v
              const uvv = u * v * v
              next[uOff + yc + x] = clamp01(u + dt * (Du * lapU - uvv + F * (1 - u)))
              next[vOff + yc + x] = clamp01(v + dt * (Dv * lapV + uvv - (F + k) * v))
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
