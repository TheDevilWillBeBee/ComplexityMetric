import type { Dims } from '../types'
import type { Rng } from '../../core/prng'

export type SeedMode = 'noise' | 'zeros' | 'ones' | 'single'

/**
 * Initial state for discrete single-channel systems (systems.py seed()):
 * noise = Bernoulli(p), single = one live cell at (H//2, W//2).
 * For Langton systems pass numStates > 2: noise draws uniform in [0, K).
 */
export function discreteSeed(
  dims: Dims,
  mode: SeedMode,
  p: number,
  rng: Rng,
  numStates = 2,
): Int32Array {
  const { B, C, H, W } = dims
  const x = new Int32Array(B * C * H * W)
  if (mode === 'zeros') return x
  if (mode === 'ones') {
    x.fill(1)
    return x
  }
  if (mode === 'single') {
    const cy = Math.floor(H / 2)
    const cx = Math.floor(W / 2)
    for (let b = 0; b < B; b++)
      for (let c = 0; c < C; c++) x[((b * C + c) * H + cy) * W + cx] = 1
    return x
  }
  // noise
  if (numStates <= 2) {
    for (let i = 0; i < x.length; i++) x[i] = rng.float() < p ? 1 : 0
  } else {
    for (let i = 0; i < x.length; i++) x[i] = rng.int(numStates)
  }
  return x
}
