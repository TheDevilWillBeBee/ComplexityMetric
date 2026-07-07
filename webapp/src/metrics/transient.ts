import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'

/**
 * DensityTransientTime (discrete.py:106-162): mean feature density per
 * step; find the first time the density stays within
 * (tail population-std * 1.0 + 0.05) of the tail mean for 100 consecutive
 * steps. tail = last min(256, T) steps. Returns T (i.e. 1.0 normalized)
 * when T < 100 or no entry point exists; normalized by T.
 */
export function makeDensityTransientTime(
  tailLength = 256,
  epsilon = 5e-2,
  confirmationWindow = 100,
  tailStdScale = 1.0,
): Metric {
  return {
    inputType: 'discrete',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      if (D === 0 || T === 0) return out
      const density = new Float64Array(T)
      for (let b = 0; b < B; b++) {
        const x = embeddingBatchView(e, b)
        for (let t = 0; t < T; t++) {
          let s = 0
          for (let d = 0; d < D; d++) s += x[t * D + d]
          density[t] = s / D
        }
        const tailStart = Math.max(0, T - tailLength)
        const n = T - tailStart
        let mean = 0
        for (let t = tailStart; t < T; t++) mean += density[t]
        mean /= n
        let varSum = 0
        for (let t = tailStart; t < T; t++) varSum += (density[t] - mean) ** 2
        const tol = Math.sqrt(varSum / n) * tailStdScale + epsilon

        let lifetime = T
        const W = confirmationWindow
        if (T >= W) {
          // sliding count of in-tolerance steps
          let run = 0
          for (let t = 0; t < T; t++) {
            if (Math.abs(density[t] - mean) <= tol) {
              run++
              if (run >= W) {
                lifetime = t - W + 1
                break
              }
            } else {
              run = 0
            }
          }
        }
        out[b] = lifetime / Math.max(T, 1)
      }
      return out
    },
  }
}
