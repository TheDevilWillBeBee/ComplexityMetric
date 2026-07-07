import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'
import { inferNumStates } from './convert'

export type EntropyMode = 'time' | 'space' | 'both'

const LOG2 = Math.log(2)

/**
 * Normalized categorical entropy of a set of counts (utils.categorical_entropy):
 * p clamped to >= 1e-8, entropy in bits / max(log2 k, 1).
 */
function entropyOfCounts(counts: Float64Array, total: number, k: number): number {
  let h = 0
  for (let s = 0; s < k; s++) {
    let p = counts[s] / total
    if (p < 1e-8) p = 1e-8
    h -= p * (Math.log(p) / LOG2)
  }
  return h / Math.max(Math.log(k) / LOG2, 1)
}

/** Entropy metric (discrete.py:13-29), modes time / space / both. */
export function makeEntropy(mode: EntropyMode): Metric {
  return {
    inputType: 'discrete',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      const k = inferNumStates(e)
      if (k <= 1 || T === 0 || D === 0) return out
      const counts = new Float64Array(k)
      for (let b = 0; b < B; b++) {
        const x = embeddingBatchView(e, b)
        let score = 0
        if (mode === 'time') {
          // entropy over time per feature, averaged over features
          for (let d = 0; d < D; d++) {
            counts.fill(0)
            for (let t = 0; t < T; t++) counts[x[t * D + d]] += 1
            score += entropyOfCounts(counts, T, k)
          }
          score /= D
        } else if (mode === 'space') {
          for (let t = 0; t < T; t++) {
            counts.fill(0)
            for (let d = 0; d < D; d++) counts[x[t * D + d]] += 1
            score += entropyOfCounts(counts, D, k)
          }
          score /= T
        } else {
          counts.fill(0)
          for (let i = 0; i < T * D; i++) counts[x[i]] += 1
          score = entropyOfCounts(counts, T * D, k)
        }
        out[b] = Math.min(1, Math.max(0, score))
      }
      return out
    },
  }
}
