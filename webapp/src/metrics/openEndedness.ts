import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'

/**
 * OpenEndedness (time.py:11-16): 1 - max pairwise cosine similarity below
 * the diagonal, clamped to >= 0. torch.tril zero-fills the rest of the
 * matrix before .max(), so the effective max is max(0, max_{i>j} sim) —
 * e.g. T=1 scores exactly 1.
 */
export function makeOpenEndedness(): Metric {
  return {
    inputType: 'continuous',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      const EPS = 1e-8 // F.cosine_similarity default eps
      for (let b = 0; b < B; b++) {
        const z = embeddingBatchView(e, b) as Float32Array
        const norms = new Float64Array(T)
        for (let t = 0; t < T; t++) {
          let s = 0
          for (let d = 0; d < D; d++) s += z[t * D + d] * z[t * D + d]
          norms[t] = Math.max(Math.sqrt(s), EPS)
        }
        let maxSim = 0 // tril zero fill
        for (let i = 1; i < T; i++) {
          for (let j = 0; j < i; j++) {
            let dot = 0
            for (let d = 0; d < D; d++) dot += z[i * D + d] * z[j * D + d]
            const sim = dot / (norms[i] * norms[j])
            if (sim > maxSim) maxSim = sim
          }
        }
        out[b] = Math.max(0, 1 - maxSim)
      }
      return out
    },
  }
}
