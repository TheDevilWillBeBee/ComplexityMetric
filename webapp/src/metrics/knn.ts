import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'
import { r2, splitIndices, standardize } from './linalg'

/**
 * KNNTimeRegression (time.py:42-59): 1-nearest-neighbor (Euclidean,
 * first-index tie-break like torch argmin) prediction of the normalized
 * time index on a seeded 50/50 split. Returns 0 when T < 6.
 */
export function makeKnnTimeRegression(trainFrac = 0.5, seed = 0): Metric {
  return {
    inputType: 'continuous',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      for (let b = 0; b < B; b++) {
        if (T < 6) continue
        const z = standardize(embeddingBatchView(e, b) as Float32Array, T, D)
        const { train, test } = splitIndices(T, trainFrac, seed + 1000 * b)
        if (test.length === 0) continue

        const y = (t: number) => (T === 1 ? 0 : t / (T - 1))
        const pred = new Float64Array(test.length)
        const yTe = new Float64Array(test.length)
        for (let i = 0; i < test.length; i++) {
          const ti = test[i]
          let bestDist = Infinity
          let bestJ = 0
          for (let j = 0; j < train.length; j++) {
            const tj = train[j]
            let dist = 0
            for (let a = 0; a < D; a++) {
              const diff = z[ti * D + a] - z[tj * D + a]
              dist += diff * diff
            }
            if (dist < bestDist) {
              bestDist = dist
              bestJ = j
            }
          }
          pred[i] = y(train[bestJ])
          yTe[i] = y(ti)
        }
        let baseline = 0
        for (let j = 0; j < train.length; j++) baseline += y(train[j])
        baseline /= train.length
        out[b] = r2(yTe, pred, baseline)
      }
      return out
    },
  }
}
