import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'
import { r2, ridgeFit, splitIndices, standardize } from './linalg'

/**
 * LinearRidgeTimeRegression (time.py:19-39): standardize features over
 * time, regress the normalized time index on a seeded 50/50 split with
 * ridge 1e-3 (bias included in the penalty), return test R^2 in [0,1].
 * Returns 0 when T < 4. Per-batch split seed = seed + 1000*i.
 */
export function makeLinearRidgeTimeRegression(trainFrac = 0.5, ridge = 1e-3, seed = 0): Metric {
  return {
    inputType: 'continuous',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      for (let b = 0; b < B; b++) {
        if (T < 4) continue
        const zRaw = embeddingBatchView(e, b) as Float32Array
        const z = standardize(zRaw, T, D)
        const { train, test } = splitIndices(T, trainFrac, seed + 1000 * b)
        if (test.length === 0) continue

        const d = D + 1 // bias column
        const XTr = new Float64Array(train.length * d)
        const yTr = new Float64Array(train.length)
        for (let i = 0; i < train.length; i++) {
          const t = train[i]
          for (let a = 0; a < D; a++) XTr[i * d + a] = z[t * D + a]
          XTr[i * d + D] = 1
          yTr[i] = T === 1 ? 0 : t / (T - 1)
        }
        const w = ridgeFit(XTr, yTr, train.length, d, ridge)

        const pred = new Float64Array(test.length)
        const yTe = new Float64Array(test.length)
        for (let i = 0; i < test.length; i++) {
          const t = test[i]
          let s = w[D]
          for (let a = 0; a < D; a++) s += z[t * D + a] * w[a]
          pred[i] = s
          yTe[i] = T === 1 ? 0 : t / (T - 1)
        }
        let baseline = 0
        for (let i = 0; i < yTr.length; i++) baseline += yTr[i]
        baseline /= yTr.length
        out[b] = r2(yTe, pred, baseline)
      }
      return out
    },
  }
}
