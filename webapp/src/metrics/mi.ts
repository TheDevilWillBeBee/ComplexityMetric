import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'
import { inferNumStates } from './convert'

const LOG2 = Math.log(2)

/**
 * Normalized mutual information from a joint distribution over k x k
 * (discrete.py _mi): the joint is clamped to >= 1e-8 BEFORE the marginals
 * are computed from it — replicate that order exactly.
 */
function miOfJoint(joint: Float64Array, k: number): number {
  for (let i = 0; i < k * k; i++) if (joint[i] < 1e-8) joint[i] = 1e-8
  const px = new Float64Array(k)
  const py = new Float64Array(k)
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      px[i] += joint[i * k + j]
      py[j] += joint[i * k + j]
    }
  }
  for (let i = 0; i < k; i++) {
    if (px[i] < 1e-8) px[i] = 1e-8
    if (py[i] < 1e-8) py[i] = 1e-8
  }
  let mi = 0
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      const p = joint[i * k + j]
      mi += p * (Math.log(p / (px[i] * py[j])) / LOG2)
    }
  }
  return mi / Math.max(Math.log(k) / LOG2, 1)
}

/**
 * FutureStateMutualInformation (discrete.py:32-55): MI between x[t] and
 * x[t+1]. Mode 'time': joint per feature over time pairs, averaged over
 * features. Mode 'space': joint per time pair over features, averaged.
 */
export function makeFutureStateMI(mode: 'time' | 'space'): Metric {
  return {
    inputType: 'discrete',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      if (T < 2 || D === 0) return out
      const k = inferNumStates(e)
      if (k <= 1) return out
      const joint = new Float64Array(k * k)
      for (let b = 0; b < B; b++) {
        const x = embeddingBatchView(e, b)
        let score = 0
        if (mode === 'time') {
          for (let d = 0; d < D; d++) {
            joint.fill(0)
            for (let t = 0; t < T - 1; t++) {
              joint[x[t * D + d] * k + x[(t + 1) * D + d]] += 1 / (T - 1)
            }
            score += miOfJoint(joint, k)
          }
          score /= D
        } else {
          for (let t = 0; t < T - 1; t++) {
            joint.fill(0)
            for (let d = 0; d < D; d++) {
              joint[x[t * D + d] * k + x[(t + 1) * D + d]] += 1 / Math.max(D, 1)
            }
            score += miOfJoint(joint, k)
          }
          score /= T - 1
        }
        out[b] = Math.min(1, Math.max(0, score))
      }
      return out
    },
  }
}
