import type { EmbeddingData } from '../extractors/types'

/**
 * Embedding dtype conversions applied by ComplexityMetric.__call__
 * (base.py:11-24 via rollout.py to_continuous/to_discrete, where the
 * embedding's D plays the role of the channel dimension).
 */

/** Bin continuous values into 2 states over [0,1]; no-op if already discrete. */
export function embeddingToDiscrete(e: EmbeddingData, numBins = 2, lo = 0, hi = 1): EmbeddingData {
  if (e.isDiscrete) return e
  const out = new Int32Array(e.data.length)
  const scale = 1 / (hi - lo)
  for (let i = 0; i < e.data.length; i++) {
    let x = (e.data[i] - lo) * scale
    x = x < 0 ? 0 : x > 1 ? 1 : x
    let v = Math.floor(x * numBins)
    if (v > numBins - 1) v = numBins - 1
    out[i] = v
  }
  return { ...e, data: out, isDiscrete: true, numStates: numBins }
}

/** One-hot expand a discrete (B,T,D) embedding to (B,T,D*K), channel d -> [d*K, (d+1)*K). */
export function embeddingToContinuous(e: EmbeddingData): EmbeddingData {
  if (!e.isDiscrete) {
    if (e.data instanceof Float32Array) return e
    return { ...e, data: Float32Array.from(e.data) }
  }
  const K = e.numStates ?? 2
  const { B, T, D } = e
  const out = new Float32Array(B * T * D * K)
  for (let bt = 0; bt < B * T; bt++) {
    const src = bt * D
    const dst = bt * D * K
    for (let d = 0; d < D; d++) {
      let s = e.data[src + d] | 0
      if (s < 0) s = 0
      else if (s >= K) s = K - 1
      out[dst + d * K + s] = 1
    }
  }
  return { ...e, data: out, D: D * K, isDiscrete: false, numStates: null }
}

/** K = explicit -> embedding metadata -> max value + 1 (utils.num_states). */
export function inferNumStates(e: EmbeddingData): number {
  if (e.numStates != null) return e.numStates
  let max = -1
  for (let i = 0; i < e.data.length; i++) if (e.data[i] > max) max = e.data[i]
  return e.data.length === 0 ? 1 : (max | 0) + 1
}
