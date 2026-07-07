import type { RolloutData } from './types'

/**
 * Rollout dtype conversions (rollout.py:93-127).
 *
 * to_continuous: one-hot encode discrete states per channel — channel c with
 * K states becomes K channels at positions [c*K, (c+1)*K). No-op (float cast)
 * for continuous rollouts.
 *
 * to_discrete: bin continuous values over value_range into num_bins integer
 * states. No-op for already-discrete rollouts (keeps K).
 */

export function rolloutToContinuous(r: RolloutData): RolloutData {
  if (!r.isDiscrete) {
    if (r.data instanceof Float32Array) return r
    return { ...r, data: Float32Array.from(r.data) }
  }
  const K = r.numStates ?? 2
  const { B, T, C, H, W } = r
  const spatial = H * W
  const out = new Float32Array(B * T * C * K * spatial)
  const inFrame = C * spatial
  const outFrame = C * K * spatial
  for (let bt = 0; bt < B * T; bt++) {
    const src = bt * inFrame
    const dst = bt * outFrame
    for (let c = 0; c < C; c++) {
      for (let i = 0; i < spatial; i++) {
        let s = r.data[src + c * spatial + i] | 0
        if (s < 0) s = 0
        else if (s >= K) s = K - 1
        out[dst + (c * K + s) * spatial + i] = 1
      }
    }
  }
  return { ...r, data: out, C: C * K, isDiscrete: false, numStates: null }
}

export function rolloutToDiscrete(r: RolloutData, numBins = 2, lo = 0, hi = 1): RolloutData {
  if (r.isDiscrete) return r
  const out = new Int32Array(r.data.length)
  const scale = 1 / (hi - lo)
  for (let i = 0; i < r.data.length; i++) {
    let x = (r.data[i] - lo) * scale
    x = x < 0 ? 0 : x > 1 ? 1 : x
    let v = Math.floor(x * numBins)
    if (v > numBins - 1) v = numBins - 1
    out[i] = v
  }
  return { ...r, data: out, isDiscrete: true, numStates: numBins }
}
