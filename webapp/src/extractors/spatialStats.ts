import type { Extractor } from './types'
import { rolloutToContinuous } from '../systems/convert'

/**
 * SpatialStatistics (extractors.py:96-104): per-channel mean and POPULATION
 * std over the spatial axes -> (B, T, 2*C), laid out [means..., stds...].
 */
export function makeSpatialStatistics(): Extractor {
  return {
    name: 'SpatialStatistics',
    inputType: 'continuous',
    outputType: 'continuous',
    async extract(r, progress) {
      const rc = rolloutToContinuous(r)
      const { B, T, C, H, W } = rc
      const spatial = H * W
      const D = 2 * C
      const out = new Float32Array(B * T * D)
      const frames = B * T
      for (let bt = 0; bt < frames; bt++) {
        const frameOff = bt * C * spatial
        for (let c = 0; c < C; c++) {
          const off = frameOff + c * spatial
          let sum = 0
          for (let i = 0; i < spatial; i++) sum += rc.data[off + i]
          const mean = sum / spatial
          let varSum = 0
          for (let i = 0; i < spatial; i++) {
            const d = rc.data[off + i] - mean
            varSum += d * d
          }
          out[bt * D + c] = mean
          out[bt * D + C + c] = Math.sqrt(varSum / spatial)
        }
        if ((bt & 63) === 0) {
          progress.set(bt / frames)
          await progress.tick()
        }
      }
      return { data: out, B, T, D, isDiscrete: false, numStates: null }
    },
    dispose() {},
  }
}
