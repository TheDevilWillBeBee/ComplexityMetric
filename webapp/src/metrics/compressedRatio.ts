import type { EmbeddingData } from '../extractors/types'
import type { Metric } from './types'
import { embeddingBatchView } from '../extractors/types'
import { inferNumStates } from './convert'
import { getBrotli } from './brotli'
import { makeEntropy } from './entropy'

const LOG2 = Math.log(2)

/** bytes per encoded state (utils.state_bytes): ceil(bitlength(max(1, k-1)) / 8). */
export function stateBytes(k: number): number {
  const bits = Math.max(1, k - 1).toString(2).length
  return Math.max(1, Math.ceil(bits / 8))
}

function encode(x: Int32Array | Float32Array, T: number, D: number, columnMajor: boolean): Uint8Array {
  // k <= 256 in this app -> always 1 byte per state
  const out = new Uint8Array(T * D)
  if (!columnMajor) {
    for (let i = 0; i < T * D; i++) out[i] = x[i]
  } else {
    let j = 0
    for (let d = 0; d < D; d++) for (let t = 0; t < T; t++) out[j++] = x[t * D + d]
  }
  return out
}

/**
 * CompressedRatio (discrete.py:58-89): brotli-compress the byte-encoded
 * states (quality 11 = Python default), normalized by the theoretical
 * entropy fraction log2(k) / (8 * state_bytes). Takes the min over
 * row-major and column-major flattening ("min_compression" default).
 */
export function makeCompressedRatio(): Metric {
  return {
    inputType: 'discrete',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const { B, T, D } = e
      const out = new Float64Array(B)
      const k = inferNumStates(e)
      if (k <= 1 || T * D === 0) return out
      if (k > 256) throw new Error(`CompressedRatio: k=${k} > 256 not supported`)
      const brotli = await getBrotli()
      const expected = (Math.log(k) / LOG2) / (8 * stateBytes(k))
      for (let b = 0; b < B; b++) {
        const x = embeddingBatchView(e, b)
        let best = Infinity
        for (const columnMajor of [false, true]) {
          const payload = encode(x, T, D, columnMajor)
          const compressed = brotli.compress(payload, { quality: 11 })
          best = Math.min(best, compressed.length / payload.length / expected)
        }
        out[b] = best
      }
      return out
    },
  }
}

/**
 * EntropyMinusCompressedRatio (discrete.py:92-100): Entropy (default mode
 * "time") minus CompressedRatio, per batch element.
 */
export function makeEntropyMinusCompressedRatio(): Metric {
  const entropy = makeEntropy('time')
  const ratio = makeCompressedRatio()
  return {
    inputType: 'discrete',
    async compute(e: EmbeddingData): Promise<Float64Array> {
      const a = await entropy.compute(e)
      const b = await ratio.compute(e)
      const out = new Float64Array(a.length)
      for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i]
      return out
    },
  }
}
