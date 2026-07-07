import type { RolloutData } from '../systems/types'
import type { ProgressReporter } from '../core/progress'

export interface EmbeddingData {
  /** (B, T, D) row-major. Int32Array only for discrete embeddings. */
  data: Float32Array | Int32Array
  B: number
  T: number
  D: number
  isDiscrete: boolean
  numStates: number | null
}

export interface Extractor {
  name: string
  inputType: 'discrete' | 'continuous'
  outputType: 'discrete' | 'continuous'
  extract(r: RolloutData, progress: ProgressReporter): Promise<EmbeddingData>
  dispose(): void
}

/** View of one batch element's (T, D) block. */
export function embeddingBatchView(e: EmbeddingData, b: number): Float32Array | Int32Array {
  return e.data.subarray(b * e.T * e.D, (b + 1) * e.T * e.D) as Float32Array | Int32Array
}

/** Time-window slice [t0, t1) of an embedding — copies (metrics mutate nothing but need contiguous (B,size,D)). */
export function sliceEmbedding(e: EmbeddingData, t0: number, t1: number): EmbeddingData {
  const size = t1 - t0
  const out =
    e.data instanceof Int32Array ? new Int32Array(e.B * size * e.D) : new Float32Array(e.B * size * e.D)
  for (let b = 0; b < e.B; b++) {
    const src = (b * e.T + t0) * e.D
    out.set(e.data.subarray(src, src + size * e.D), b * size * e.D)
  }
  return { data: out, B: e.B, T: size, D: e.D, isDiscrete: e.isDiscrete, numStates: e.numStates }
}
