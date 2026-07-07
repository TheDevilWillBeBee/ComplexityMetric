import * as tf from '@tensorflow/tfjs-core'
import type { RolloutData } from '../../systems/types'
import type { ProgressReporter } from '../../core/progress'
import type { EmbeddingData } from '../types'
import { rolloutToContinuous } from '../../systems/convert'
import { initTf } from './backend'
import { l2Normalize } from './ops'

/** App-parity frame chunk size for the neural extractors (app_dynamics.py:359). */
export const NET_CHUNK = 64

/**
 * Convert a chunk of frames [start, start+n) of a continuous rollout
 * (B,T,C,H,W row-major; frame index = b*T + t) into an NHWC tensor.
 */
function framesChunkToNhwc(r: RolloutData, start: number, n: number): tf.Tensor4D {
  const { C, H, W } = r
  const frameSize = C * H * W
  const out = new Float32Array(n * H * W * C)
  for (let i = 0; i < n; i++) {
    const src = (start + i) * frameSize
    for (let c = 0; c < C; c++) {
      const cOff = src + c * H * W
      for (let s = 0; s < H * W; s++) {
        out[(i * H * W + s) * C + c] = r.data[cOff + s]
      }
    }
  }
  return tf.tensor4d(out, [n, H, W, C])
}

/**
 * Shared chunked-execution loop for the tfjs extractors: one-hot/float
 * conversion, chunks of NET_CHUNK frames, tf.tidy per chunk, final row
 * L2-normalization (unit_norm=True for both nets), async readback as the
 * cooperative yield point.
 */
export async function extractChunked(
  r: RolloutData,
  embedDim: number,
  buildOnce: (inputChannels: number) => void,
  run: (x: tf.Tensor4D) => tf.Tensor2D,
  progress: ProgressReporter,
): Promise<EmbeddingData> {
  await initTf()
  const rc = rolloutToContinuous(r)
  buildOnce(rc.C)
  const total = rc.B * rc.T
  const out = new Float32Array(total * embedDim)
  for (let start = 0; start < total; start += NET_CHUNK) {
    const n = Math.min(NET_CHUNK, total - start)
    const y = tf.tidy(() => {
      const x = framesChunkToNhwc(rc, start, n)
      return l2Normalize(run(x))
    })
    const data = await y.data()
    y.dispose()
    out.set(data as Float32Array, start * embedDim)
    progress.set((start + n) / total)
    await progress.tick()
  }
  return { data: out, B: rc.B, T: rc.T, D: embedDim, isDiscrete: false, numStates: null }
}
