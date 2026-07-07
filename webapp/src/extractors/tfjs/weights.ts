import * as tf from '@tensorflow/tfjs-core'
import type { Rng } from '../../core/prng'
import { forkRng } from '../../core/prng'

/**
 * PyTorch default layer init (kaiming_uniform(a=sqrt(5)) for weights,
 * uniform(+-1/sqrt(fan_in)) for biases) both reduce to U(-b, b) with
 * b = 1/sqrt(fan_in). Weights come from our PRNG so results are
 * deterministic per seed and identical across tfjs backends.
 */
function uniformData(rng: Rng, n: number, bound: number): Float32Array {
  const out = new Float32Array(n)
  for (let i = 0; i < n; i++) out[i] = (rng.float() * 2 - 1) * bound
  return out
}

export interface LinearWeights {
  /** (in, out) — apply as x @ W + b. */
  W: tf.Tensor2D
  b: tf.Tensor1D
}

export function makeLinear(seed: number, label: string, fanIn: number, fanOut: number): LinearWeights {
  const bound = 1 / Math.sqrt(fanIn)
  const wRng = forkRng(seed, `${label}-w`)
  const bRng = forkRng(seed, `${label}-b`)
  return {
    W: tf.tensor2d(uniformData(wRng, fanIn * fanOut, bound), [fanIn, fanOut]),
    b: tf.tensor1d(uniformData(bRng, fanOut, bound)),
  }
}

export interface ConvWeights {
  /** (kh, kw, inC, outC) — tfjs HWIO layout. */
  W: tf.Tensor4D
  b: tf.Tensor1D
}

export function makeConv(
  seed: number,
  label: string,
  kh: number,
  kw: number,
  inC: number,
  outC: number,
): ConvWeights {
  const fanIn = inC * kh * kw
  const bound = 1 / Math.sqrt(fanIn)
  const wRng = forkRng(seed, `${label}-w`)
  const bRng = forkRng(seed, `${label}-b`)
  return {
    W: tf.tensor4d(uniformData(wRng, kh * kw * inC * outC, bound), [kh, kw, inC, outC]),
    b: tf.tensor1d(uniformData(bRng, outC, bound)),
  }
}
