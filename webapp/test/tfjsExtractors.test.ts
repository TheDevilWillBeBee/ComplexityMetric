import { beforeAll, describe, expect, it } from 'vitest'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-cpu'
import type { RolloutData } from '../src/systems/types'
import { nullProgress } from '../src/core/progress'
import { makeRandomConvNet } from '../src/extractors/tfjs/randomConvNet'
import { makeRandomVgg } from '../src/extractors/tfjs/randomVgg'
import { circularPad, gramReduce, instanceNorm } from '../src/extractors/tfjs/ops'
import { makeRng } from '../src/core/prng'

beforeAll(async () => {
  await tf.setBackend('cpu')
  await tf.ready()
})

function noiseRollout(B: number, T: number, H: number, W: number, seed = 7): RolloutData {
  const rng = makeRng(seed)
  const data = new Float32Array(B * T * H * W)
  for (let i = 0; i < data.length; i++) data[i] = rng.float()
  return {
    data,
    B,
    T,
    C: 1,
    H,
    W,
    spatialDim: H > 1 ? 2 : 1,
    isDiscrete: false,
    numStates: null,
    steps: T,
    every: 1,
    skip: 0,
    systemName: 'test',
    toRgb: () => {},
  }
}

describe('tfjs ops', () => {
  it('circularPad wraps boundaries', async () => {
    const x = tf.tensor4d([1, 2, 3, 4, 5, 6], [1, 2, 3, 1])
    const padded = circularPad(x, 1, 1)
    expect(padded.shape).toEqual([1, 4, 5, 1])
    const got = Array.from(await padded.data())
    // row layout after pad: wrap of [[1,2,3],[4,5,6]]
    expect(got.slice(0, 5)).toEqual([6, 4, 5, 6, 4]) // top pad = last row wrapped
    expect(got.slice(5, 10)).toEqual([3, 1, 2, 3, 1]) // first data row wrapped
  })

  it('instanceNorm normalizes each channel to ~zero mean, unit variance', async () => {
    const rng = makeRng(1)
    const data = Float32Array.from({ length: 2 * 8 * 8 * 3 }, () => rng.float() * 5)
    const x = tf.tensor4d(data, [2, 8, 8, 3])
    const y = instanceNorm(x)
    const { mean, variance } = tf.moments(y, [1, 2])
    const means = Array.from(await mean.data())
    const vars_ = Array.from(await variance.data())
    for (const m of means) expect(Math.abs(m)).toBeLessThan(1e-4)
    for (const v of vars_) expect(v).toBeCloseTo(1, 2)
  })

  it('gramReduce computes f f^T / spatial', async () => {
    // 1 frame, 2x1 spatial, 2 channels: f = [[1,2],[3,4]] (spatial x ch)
    const x = tf.tensor4d([1, 2, 3, 4], [1, 2, 1, 2])
    const g = Array.from(await gramReduce(x).data())
    // gram[c1,c2] = sum_s f[s,c1] f[s,c2] / 2
    expect(g[0]).toBeCloseTo((1 * 1 + 3 * 3) / 2, 5)
    expect(g[1]).toBeCloseTo((1 * 2 + 3 * 4) / 2, 5)
    expect(g[2]).toBeCloseTo((2 * 1 + 4 * 3) / 2, 5)
    expect(g[3]).toBeCloseTo((2 * 2 + 4 * 4) / 2, 5)
  })
})

describe('RandomConvNet', () => {
  it('produces (B,T,embedDim) with unit-norm rows, deterministic per seed', async () => {
    const r = noiseRollout(2, 3, 16, 16)
    const make = () => makeRandomConvNet({ embedDim: 32, seed: 5 })
    const e1 = await make().extract(r, nullProgress)
    const e2 = await make().extract(r, nullProgress)
    expect(e1.B).toBe(2)
    expect(e1.T).toBe(3)
    expect(e1.D).toBe(32)
    for (let row = 0; row < e1.B * e1.T; row++) {
      let norm = 0
      for (let d = 0; d < 32; d++) norm += e1.data[row * 32 + d] ** 2
      expect(Math.sqrt(norm)).toBeCloseTo(1, 4)
    }
    expect(Array.from(e1.data as Float32Array)).toEqual(Array.from(e2.data as Float32Array))

    const e3 = await makeRandomConvNet({ embedDim: 32, seed: 6 }).extract(r, nullProgress)
    expect(Array.from(e3.data as Float32Array)).not.toEqual(Array.from(e1.data as Float32Array))
  })

  it('handles 1D rollouts', async () => {
    const r = noiseRollout(1, 2, 1, 32)
    const e = await makeRandomConvNet({ embedDim: 16, seed: 0 }).extract(r, nullProgress)
    expect(e.D).toBe(16)
    expect(e.data.length).toBe(2 * 16)
  })
})

describe('RandomVGG', () => {
  it('produces (B,T,embedDim) with unit-norm rows, deterministic per seed', async () => {
    const r = noiseRollout(1, 2, 16, 16)
    const make = () => makeRandomVgg({ embedDim: 24, seed: 9 })
    const e1 = await make().extract(r, nullProgress)
    const e2 = await make().extract(r, nullProgress)
    expect(e1.D).toBe(24)
    for (let row = 0; row < e1.B * e1.T; row++) {
      let norm = 0
      for (let d = 0; d < 24; d++) norm += e1.data[row * 24 + d] ** 2
      expect(Math.sqrt(norm)).toBeCloseTo(1, 4)
    }
    expect(Array.from(e1.data as Float32Array)).toEqual(Array.from(e2.data as Float32Array))
  })

  it('handles 1D rollouts (pools width only)', async () => {
    const r = noiseRollout(1, 2, 1, 64)
    const e = await makeRandomVgg({ embedDim: 16, seed: 0 }).extract(r, nullProgress)
    expect(e.D).toBe(16)
    expect(e.data.length).toBe(2 * 16)
  })

  it('one-hot expands discrete rollouts before convolving', async () => {
    const rng = makeRng(3)
    const data = new Int32Array(1 * 2 * 16 * 16)
    for (let i = 0; i < data.length; i++) data[i] = rng.int(4)
    const r: RolloutData = { ...noiseRollout(1, 2, 16, 16), data, isDiscrete: true, numStates: 4 }
    const e = await makeRandomVgg({ embedDim: 16, seed: 0 }).extract(r, nullProgress)
    expect(e.D).toBe(16)
    expect(Number.isFinite(e.data[0])).toBe(true)
  })
})
