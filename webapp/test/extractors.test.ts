import { describe, expect, it } from 'vitest'
import type { RolloutData } from '../src/systems/types'
import { rolloutToContinuous, rolloutToDiscrete } from '../src/systems/convert'
import { makeContinuousFlatten, makeDiscreteFlatten } from '../src/extractors/flatten'
import { makeSpatialStatistics } from '../src/extractors/spatialStats'
import { nullProgress } from '../src/core/progress'
import { pcaProject } from '../src/projection/pca'

function discreteRollout(values: number[], B: number, T: number, W: number, K: number): RolloutData {
  return {
    data: Int32Array.from(values),
    B,
    T,
    C: 1,
    H: 1,
    W,
    spatialDim: 1,
    isDiscrete: true,
    numStates: K,
    steps: T,
    every: 1,
    skip: 0,
    systemName: 'test',
    toRgb: () => {},
  }
}

describe('convert', () => {
  it('one-hot encodes discrete rollouts per channel', () => {
    const r = discreteRollout([0, 2, 1, 1], 1, 2, 2, 3) // (1,2,1,2) K=3
    const c = rolloutToContinuous(r)
    expect(c.C).toBe(3)
    expect(c.isDiscrete).toBe(false)
    // frame (t=0): states [0,2] -> channels k=0:[1,0] k=1:[0,0] k=2:[0,1]
    expect(Array.from(c.data.subarray(0, 6))).toEqual([1, 0, 0, 0, 0, 1])
    // frame (t=1): states [1,1]
    expect(Array.from(c.data.subarray(6, 12))).toEqual([0, 0, 1, 1, 0, 0])
  })

  it('bins continuous rollouts into 2 states over [0,1]', () => {
    const r: RolloutData = {
      ...discreteRollout([], 1, 1, 4, 2),
      data: Float32Array.of(0, 0.49, 0.5, 1),
      isDiscrete: false,
      numStates: null,
    }
    const d = rolloutToDiscrete(r)
    expect(Array.from(d.data)).toEqual([0, 0, 1, 1])
    expect(d.numStates).toBe(2)
  })

  it('to_discrete is a no-op for discrete rollouts (keeps K)', () => {
    const r = discreteRollout([0, 3, 2, 1], 1, 2, 2, 4)
    const d = rolloutToDiscrete(r)
    expect(d.numStates).toBe(4)
    expect(d.data).toBe(r.data)
  })
})

describe('extractors', () => {
  it('DiscreteFlatten flattens frames and keeps K states', async () => {
    const r = discreteRollout([0, 2, 1, 1, 3, 0], 1, 3, 2, 4)
    const e = await makeDiscreteFlatten().extract(r, nullProgress)
    expect(e.D).toBe(2)
    expect(e.T).toBe(3)
    expect(e.isDiscrete).toBe(true)
    expect(e.numStates).toBe(4)
    expect(Array.from(e.data)).toEqual([0, 2, 1, 1, 3, 0])
  })

  it('ContinuousFlatten one-hot expands discrete rollouts', async () => {
    const r = discreteRollout([1, 0], 1, 1, 2, 2)
    const e = await makeContinuousFlatten().extract(r, nullProgress)
    // (C*K, W) = (2, 2) flattened row-major: k=0 channel [0,1], k=1 channel [1,0]
    expect(e.D).toBe(4)
    expect(Array.from(e.data)).toEqual([0, 1, 1, 0])
  })

  it('SpatialStatistics computes per-channel mean and population std', async () => {
    const r: RolloutData = {
      ...discreteRollout([], 1, 1, 4, 2),
      data: Float32Array.of(1, 2, 3, 4),
      isDiscrete: false,
      numStates: null,
    }
    const e = await makeSpatialStatistics().extract(r, nullProgress)
    expect(e.D).toBe(2)
    expect(e.data[0]).toBeCloseTo(2.5, 6)
    expect(e.data[1]).toBeCloseTo(Math.sqrt(1.25), 6)
  })
})

describe('pca', () => {
  it('captures a 1D line embedded in 5D in the first component', () => {
    const T = 20
    const D = 5
    const dir = [0.5, -0.2, 0.8, 0.1, -0.4]
    const z = new Float32Array(T * D)
    for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) z[t * D + d] = t * dir[d]
    const y = pcaProject(z, T, D, 2)
    // second component carries ~no variance
    let var1 = 0
    let var2 = 0
    for (let t = 0; t < T; t++) {
      var1 += y[t * 2] * y[t * 2]
      var2 += y[t * 2 + 1] * y[t * 2 + 1]
    }
    expect(var2 / (var1 + 1e-12)).toBeLessThan(1e-6)
    // first component is monotone in t (the line parametrization)
    const sign = Math.sign(y[2] - y[0])
    for (let t = 1; t < T; t++) {
      expect(Math.sign(y[t * 2] - y[(t - 1) * 2])).toBe(sign)
    }
  })

  it('handles T=1 and D < nComponents without blowing up', () => {
    expect(Array.from(pcaProject(Float32Array.of(1, 2), 1, 2, 2))).toEqual([0, 0])
    const y = pcaProject(Float32Array.of(0, 1, 2, 3), 4, 1, 2)
    expect(y.length).toBe(8)
    for (let t = 0; t < 4; t++) expect(y[t * 2 + 1]).toBe(0)
  })
})
