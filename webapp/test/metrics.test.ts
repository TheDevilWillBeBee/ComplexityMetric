import { describe, expect, it } from 'vitest'
import type { EmbeddingData } from '../src/extractors/types'
import { makeRng } from '../src/core/prng'
import { makeEntropy } from '../src/metrics/entropy'
import { makeFutureStateMI } from '../src/metrics/mi'
import { makeCompressedRatio } from '../src/metrics/compressedRatio'
import { makeDensityTransientTime } from '../src/metrics/transient'
import { makeLinearRidgeTimeRegression } from '../src/metrics/ridge'
import { makeKnnTimeRegression } from '../src/metrics/knn'
import { makeOpenEndedness } from '../src/metrics/openEndedness'
import { metricWindows } from '../src/metrics/windows'
import { embeddingToContinuous, embeddingToDiscrete } from '../src/metrics/convert'
import { cholSolve, ridgeFit, roundHalfEven, splitIndices } from '../src/metrics/linalg'

function disc(values: number[], B: number, T: number, D: number, K = 2): EmbeddingData {
  return { data: Int32Array.from(values), B, T, D, isDiscrete: true, numStates: K }
}

function cont(values: number[], B: number, T: number, D: number): EmbeddingData {
  return { data: Float32Array.from(values), B, T, D, isDiscrete: false, numStates: null }
}

describe('Entropy', () => {
  it('is 0 for a constant embedding', async () => {
    const e = disc(new Array(2 * 10 * 4).fill(1), 2, 10, 4)
    for (const mode of ['time', 'space', 'both'] as const) {
      const scores = await makeEntropy(mode).compute(e)
      expect(scores[0]).toBeCloseTo(0, 5)
      expect(scores[1]).toBeCloseTo(0, 5)
    }
  })

  it('is ~1 for iid uniform binary data', async () => {
    const rng = makeRng(0)
    const T = 200
    const D = 50
    const values = Array.from({ length: T * D }, () => rng.int(2))
    const scores = await makeEntropy('both').compute(disc(values, 1, T, D))
    expect(scores[0]).toBeGreaterThan(0.99)
  })

  it('distinguishes time vs space structure', async () => {
    // x[t,d] = t % 2: every feature alternates over time; rows are constant
    const T = 8
    const D = 4
    const values: number[] = []
    for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) values.push(t % 2)
    const e = disc(values, 1, T, D)
    expect((await makeEntropy('time').compute(e))[0]).toBeCloseTo(1, 5)
    expect((await makeEntropy('space').compute(e))[0]).toBeCloseTo(0, 5)
  })
})

describe('FutureStateMutualInformation', () => {
  it('is ~1 when x[t+1] deterministically equals x[t] with balanced states', async () => {
    // alternating per feature in time -> next state fully determined
    const T = 64
    const D = 8
    const values: number[] = []
    for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) values.push((t + d) % 2)
    const scores = await makeFutureStateMI('time').compute(disc(values, 1, T, D))
    expect(scores[0]).toBeGreaterThan(0.99)
  })

  it('is ~0 for independent uniform noise', async () => {
    const rng = makeRng(1)
    const T = 500
    const D = 20
    const values = Array.from({ length: T * D }, () => rng.int(2))
    const scores = await makeFutureStateMI('time').compute(disc(values, 1, T, D))
    expect(scores[0]).toBeLessThan(0.05)
  })
})

describe('CompressedRatio', () => {
  it('is near 0 for constant data and >= ~1 for uniform noise', async () => {
    const T = 128
    const D = 32
    const constant = await makeCompressedRatio().compute(disc(new Array(T * D).fill(1), 1, T, D))
    expect(constant[0]).toBeLessThan(0.15)

    const rng = makeRng(2)
    const noise = Array.from({ length: T * D }, () => rng.int(2))
    const random = await makeCompressedRatio().compute(disc(noise, 1, T, D))
    expect(random[0]).toBeGreaterThan(0.9)
  })
})

describe('DensityTransientTime', () => {
  it('returns 1 when T < confirmation window', async () => {
    const scores = await makeDensityTransientTime().compute(disc([0, 1, 0, 1], 1, 2, 2))
    expect(scores[0]).toBe(1)
  })

  it('finds the settling time of a step-like density', async () => {
    const T = 400
    const D = 10
    const values: number[] = []
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < D; d++) values.push(t < 100 ? 1 : d < 5 ? 1 : 0)
    }
    // density: 1.0 for t<100, then 0.5; tail(last 256) all 0.5 -> tol 0.05
    const scores = await makeDensityTransientTime().compute(disc(values, 1, T, D))
    expect(scores[0]).toBeCloseTo(100 / 400, 6)
  })
})

describe('time regressions', () => {
  it('ridge R^2 is ~1 for features linear in t', async () => {
    const T = 64
    const D = 3
    const values: number[] = []
    for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) values.push(t * (d + 1) * 0.01)
    const scores = await makeLinearRidgeTimeRegression().compute(cont(values, 1, T, D))
    expect(scores[0]).toBeGreaterThan(0.999)
  })

  it('ridge returns 0 below the T=4 threshold; KNN below T=6', async () => {
    const values = [0, 1, 2]
    expect((await makeLinearRidgeTimeRegression().compute(cont(values, 1, 3, 1)))[0]).toBe(0)
    const v5 = [0, 1, 2, 3, 4]
    expect((await makeKnnTimeRegression().compute(cont(v5, 1, 5, 1)))[0]).toBe(0)
  })

  it('KNN scores high on a smooth 1D trajectory', async () => {
    const T = 100
    const values = Array.from({ length: T }, (_, t) => t / T)
    const scores = await makeKnnTimeRegression().compute(cont(values, 1, T, 1))
    expect(scores[0]).toBeGreaterThan(0.9)
  })

  it('dual and primal ridge solutions agree', () => {
    const rng = makeRng(3)
    const n = 10
    const d = 6
    const X = new Float64Array(n * d)
    const y = new Float64Array(n)
    for (let i = 0; i < n * d; i++) X[i] = rng.float() - 0.5
    for (let i = 0; i < n; i++) y[i] = rng.float()
    const primal = ridgeFit(X, y, n, d, 1e-3) // d <= n -> primal
    // force dual by transposing the roles: call with padded features d' > n
    const dPad = n + 3
    const Xp = new Float64Array(n * dPad)
    for (let i = 0; i < n; i++) for (let a = 0; a < d; a++) Xp[i * dPad + a] = X[i * d + a]
    const dual = ridgeFit(Xp, y, n, dPad, 1e-3)
    // padded dims are all-zero features; dual weights there must be 0 and
    // shared dims must match the primal solution of the same problem
    const primalPadded = (() => {
      // solve primal on padded system via normal equations for reference
      const A = new Float64Array(dPad * dPad)
      for (let i = 0; i < n; i++)
        for (let a = 0; a < dPad; a++)
          for (let b = 0; b < dPad; b++) A[a * dPad + b] += Xp[i * dPad + a] * Xp[i * dPad + b]
      for (let a = 0; a < dPad; a++) A[a * dPad + a] += 1e-3
      const rhs = new Float64Array(dPad)
      for (let i = 0; i < n; i++) for (let a = 0; a < dPad; a++) rhs[a] += Xp[i * dPad + a] * y[i]
      return cholSolve(A, rhs, dPad)
    })()
    for (let a = 0; a < dPad; a++) expect(dual[a]).toBeCloseTo(primalPadded[a], 8)
    for (let a = 0; a < d; a++) expect(dual[a]).toBeCloseTo(primal[a], 8)
  })
})

describe('OpenEndedness', () => {
  it('scores 1 for a single frame and 0 for repeated frames', async () => {
    expect((await makeOpenEndedness().compute(cont([1, 2], 1, 1, 2)))[0]).toBe(1)
    const repeated = [1, 2, 1, 2, 1, 2]
    expect((await makeOpenEndedness().compute(cont(repeated, 1, 3, 2)))[0]).toBeCloseTo(0, 6)
  })

  it('scores 1 for mutually orthogonal frames', async () => {
    const values = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    expect((await makeOpenEndedness().compute(cont(values, 1, 3, 3)))[0]).toBe(1)
  })
})

describe('metricWindows', () => {
  it('matches the Python metric_windows semantics', () => {
    expect(metricWindows(10, 4, 2)).toEqual([
      [0, 4],
      [2, 6],
      [4, 8],
      [6, 10],
    ])
    expect(metricWindows(5, 0, 1)).toEqual([[0, 5]])
    expect(metricWindows(8, 8, 1)).toEqual([[0, 8]])
    expect(metricWindows(8, 12, 3)).toEqual([[0, 8]]) // size clamped to T
    expect(metricWindows(6, 2, 5)).toEqual([[0, 2]]) // next start (5) would overshoot
    expect(metricWindows(0, 4, 1)).toEqual([])
  })
})

describe('metric input conversions', () => {
  it('one-hot expands discrete embeddings to (B,T,D*K)', () => {
    const e = disc([0, 2, 1, 0], 1, 2, 2, 3)
    const c = embeddingToContinuous(e)
    expect(c.D).toBe(6)
    expect(Array.from(c.data)).toEqual([1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0])
  })

  it('bins continuous embeddings to 2 states', () => {
    const e = cont([0, 0.4, 0.6, 1], 1, 2, 2)
    const d = embeddingToDiscrete(e)
    expect(Array.from(d.data)).toEqual([0, 0, 1, 1])
    expect(d.numStates).toBe(2)
  })
})

describe('split helpers', () => {
  it("roundHalfEven matches Python's round()", () => {
    expect(roundHalfEven(2.5)).toBe(2)
    expect(roundHalfEven(3.5)).toBe(4)
    expect(roundHalfEven(2.4)).toBe(2)
    expect(roundHalfEven(2.6)).toBe(3)
  })

  it('splitIndices always yields non-empty train and test for n > 1', () => {
    for (const n of [2, 3, 7, 100]) {
      const { train, test } = splitIndices(n, 0.5, 42)
      expect(train.length).toBeGreaterThan(0)
      expect(test.length).toBeGreaterThan(0)
      expect(train.length + test.length).toBe(n)
      const all = [...train, ...test].sort((a, b) => a - b)
      expect(all).toEqual(Array.from({ length: n }, (_, i) => i))
    }
  })
})
