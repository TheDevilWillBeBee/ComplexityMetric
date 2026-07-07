import type { Rng } from '../core/prng'
import { makeRng } from '../core/prng'

/**
 * Numeric helpers for the time-regression metrics (Complexity/utils.py).
 * All solves are done in float64.
 */

/** Standardize over time (dim -2): (z - mean) / (sample std + 1e-8). */
export function standardize(z: Float32Array, T: number, D: number): Float64Array {
  const out = new Float64Array(T * D)
  for (let d = 0; d < D; d++) {
    let sum = 0
    for (let t = 0; t < T; t++) sum += z[t * D + d]
    const mean = sum / T
    let varSum = 0
    for (let t = 0; t < T; t++) {
      const dev = z[t * D + d] - mean
      varSum += dev * dev
    }
    // torch.std default: unbiased (N-1); NaN-free guard for T=1
    const std = T > 1 ? Math.sqrt(varSum / (T - 1)) : 0
    const inv = 1 / (std + 1e-8)
    for (let t = 0; t < T; t++) out[t * D + d] = (z[t * D + d] - mean) * inv
  }
  return out
}

/** Python round() semantics (banker's rounding) for split sizes. */
export function roundHalfEven(x: number): number {
  const floor = Math.floor(x)
  const diff = x - floor
  if (diff > 0.5) return floor + 1
  if (diff < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

/**
 * Seeded train/test split (utils.split_indices): random permutation, then
 * n_train = clamp(round(frac*n), 1, n-1). Uses our PRNG, not torch's —
 * split membership differs from Python, split SIZES match.
 */
export function splitIndices(
  n: number,
  trainFrac: number,
  seed: number,
): { train: Int32Array; test: Int32Array } {
  if (n <= 1) {
    const train = new Int32Array(n)
    for (let i = 0; i < n; i++) train[i] = i
    return { train, test: new Int32Array(0) }
  }
  const rng: Rng = makeRng(seed >>> 0)
  const perm = new Int32Array(n)
  for (let i = 0; i < n; i++) perm[i] = i
  for (let i = n - 1; i > 0; i--) {
    const j = rng.int(i + 1)
    const tmp = perm[i]
    perm[i] = perm[j]
    perm[j] = tmp
  }
  const nTrain = Math.min(Math.max(roundHalfEven(trainFrac * n), 1), n - 1)
  return { train: perm.slice(0, nTrain), test: perm.slice(nTrain) }
}

/** R^2 clamped to [0,1] vs a baseline prediction (utils.r2). */
export function r2(y: Float64Array, pred: Float64Array, baseline: number): number {
  let msePred = 0
  let mseBase = 0
  for (let i = 0; i < y.length; i++) {
    msePred += (y[i] - pred[i]) ** 2
    mseBase += (y[i] - baseline) ** 2
  }
  msePred /= y.length
  mseBase = mseBase / y.length + 1e-8
  const score = 1 - msePred / mseBase
  return score < 0 ? 0 : score > 1 ? 1 : score
}

/** Solve A x = b for symmetric positive-definite A (in place Cholesky). */
export function cholSolve(A: Float64Array, b: Float64Array, n: number): Float64Array {
  const L = Float64Array.from(A)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = L[i * n + j]
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k]
      if (i === j) {
        if (s <= 0) throw new Error('cholSolve: matrix not positive definite')
        L[i * n + i] = Math.sqrt(s)
      } else {
        L[i * n + j] = s / L[j * n + j]
      }
    }
  }
  const x = Float64Array.from(b)
  for (let i = 0; i < n; i++) {
    let s = x[i]
    for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]
    x[i] = s / L[i * n + i]
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i]
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]
    x[i] = s / L[i * n + i]
  }
  return x
}

/**
 * Ridge regression with bias column: min ||Xw - y||^2 + lambda ||w||^2 where
 * X already includes the bias column. Solves the primal (D x D) normal
 * equations when D <= n, else the exact dual X^T (XX^T + lambda I)^{-1} y.
 */
export function ridgeFit(
  X: Float64Array,
  y: Float64Array,
  n: number,
  d: number,
  lambda: number,
): Float64Array {
  if (d <= n) {
    const A = new Float64Array(d * d)
    for (let i = 0; i < n; i++) {
      const off = i * d
      for (let a = 0; a < d; a++) {
        const xa = X[off + a]
        if (xa === 0) continue
        for (let b = a; b < d; b++) A[a * d + b] += xa * X[off + b]
      }
    }
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < a; b++) A[a * d + b] = A[b * d + a]
      A[a * d + a] += lambda
    }
    const rhs = new Float64Array(d)
    for (let i = 0; i < n; i++) {
      const off = i * d
      for (let a = 0; a < d; a++) rhs[a] += X[off + a] * y[i]
    }
    return cholSolve(A, rhs, d)
  }
  // dual: w = X^T (X X^T + lambda I_n)^{-1} y
  const G = new Float64Array(n * n)
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let s = 0
      const oi = i * d
      const oj = j * d
      for (let a = 0; a < d; a++) s += X[oi + a] * X[oj + a]
      G[i * n + j] = s
      G[j * n + i] = s
    }
    G[i * n + i] += lambda
  }
  const alpha = cholSolve(G, y, n)
  const w = new Float64Array(d)
  for (let i = 0; i < n; i++) {
    const off = i * d
    const ai = alpha[i]
    for (let a = 0; a < d; a++) w[a] += X[off + a] * ai
  }
  return w
}
