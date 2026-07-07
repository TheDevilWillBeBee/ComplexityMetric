import { makeRng } from '../core/prng'

/**
 * PCA projection of one trajectory (T, D) -> (T, nComponents), matching
 * Embedding._reduce (embedding.py:74-81): center over time, project onto
 * the top right-singular vectors. Uses power iteration with deflation
 * (O(T*D) per iteration) instead of a full SVD; component signs are
 * arbitrary, as with any SVD.
 */
export function pcaProject(z: Float32Array, T: number, D: number, nComponents = 2): Float32Array {
  const out = new Float32Array(T * nComponents)
  if (T === 0 || D === 0) return out

  // center over time
  const zc = new Float64Array(T * D)
  const mean = new Float64Array(D)
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) mean[d] += z[t * D + d]
  for (let d = 0; d < D; d++) mean[d] /= T
  for (let t = 0; t < T; t++) for (let d = 0; d < D; d++) zc[t * D + d] = z[t * D + d] - mean[d]

  const rng = makeRng(1234)
  const comps: Float64Array[] = []
  const zv = new Float64Array(T)

  const nc = Math.min(nComponents, D, T)
  for (let c = 0; c < nc; c++) {
    let v = new Float64Array(D)
    for (let d = 0; d < D; d++) v[d] = rng.float() - 0.5
    let next = new Float64Array(D)

    for (let iter = 0; iter < 128; iter++) {
      // w = Zc^T (Zc v), staying orthogonal to found components
      for (const u of comps) {
        let dot = 0
        for (let d = 0; d < D; d++) dot += v[d] * u[d]
        for (let d = 0; d < D; d++) v[d] -= dot * u[d]
      }
      for (let t = 0; t < T; t++) {
        let s = 0
        const off = t * D
        for (let d = 0; d < D; d++) s += zc[off + d] * v[d]
        zv[t] = s
      }
      next.fill(0)
      for (let t = 0; t < T; t++) {
        const s = zv[t]
        const off = t * D
        for (let d = 0; d < D; d++) next[d] += zc[off + d] * s
      }
      let norm = 0
      for (let d = 0; d < D; d++) norm += next[d] * next[d]
      norm = Math.sqrt(norm)
      if (norm < 1e-12) break // no variance left
      let delta = 0
      for (let d = 0; d < D; d++) {
        const nv = next[d] / norm
        delta += Math.abs(nv - v[d])
        next[d] = nv
      }
      const tmp = v
      v = next
      next = tmp
      if (delta < 1e-9) break
    }
    comps.push(v)
    for (let t = 0; t < T; t++) {
      let s = 0
      const off = t * D
      for (let d = 0; d < D; d++) s += zc[off + d] * v[d]
      out[t * nComponents + c] = s
    }
  }
  return out
}
