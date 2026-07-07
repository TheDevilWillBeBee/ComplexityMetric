import { UMAP } from 'umap-js'
import { makeRng } from '../core/prng'
import { pcaProject } from './pca'

/**
 * UMAP projection of one trajectory, seeded through our PRNG for
 * reproducibility. nNeighbors is clamped below the point count
 * (umap-js throws otherwise); tiny T falls back to PCA.
 */
export function umapProject(z: Float32Array, T: number, D: number): Float32Array {
  if (T < 5) return pcaProject(z, T, D, 2)
  const rng = makeRng(0)
  const data: number[][] = []
  for (let t = 0; t < T; t++) {
    const row = new Array<number>(D)
    for (let d = 0; d < D; d++) row[d] = z[t * D + d]
    data.push(row)
  }
  const umap = new UMAP({
    nComponents: 2,
    nNeighbors: Math.min(15, T - 2),
    random: () => rng.float(),
  })
  const out = umap.fit(data)
  const result = new Float32Array(T * 2)
  for (let t = 0; t < T; t++) {
    result[t * 2] = out[t][0]
    result[t * 2 + 1] = out[t][1]
  }
  return result
}
