import TSNE from 'tsne-js'
import { pcaProject } from './pca'

/**
 * t-SNE projection of one trajectory. Perplexity is clamped like the
 * Python code (embedding.py:89): min(30, max(1, (T-1)//3), T-1).
 * Note: tsne-js is not seedable, so layouts vary between runs (flagged in
 * the UI); tiny T falls back to PCA.
 */
export function tsneProject(z: Float32Array, T: number, D: number): Float32Array {
  if (T < 4) return pcaProject(z, T, D, 2)
  const perplexity = Math.min(30, Math.max(1, Math.floor((T - 1) / 3)), T - 1)
  const data: number[][] = []
  for (let t = 0; t < T; t++) {
    const row = new Array<number>(D)
    for (let d = 0; d < D; d++) row[d] = z[t * D + d]
    data.push(row)
  }
  const model = new TSNE({ dim: 2, perplexity, nIter: 500 })
  model.init({ data, type: 'dense' })
  model.run()
  const out = model.getOutputScaled()
  const result = new Float32Array(T * 2)
  for (let t = 0; t < T; t++) {
    result[t * 2] = out[t][0]
    result[t * 2 + 1] = out[t][1]
  }
  return result
}
