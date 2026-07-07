import type { DimMethod } from '../config'
import { pcaProject } from './pca'

export const PROJECTION_METHODS: DimMethod[] = ['pca', 'tsne', 'umap']

export async function project(
  method: DimMethod,
  z: Float32Array,
  T: number,
  D: number,
): Promise<Float32Array> {
  switch (method) {
    case 'pca':
      return pcaProject(z, T, D, 2)
    case 'tsne':
      return (await import('./tsne')).tsneProject(z, T, D)
    case 'umap':
      return (await import('./umap')).umapProject(z, T, D)
    default:
      throw new Error(`unknown projection method: ${method}`)
  }
}
