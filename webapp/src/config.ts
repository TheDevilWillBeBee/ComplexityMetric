/**
 * Run configuration — mirrors the dict returned by sidebar() in
 * app_dynamics.py (same keys, same defaults, camelCased).
 */

export type SeedMode = 'noise' | 'single' | 'zeros'
export type DimMethod = 'pca' | 'tsne' | 'umap'

export interface AppConfig {
  systemName: string
  batchSize: number
  /** null for 1D systems. */
  H: number | null
  W: number
  steps: number
  every: number
  skip: number

  // per-system parameters (flat, like the Python cfg dict)
  kernelSize: number
  randomRule: boolean
  ruleInt: number
  desc: string
  /** BinaryCA2D rule density. */
  p: number
  numStates: number
  lambda: number
  ruleSeed: number
  r: number
  eps: number
  Du: number
  Dv: number
  F: number
  k: number

  seedMode: SeedMode
  seedP: number

  extractors: string[]
  dimMethod: DimMethod
  embedDim: number
  extractorSeed: number

  metrics: string[]
  metricChunksEnabled: boolean
  metricChunkSize: number
  metricStride: number

  /** "Torch seed" in the Python app. */
  appSeed: number
}

export function createDefaultConfig(): AppConfig {
  return {
    systemName: 'Binary CA 1D',
    batchSize: 4,
    H: null,
    W: 128,
    steps: 128,
    every: 1,
    skip: 0,
    kernelSize: 3,
    randomRule: false,
    ruleInt: 30,
    desc: 'B1/S23',
    p: 0.5,
    numStates: 2,
    lambda: 0.5,
    ruleSeed: 0,
    r: 3.8,
    eps: 0.25,
    Du: 0.16,
    Dv: 0.08,
    F: 0.035,
    k: 0.06,
    seedMode: 'noise',
    seedP: 0.5,
    extractors: ['DiscreteFlatten'],
    dimMethod: 'pca',
    embedDim: 64,
    extractorSeed: 0,
    metrics: ['Linear ridge time regression', 'Entropy both', 'CompressedRatio'],
    metricChunksEnabled: false,
    metricChunkSize: 0,
    metricStride: 1,
    appSeed: 0,
  }
}
