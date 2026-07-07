import type { AppConfig } from '../config'
import type { Extractor } from './types'
import { makeContinuousFlatten, makeDiscreteFlatten } from './flatten'
import { makeSpatialStatistics } from './spatialStats'

export interface ExtractorSpecUI {
  name: string
  dims: ReadonlyArray<1 | 2>
  inputType: 'discrete' | 'continuous'
  outputType: 'discrete' | 'continuous'
  note: string
  /** Async so the tfjs-backed extractors can load their chunk on demand. */
  make(cfg: AppConfig): Promise<Extractor>
}

export const EXTRACTORS: Record<string, ExtractorSpecUI> = {
  DiscreteFlatten: {
    name: 'DiscreteFlatten',
    dims: [1, 2],
    inputType: 'discrete',
    outputType: 'discrete',
    note: 'Discrete in, discrete out. Keeps spatial states as raw features.',
    make: async () => makeDiscreteFlatten(),
  },
  ContinuousFlatten: {
    name: 'ContinuousFlatten',
    dims: [1, 2],
    inputType: 'continuous',
    outputType: 'continuous',
    note: 'Continuous in, continuous out. Flattens raw continuous frames.',
    make: async () => makeContinuousFlatten(),
  },
  SpatialStatistics: {
    name: 'SpatialStatistics',
    dims: [1, 2],
    inputType: 'continuous',
    outputType: 'continuous',
    note: 'Continuous in, continuous out. Per-channel mean/std over spatial axes.',
    make: async () => makeSpatialStatistics(),
  },
  RandomConvNet: {
    name: 'RandomConvNet',
    dims: [1, 2],
    inputType: 'continuous',
    outputType: 'continuous',
    note: 'Continuous in, continuous out. Stride-1 random convolutions, growing kernels, gram reduction.',
    make: async (cfg) =>
      (await import('./tfjs/randomConvNet')).makeRandomConvNet({
        embedDim: cfg.embedDim,
        seed: cfg.extractorSeed,
      }),
  },
  RandomVGG: {
    name: 'RandomVGG',
    dims: [1, 2],
    inputType: 'continuous',
    outputType: 'continuous',
    note: 'Continuous in, continuous out. VGG16 block structure with random weights.',
    make: async (cfg) =>
      (await import('./tfjs/randomVgg')).makeRandomVgg({
        embedDim: cfg.embedDim,
        seed: cfg.extractorSeed,
      }),
  },
}

export function availableExtractors(spatialDim: 1 | 2): string[] {
  return Object.values(EXTRACTORS)
    .filter((e) => e.dims.includes(spatialDim))
    .map((e) => e.name)
}

/** Default selection matching the Python app (DiscreteFlatten / RandomConvNet). */
export function defaultExtractors(isDiscrete: boolean): string[] {
  const preferred = isDiscrete ? 'DiscreteFlatten' : 'RandomConvNet'
  if (EXTRACTORS[preferred]) return [preferred]
  return [isDiscrete ? 'DiscreteFlatten' : 'ContinuousFlatten']
}
