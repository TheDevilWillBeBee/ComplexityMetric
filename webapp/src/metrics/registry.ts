import type { Metric } from './types'
import { makeEntropy } from './entropy'
import { makeFutureStateMI } from './mi'
import { makeCompressedRatio, makeEntropyMinusCompressedRatio } from './compressedRatio'
import { makeDensityTransientTime } from './transient'
import { makeLinearRidgeTimeRegression } from './ridge'
import { makeKnnTimeRegression } from './knn'
import { makeOpenEndedness } from './openEndedness'

/** Same names and factories as METRICS in app_dynamics.py:88-100. */
export const METRICS: Record<string, { inputType: 'discrete' | 'continuous'; make: () => Metric }> = {
  'Linear ridge time regression': { inputType: 'continuous', make: makeLinearRidgeTimeRegression },
  'KNN time regression': { inputType: 'continuous', make: makeKnnTimeRegression },
  'Open-endedness': { inputType: 'continuous', make: makeOpenEndedness },
  'Entropy time': { inputType: 'discrete', make: () => makeEntropy('time') },
  'Entropy space': { inputType: 'discrete', make: () => makeEntropy('space') },
  'Entropy both': { inputType: 'discrete', make: () => makeEntropy('both') },
  CompressedRatio: { inputType: 'discrete', make: makeCompressedRatio },
  'Entropy - CompressedRatio': { inputType: 'discrete', make: makeEntropyMinusCompressedRatio },
  'Density transient time': { inputType: 'discrete', make: makeDensityTransientTime },
  'Future mutual information time': { inputType: 'discrete', make: () => makeFutureStateMI('time') },
  'Future mutual information space': { inputType: 'discrete', make: () => makeFutureStateMI('space') },
}

export const METRIC_NAMES = Object.keys(METRICS)
