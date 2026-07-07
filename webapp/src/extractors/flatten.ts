import type { Extractor } from './types'
import { rolloutToContinuous, rolloutToDiscrete } from '../systems/convert'

/**
 * DiscreteFlatten (extractors.py:70-78): to_discrete() then flatten each
 * frame. For already-discrete rollouts this keeps the K states; continuous
 * rollouts are binned to 2 states over [0,1].
 */
export function makeDiscreteFlatten(): Extractor {
  return {
    name: 'DiscreteFlatten',
    inputType: 'discrete',
    outputType: 'discrete',
    async extract(r) {
      const rd = rolloutToDiscrete(r)
      // (B,T,C,H,W) row-major is already (B,T,D) with D = C*H*W
      return {
        data: rd.data as Int32Array,
        B: rd.B,
        T: rd.T,
        D: rd.C * rd.H * rd.W,
        isDiscrete: true,
        numStates: rd.numStates,
      }
    },
    dispose() {},
  }
}

/**
 * ContinuousFlatten (extractors.py:81-89): to_continuous() (one-hot for
 * discrete rollouts) then flatten each frame.
 */
export function makeContinuousFlatten(): Extractor {
  return {
    name: 'ContinuousFlatten',
    inputType: 'continuous',
    outputType: 'continuous',
    async extract(r) {
      const rc = rolloutToContinuous(r)
      return {
        data: rc.data as Float32Array,
        B: rc.B,
        T: rc.T,
        D: rc.C * rc.H * rc.W,
        isDiscrete: false,
        numStates: null,
      }
    },
    dispose() {},
  }
}
