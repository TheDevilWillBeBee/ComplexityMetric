import * as tf from '@tensorflow/tfjs-core'
import type { Extractor } from '../types'
import { extractChunked } from './common'
import { circularPad, gramReduce, l2Normalize } from './ops'
import { makeConv, makeLinear, type ConvWeights, type LinearWeights } from './weights'

/**
 * RandomVGG with the app's configuration (app_dynamics.py:361-362 /
 * extractors.py:179-254): blocks of 3x3 circular convs + ReLU with
 * maxpool(2) after each block, channels (8,16,32), convs per block
 * (2,2,3). Texture taps at the ReLU output of the FIRST conv of each
 * block (conv indices 0,2,4): gram -> Linear -> embed_dim -> L2 norm;
 * taps concatenated -> final Linear -> embed_dim, rows L2-normalized.
 */
export interface RandomVggConfig {
  embedDim: number
  seed: number
  channels?: number[]
  convsPerBlock?: number[]
}

export function makeRandomVgg(cfg: RandomVggConfig): Extractor {
  const channels = cfg.channels ?? [8, 16, 32]
  const convsPerBlock = cfg.convsPerBlock ?? [2, 2, 3]
  // taps: first conv of each block (extractors.py _vgg_texture_taps)
  const taps: number[] = []
  {
    let start = 0
    for (const depth of convsPerBlock) {
      taps.push(start)
      start += depth
    }
  }

  interface ConvLayer {
    conv: ConvWeights
    tapIndex: number | null // index into tapProjs when tapped
    poolAfter: boolean
  }
  let convLayers: ConvLayer[] | null = null
  let tapProjs: LinearWeights[] | null = null
  let proj: LinearWeights | null = null
  let is1d = false

  const buildOnce = (inputChannels: number): void => {
    if (convLayers) return
    convLayers = []
    tapProjs = []
    let c = inputChannels
    let convIdx = 0
    for (let block = 0; block < channels.length; block++) {
      const out = channels[block]
      for (let l = 0; l < convsPerBlock[block]; l++) {
        const isTap = taps.includes(convIdx)
        let tapIndex: number | null = null
        if (isTap) {
          tapIndex = tapProjs.length
          tapProjs.push(makeLinear(cfg.seed, `rvgg-tap${tapIndex}`, out * out, cfg.embedDim))
        }
        convLayers.push({
          conv: makeConv(cfg.seed, `rvgg-b${block}-c${l}`, is1d ? 1 : 3, 3, c, out),
          tapIndex,
          poolAfter: l === convsPerBlock[block] - 1,
        })
        convIdx += 1
        c = out
      }
    }
    proj = makeLinear(cfg.seed, 'rvgg-proj', taps.length * cfg.embedDim, cfg.embedDim)
  }

  const run = (x: tf.Tensor4D): tf.Tensor2D => {
    let h = tf.sub(x, 0.5) as tf.Tensor4D
    const zs: tf.Tensor2D[] = []
    for (const layer of convLayers!) {
      const padded = circularPad(h, is1d ? 0 : 1, 1)
      h = tf.add(tf.conv2d(padded, layer.conv.W, 1, 'valid'), layer.conv.b) as tf.Tensor4D
      h = tf.relu(h)
      if (layer.tapIndex !== null) {
        const tapProj = tapProjs![layer.tapIndex]
        const z = tf.add(tf.matMul(gramReduce(h), tapProj.W), tapProj.b) as tf.Tensor2D
        zs.push(l2Normalize(z))
      }
      if (layer.poolAfter) {
        const pool: [number, number] = is1d ? [1, 2] : [2, 2]
        h = tf.maxPool(h, pool, pool, 'valid')
      }
    }
    const z = tf.concat(zs, 1) as tf.Tensor2D
    return tf.add(tf.matMul(z, proj!.W), proj!.b) as tf.Tensor2D
  }

  return {
    name: 'RandomVGG',
    inputType: 'continuous',
    outputType: 'continuous',
    async extract(r, progress) {
      is1d = r.spatialDim === 1
      return extractChunked(r, cfg.embedDim, buildOnce, run, progress)
    },
    dispose() {
      for (const layer of convLayers ?? []) {
        layer.conv.W.dispose()
        layer.conv.b.dispose()
      }
      for (const tap of tapProjs ?? []) {
        tap.W.dispose()
        tap.b.dispose()
      }
      proj?.W.dispose()
      proj?.b.dispose()
      convLayers = null
      tapProjs = null
      proj = null
    },
  }
}
