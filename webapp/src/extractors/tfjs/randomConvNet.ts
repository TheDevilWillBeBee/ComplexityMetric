import * as tf from '@tensorflow/tfjs-core'
import type { Extractor } from '../types'
import { extractChunked } from './common'
import { circularPad, gramReduce, instanceNorm } from './ops'
import { makeConv, makeLinear, type ConvWeights, type LinearWeights } from './weights'

/**
 * RandomConvNet with the app's configuration (app_dynamics.py:343-360 /
 * extractors.py:107-176): stages of [circular conv -> instance norm ->
 * tanh], channels Cin->8->8->16->16->32->32, kernels 5,5 / 7,7 / 9,9
 * (kernel_size 5 + stage*2, no pooling), gram reduction (32^2 = 1024) ->
 * Linear -> embed_dim, inputs centered by -0.5, rows L2-normalized.
 */
export interface RandomConvNetConfig {
  embedDim: number
  seed: number
  baseChannels?: number
  numStages?: number
  layersPerStage?: number
  kernelSize?: number
  kernelGrowth?: number
}

export function makeRandomConvNet(cfg: RandomConvNetConfig): Extractor {
  const base = cfg.baseChannels ?? 8
  const numStages = cfg.numStages ?? 3
  const layersPerStage = cfg.layersPerStage ?? 2
  const kernelSize = cfg.kernelSize ?? 5
  const kernelGrowth = cfg.kernelGrowth ?? 2

  interface Layer {
    conv: ConvWeights
    k: number
  }
  let layers: Layer[] | null = null
  let proj: LinearWeights | null = null
  let is1d = false

  const buildOnce = (inputChannels: number): void => {
    if (layers) return
    layers = []
    let c = inputChannels
    let out = base
    for (let stage = 0; stage < numStages; stage++) {
      let k = kernelSize + stage * kernelGrowth
      if (k % 2 === 0) k += 1
      for (let l = 0; l < layersPerStage; l++) {
        const kh = is1d ? 1 : k
        layers.push({ conv: makeConv(cfg.seed, `rcn-s${stage}-l${l}`, kh, k, c, out), k })
        c = out
      }
      out *= 2
    }
    proj = makeLinear(cfg.seed, 'rcn-proj', c * c, cfg.embedDim)
  }

  const run = (x: tf.Tensor4D): tf.Tensor2D => {
    let h = tf.sub(x, 0.5) as tf.Tensor4D
    for (const layer of layers!) {
      const pad = layer.k >> 1
      const padded = circularPad(h, is1d ? 0 : pad, pad)
      h = tf.add(tf.conv2d(padded, layer.conv.W, 1, 'valid'), layer.conv.b) as tf.Tensor4D
      h = instanceNorm(h)
      h = tf.tanh(h)
    }
    const z = gramReduce(h)
    return tf.add(tf.matMul(z, proj!.W), proj!.b) as tf.Tensor2D
  }

  return {
    name: 'RandomConvNet',
    inputType: 'continuous',
    outputType: 'continuous',
    async extract(r, progress) {
      is1d = r.spatialDim === 1
      return extractChunked(r, cfg.embedDim, buildOnce, run, progress)
    },
    dispose() {
      for (const layer of layers ?? []) {
        layer.conv.W.dispose()
        layer.conv.b.dispose()
      }
      proj?.W.dispose()
      proj?.b.dispose()
      layers = null
      proj = null
    },
  }
}
