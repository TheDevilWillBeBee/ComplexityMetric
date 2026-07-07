import type { Dims } from '../types'
import type { BinaryCa1dParams } from '../cpu/binaryCa1d'
import type { BinaryCa2dParams } from '../cpu/binaryCa2d'
import type { OuterTotalisticParams } from '../cpu/outerTotalistic'
import type { LangtonParams } from '../cpu/langtonCa'
import type { CoupledLogisticParams } from '../cpu/coupledLogistic1d'
import type { GrayScottParams } from '../cpu/grayScott2d'
import { makeCoeffU32, qThreshold, seedTerm32 } from '../langtonHash'
import { makeGpuStepper, type GpuStepper } from './gpuStepper'
import binaryCa1dWgsl from './shaders/binaryCa1d.wgsl?raw'
import binaryCa2dWgsl from './shaders/binaryCa2d.wgsl?raw'
import outerTotalistic1dWgsl from './shaders/outerTotalistic1d.wgsl?raw'
import outerTotalistic2dWgsl from './shaders/outerTotalistic2d.wgsl?raw'
import langtonCa1dWgsl from './shaders/langtonCa1d.wgsl?raw'
import langtonCa2dWgsl from './shaders/langtonCa2d.wgsl?raw'
import coupledLogistic1dWgsl from './shaders/coupledLogistic1d.wgsl?raw'
import grayScott2dWgsl from './shaders/grayScott2d.wgsl?raw'

/**
 * GPU counterparts of the CPU steppers, sharing the same parameter
 * structures. Discrete systems are bit-exact vs the CPU implementations;
 * the two continuous systems agree to float32 tolerance.
 */

const uints = (
  dims: Dims,
  K: number,
  numStates: number,
  tableLen: number,
): [number, number, number, number, number, number, number, number] => [
  dims.B,
  dims.C,
  dims.H,
  dims.W,
  K,
  numStates,
  tableLen,
  0,
]

const NO_FLOATS: [number, number, number, number] = [0, 0, 0, 0]

export function makeGpuBinaryCa1d(
  device: GPUDevice,
  params: BinaryCa1dParams,
  x0: Int32Array,
  dims: Dims,
): GpuStepper {
  const tableLen = 1 << params.K
  return makeGpuStepper(
    device,
    {
      code: binaryCa1dWgsl,
      label: 'binaryCa1d',
      dims,
      uniformInts: uints(dims, params.K, 2, tableLen),
      uniformFloats: NO_FLOATS,
      storages: [Uint32Array.from(params.rule)],
      workgroup: '1d',
    },
    x0,
  )
}

export function makeGpuBinaryCa2d(
  device: GPUDevice,
  params: BinaryCa2dParams,
  x0: Int32Array,
  dims: Dims,
): GpuStepper {
  return makeGpuStepper(
    device,
    {
      code: binaryCa2dWgsl,
      label: 'binaryCa2d',
      dims,
      uniformInts: uints(dims, 3, 2, 512),
      uniformFloats: NO_FLOATS,
      storages: [Uint32Array.from(params.rule)],
      workgroup: '2d',
    },
    x0,
  )
}

function packBs(params: OuterTotalisticParams, B: number, L: number): Uint32Array {
  const bs = new Uint32Array(B * 2 * L)
  for (let b = 0; b < B; b++) {
    for (let n = 0; n < L; n++) {
      bs[b * 2 * L + n] = params.birth[b * L + n]
      bs[b * 2 * L + L + n] = params.survive[b * L + n]
    }
  }
  return bs
}

export function makeGpuOuterTotalistic1d(
  device: GPUDevice,
  params: OuterTotalisticParams,
  x0: Int32Array,
  dims: Dims,
): GpuStepper {
  const L = params.K
  return makeGpuStepper(
    device,
    {
      code: outerTotalistic1dWgsl,
      label: 'outerTotalistic1d',
      dims,
      uniformInts: uints(dims, params.K, 2, L),
      uniformFloats: NO_FLOATS,
      storages: [packBs(params, dims.B, L)],
      workgroup: '1d',
    },
    x0,
  )
}

export function makeGpuOuterTotalistic2d(
  device: GPUDevice,
  params: OuterTotalisticParams,
  x0: Int32Array,
  dims: Dims,
): GpuStepper {
  const L = params.K * params.K
  return makeGpuStepper(
    device,
    {
      code: outerTotalistic2dWgsl,
      label: 'outerTotalistic2d',
      dims,
      uniformInts: uints(dims, params.K, 2, L),
      uniformFloats: NO_FLOATS,
      storages: [packBs(params, dims.B, L)],
      workgroup: '2d',
    },
    x0,
  )
}

function langtonAux(params: LangtonParams): Uint32Array {
  const B = params.lambda.length
  const aux = new Uint32Array(B * 2)
  for (let b = 0; b < B; b++) {
    aux[b * 2] = seedTerm32(params.seed[b])
    aux[b * 2 + 1] = qThreshold(params.lambda[b]) // 2^31 fits in u32
  }
  return aux
}

export function makeGpuLangtonCa1d(
  device: GPUDevice,
  params: LangtonParams,
  x0: Int32Array,
  dims: Dims,
): GpuStepper {
  return makeGpuStepper(
    device,
    {
      code: langtonCa1dWgsl,
      label: 'langtonCa1d',
      dims,
      uniformInts: uints(dims, params.kernelSize, params.numStates, params.kernelSize),
      uniformFloats: NO_FLOATS,
      storages: [makeCoeffU32(params.kernelSize), langtonAux(params)],
      workgroup: '1d',
    },
    x0,
  )
}

export function makeGpuLangtonCa2d(
  device: GPUDevice,
  params: LangtonParams,
  x0: Int32Array,
  dims: Dims,
): GpuStepper {
  const L = params.kernelSize * params.kernelSize
  return makeGpuStepper(
    device,
    {
      code: langtonCa2dWgsl,
      label: 'langtonCa2d',
      dims,
      uniformInts: uints(dims, params.kernelSize, params.numStates, L),
      uniformFloats: NO_FLOATS,
      storages: [makeCoeffU32(L), langtonAux(params)],
      workgroup: '2d',
    },
    x0,
  )
}

export function makeGpuCoupledLogistic1d(
  device: GPUDevice,
  params: CoupledLogisticParams,
  x0: Float32Array,
  dims: Dims,
): GpuStepper {
  const B = dims.B
  const sp = new Float32Array(B * 2)
  for (let b = 0; b < B; b++) {
    sp[b * 2] = params.r[b]
    sp[b * 2 + 1] = params.eps[b]
  }
  return makeGpuStepper(
    device,
    {
      code: coupledLogistic1dWgsl,
      label: 'coupledLogistic1d',
      dims,
      uniformInts: uints(dims, 3, 0, 0),
      uniformFloats: NO_FLOATS,
      storages: [sp],
      workgroup: '1d',
    },
    x0,
  )
}

export function makeGpuGrayScott2d(
  device: GPUDevice,
  params: GrayScottParams,
  x0: Float32Array,
  dims: Dims,
  dt = 1.0,
): GpuStepper {
  const B = dims.B
  const sp = new Float32Array(B * 4)
  for (let b = 0; b < B; b++) {
    sp[b * 4] = params.Du[b]
    sp[b * 4 + 1] = params.Dv[b]
    sp[b * 4 + 2] = params.F[b]
    sp[b * 4 + 3] = params.k[b]
  }
  return makeGpuStepper(
    device,
    {
      code: grayScott2dWgsl,
      label: 'grayScott2d',
      dims,
      uniformInts: uints(dims, 3, 0, 0),
      uniformFloats: [dt, 0, 0, 0],
      storages: [sp],
      workgroup: '2d',
    },
    x0,
  )
}
