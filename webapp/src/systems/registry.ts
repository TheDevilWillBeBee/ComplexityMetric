import type { AppConfig, SeedMode } from '../config'
import type { Rng } from '../core/prng'
import { forkRng } from '../core/prng'
import type { Dims, Stepper } from './types'
import type { RolloutMeta } from './rollout'
import { discreteSeed } from './cpu/seeds'
import { expandRule, makeBinaryCa1dStepper, sampleBinaryCa1dRule } from './cpu/binaryCa1d'
import { makeBinaryCa2dStepper, sampleBinaryCa2dRule } from './cpu/binaryCa2d'
import {
  expandTable,
  makeOuterTotalistic1dStepper,
  makeOuterTotalistic2dStepper,
} from './cpu/outerTotalistic'
import { makeLangtonCa1dStepper, makeLangtonCa2dStepper } from './cpu/langtonCa'
import { coupledLogisticSeed, makeCoupledLogistic1dStepper } from './cpu/coupledLogistic1d'
import { grayScottSeed, makeGrayScott2dStepper } from './cpu/grayScott2d'
import { bsTables, parseBs, randomBsTables } from './rules'
import {
  makeGpuBinaryCa1d,
  makeGpuBinaryCa2d,
  makeGpuCoupledLogistic1d,
  makeGpuGrayScott2d,
  makeGpuLangtonCa1d,
  makeGpuLangtonCa2d,
  makeGpuOuterTotalistic1d,
  makeGpuOuterTotalistic2d,
} from './gpu/steppers'
import {
  binaryToRgb,
  continuousToRgb,
  grayScottToRgb,
  makeLangtonToRgb,
} from '../ui/render'

/** Declarative schema for the "System Parameters" sidebar section. */
export type ParamControl =
  | { kind: 'number'; key: keyof AppConfig; label: string; min: number; max: number; step?: number }
  | { kind: 'slider'; key: keyof AppConfig; label: string; min: number; max: number; step: number }
  | { kind: 'checkbox'; key: keyof AppConfig; label: string }
  | { kind: 'text'; key: keyof AppConfig; label: string }

export interface SystemSpec {
  name: string
  spatialDim: 1 | 2
  isDiscrete: boolean
  channels: number
  defaultSize: { H: number | null; W: number }
  defaultSteps: { steps: number; every: number; skip: number }
  note: string
  paramControls: ParamControl[]
  /** Defaults applied to cfg when this system is selected. */
  paramDefaults: Partial<AppConfig>
  seedModes: SeedMode[]
  build(cfg: AppConfig, device: GPUDevice | null): BuiltSystem
}

export interface BuiltSystem {
  dims: Dims
  stepper: Stepper
  meta: RolloutMeta
}

function dims1d(cfg: AppConfig, C = 1): Dims {
  return { B: cfg.batchSize, C, H: 1, W: cfg.W }
}

function dims2d(cfg: AppConfig, C = 1): Dims {
  return { B: cfg.batchSize, C, H: cfg.H ?? 64, W: cfg.W }
}

function rngs(cfg: AppConfig): { params: Rng; init: Rng } {
  return { params: forkRng(cfg.appSeed, 'params'), init: forkRng(cfg.appSeed, 'init') }
}

const KERNEL_1D: ParamControl[] = [
  { kind: 'number', key: 'kernelSize', label: 'Kernel size', min: 3, max: 9, step: 2 },
]

export const SYSTEMS: Record<string, SystemSpec> = {
  'Binary CA 1D': {
    name: 'Binary CA 1D',
    spatialDim: 1,
    isDiscrete: true,
    channels: 1,
    defaultSize: { H: null, W: 128 },
    defaultSteps: { steps: 128, every: 1, skip: 0 },
    note: 'Elementary binary cellular automata.',
    paramControls: [
      ...KERNEL_1D,
      { kind: 'checkbox', key: 'randomRule', label: 'Random rule' },
      { kind: 'number', key: 'ruleInt', label: 'Rule', min: 0, max: 255 },
    ],
    paramDefaults: { kernelSize: 3, randomRule: false, ruleInt: 30 },
    seedModes: ['noise', 'single', 'zeros'],
    build(cfg, device) {
      const { params, init } = rngs(cfg)
      const dims = dims1d(cfg)
      const rule = sampleBinaryCa1dRule(cfg.kernelSize, cfg.randomRule ? null : cfg.ruleInt, params)
      const p = { K: cfg.kernelSize, rule: expandRule(rule, dims.B) }
      const x0 = discreteSeed(dims, cfg.seedMode, cfg.seedP, init)
      const stepper = device
        ? makeGpuBinaryCa1d(device, p, x0, dims)
        : makeBinaryCa1dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: { spatialDim: 1, isDiscrete: true, numStates: 2, systemName: 'BinaryCA1D', toRgb: binaryToRgb },
      }
    },
  },

  'Outer-totalistic CA 1D': {
    name: 'Outer-totalistic CA 1D',
    spatialDim: 1,
    isDiscrete: true,
    channels: 1,
    defaultSize: { H: null, W: 128 },
    defaultSteps: { steps: 128, every: 1, skip: 0 },
    note: 'Binary birth/survival rules on a 1D neighborhood.',
    paramControls: [
      { kind: 'number', key: 'kernelSize', label: 'Kernel size', min: 3, max: 11, step: 2 },
      { kind: 'checkbox', key: 'randomRule', label: 'Random rule' },
      { kind: 'text', key: 'desc', label: 'Rule' },
    ],
    paramDefaults: { kernelSize: 5, randomRule: false, desc: 'B1/S23' },
    seedModes: ['noise', 'single', 'zeros'],
    build(cfg, device) {
      const { params, init } = rngs(cfg)
      const dims = dims1d(cfg)
      const L = cfg.kernelSize
      const tables = cfg.randomRule
        ? randomBsTables(L, 0.5, 0.5, () => params.float())
        : bsTables(parseBs(cfg.desc), L)
      const p = { K: cfg.kernelSize, birth: expandTable(tables.birth, dims.B), survive: expandTable(tables.survive, dims.B) }
      const x0 = discreteSeed(dims, cfg.seedMode, cfg.seedP, init)
      const stepper = device
        ? makeGpuOuterTotalistic1d(device, p, x0, dims)
        : makeOuterTotalistic1dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: { spatialDim: 1, isDiscrete: true, numStates: 2, systemName: 'OuterTotalisticCA1D', toRgb: binaryToRgb },
      }
    },
  },

  'Binary CA 2D': {
    name: 'Binary CA 2D',
    spatialDim: 2,
    isDiscrete: true,
    channels: 1,
    defaultSize: { H: 64, W: 64 },
    defaultSteps: { steps: 96, every: 2, skip: 0 },
    note: 'Random 3x3 binary rule table.',
    paramControls: [{ kind: 'slider', key: 'p', label: 'Rule density', min: 0, max: 1, step: 0.01 }],
    paramDefaults: { p: 0.5 },
    seedModes: ['noise', 'single', 'zeros'],
    build(cfg, device) {
      const { params, init } = rngs(cfg)
      const dims = dims2d(cfg)
      const rule = sampleBinaryCa2dRule(params, cfg.p)
      const expanded = new Uint8Array(dims.B * 512)
      for (let b = 0; b < dims.B; b++) expanded.set(rule, b * 512)
      const p = { rule: expanded }
      const x0 = discreteSeed(dims, cfg.seedMode, cfg.seedP, init)
      const stepper = device
        ? makeGpuBinaryCa2d(device, p, x0, dims)
        : makeBinaryCa2dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: { spatialDim: 2, isDiscrete: true, numStates: 2, systemName: 'BinaryCA2D', toRgb: binaryToRgb },
      }
    },
  },

  'Outer-totalistic CA 2D': {
    name: 'Outer-totalistic CA 2D',
    spatialDim: 2,
    isDiscrete: true,
    channels: 1,
    defaultSize: { H: 64, W: 64 },
    defaultSteps: { steps: 96, every: 2, skip: 0 },
    note: 'Life-like birth/survival cellular automata.',
    paramControls: [
      { kind: 'number', key: 'kernelSize', label: 'Kernel size', min: 3, max: 11, step: 2 },
      { kind: 'checkbox', key: 'randomRule', label: 'Random rule' },
      { kind: 'text', key: 'desc', label: 'Rule' },
    ],
    paramDefaults: { kernelSize: 3, randomRule: false, desc: 'B3/S23' },
    seedModes: ['noise', 'single', 'zeros'],
    build(cfg, device) {
      const { params, init } = rngs(cfg)
      const dims = dims2d(cfg)
      const L = cfg.kernelSize * cfg.kernelSize
      const tables = cfg.randomRule
        ? randomBsTables(L, 0.5, 0.5, () => params.float())
        : bsTables(parseBs(cfg.desc), L)
      const p = { K: cfg.kernelSize, birth: expandTable(tables.birth, dims.B), survive: expandTable(tables.survive, dims.B) }
      const x0 = discreteSeed(dims, cfg.seedMode, cfg.seedP, init)
      const stepper = device
        ? makeGpuOuterTotalistic2d(device, p, x0, dims)
        : makeOuterTotalistic2dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: { spatialDim: 2, isDiscrete: true, numStates: 2, systemName: 'OuterTotalisticCA2D', toRgb: binaryToRgb },
      }
    },
  },

  'Langton CA 1D': {
    name: 'Langton CA 1D',
    spatialDim: 1,
    isDiscrete: true,
    channels: 1,
    defaultSize: { H: null, W: 128 },
    defaultSteps: { steps: 128, every: 1, skip: 0 },
    note: 'Implicit random K-state CA controlled by Langton lambda.',
    paramControls: [
      { kind: 'number', key: 'numStates', label: 'States', min: 2, max: 16 },
      { kind: 'number', key: 'kernelSize', label: 'Kernel size', min: 1, max: 7, step: 2 },
      { kind: 'slider', key: 'lambda', label: 'Lambda', min: 0, max: 1, step: 0.01 },
      { kind: 'number', key: 'ruleSeed', label: 'Rule seed', min: 0, max: 2147483647 },
    ],
    paramDefaults: { numStates: 2, kernelSize: 3, lambda: 0.5, ruleSeed: 0 },
    seedModes: ['noise', 'single', 'zeros'],
    build(cfg, device) {
      const { init } = rngs(cfg)
      const dims = dims1d(cfg)
      const B = dims.B
      const p = {
        numStates: cfg.numStates,
        kernelSize: cfg.kernelSize,
        lambda: new Float32Array(B).fill(cfg.lambda),
        seed: new Uint32Array(B).fill(cfg.ruleSeed),
      }
      const x0 = discreteSeed(dims, cfg.seedMode, cfg.seedP, init, cfg.numStates)
      const stepper = device
        ? makeGpuLangtonCa1d(device, p, x0, dims)
        : makeLangtonCa1dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: {
          spatialDim: 1,
          isDiscrete: true,
          numStates: cfg.numStates,
          systemName: 'LangtonCA1D',
          toRgb: makeLangtonToRgb(cfg.numStates),
        },
      }
    },
  },

  'Langton CA 2D': {
    name: 'Langton CA 2D',
    spatialDim: 2,
    isDiscrete: true,
    channels: 1,
    defaultSize: { H: 64, W: 64 },
    defaultSteps: { steps: 96, every: 2, skip: 0 },
    note: 'Implicit random K-state 2D CA controlled by Langton lambda.',
    paramControls: [
      { kind: 'number', key: 'numStates', label: 'States', min: 2, max: 16 },
      { kind: 'number', key: 'kernelSize', label: 'Kernel size', min: 1, max: 7, step: 2 },
      { kind: 'slider', key: 'lambda', label: 'Lambda', min: 0, max: 1, step: 0.01 },
      { kind: 'number', key: 'ruleSeed', label: 'Rule seed', min: 0, max: 2147483647 },
    ],
    paramDefaults: { numStates: 4, kernelSize: 3, lambda: 0.5, ruleSeed: 0 },
    seedModes: ['noise', 'single', 'zeros'],
    build(cfg, device) {
      const { init } = rngs(cfg)
      const dims = dims2d(cfg)
      const B = dims.B
      const p = {
        numStates: cfg.numStates,
        kernelSize: cfg.kernelSize,
        lambda: new Float32Array(B).fill(cfg.lambda),
        seed: new Uint32Array(B).fill(cfg.ruleSeed),
      }
      const x0 = discreteSeed(dims, cfg.seedMode, cfg.seedP, init, cfg.numStates)
      const stepper = device
        ? makeGpuLangtonCa2d(device, p, x0, dims)
        : makeLangtonCa2dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: {
          spatialDim: 2,
          isDiscrete: true,
          numStates: cfg.numStates,
          systemName: 'LangtonCA2D',
          toRgb: makeLangtonToRgb(cfg.numStates),
        },
      }
    },
  },

  'Coupled logistic map 1D': {
    name: 'Coupled logistic map 1D',
    spatialDim: 1,
    isDiscrete: false,
    channels: 1,
    defaultSize: { H: null, W: 256 },
    defaultSteps: { steps: 256, every: 2, skip: 32 },
    note: 'Continuous coupled chaotic map lattice.',
    paramControls: [
      { kind: 'slider', key: 'r', label: 'r', min: 3, max: 4, step: 0.01 },
      { kind: 'slider', key: 'eps', label: 'epsilon', min: 0, max: 1, step: 0.01 },
    ],
    paramDefaults: { r: 3.8, eps: 0.25 },
    seedModes: ['noise'],
    build(cfg, device) {
      const { init } = rngs(cfg)
      const dims = dims1d(cfg)
      const B = dims.B
      const p = { r: new Float32Array(B).fill(cfg.r), eps: new Float32Array(B).fill(cfg.eps) }
      const x0 = coupledLogisticSeed(dims, init)
      const stepper = device
        ? makeGpuCoupledLogistic1d(device, p, x0, dims)
        : makeCoupledLogistic1dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: {
          spatialDim: 1,
          isDiscrete: false,
          numStates: null,
          systemName: 'CoupledLogistic1D',
          toRgb: continuousToRgb,
        },
      }
    },
  },

  'Gray-Scott 2D': {
    name: 'Gray-Scott 2D',
    spatialDim: 2,
    isDiscrete: false,
    channels: 2,
    defaultSize: { H: 96, W: 96 },
    defaultSteps: { steps: 800, every: 10, skip: 100 },
    note: 'Continuous reaction-diffusion system.',
    paramControls: [
      { kind: 'number', key: 'Du', label: 'Du', min: 0, max: 1, step: 0.0001 },
      { kind: 'number', key: 'Dv', label: 'Dv', min: 0, max: 1, step: 0.0001 },
      { kind: 'number', key: 'F', label: 'F', min: 0, max: 0.2, step: 0.0001 },
      { kind: 'number', key: 'k', label: 'k', min: 0, max: 0.2, step: 0.0001 },
    ],
    paramDefaults: { Du: 0.16, Dv: 0.08, F: 0.035, k: 0.06 },
    seedModes: ['noise'],
    build(cfg, device) {
      const { init } = rngs(cfg)
      const dims = dims2d(cfg, 2)
      const B = dims.B
      const p = {
        Du: new Float32Array(B).fill(cfg.Du),
        Dv: new Float32Array(B).fill(cfg.Dv),
        F: new Float32Array(B).fill(cfg.F),
        k: new Float32Array(B).fill(cfg.k),
      }
      const x0 = grayScottSeed(dims, init)
      const stepper = device
        ? makeGpuGrayScott2d(device, p, x0, dims)
        : makeGrayScott2dStepper(p, x0, dims)
      return {
        dims,
        stepper,
        meta: {
          spatialDim: 2,
          isDiscrete: false,
          numStates: null,
          systemName: 'GrayScott2D',
          toRgb: grayScottToRgb,
        },
      }
    },
  },
}
