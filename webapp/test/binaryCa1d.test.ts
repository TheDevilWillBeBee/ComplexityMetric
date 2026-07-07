import { describe, expect, it } from 'vitest'
import { makeRng } from '../src/core/prng'
import { countRecordedFrames, isRecorded, runRollout } from '../src/systems/rollout'
import { discreteSeed } from '../src/systems/cpu/seeds'
import {
  expandRule,
  makeBinaryCa1dStepper,
  sampleBinaryCa1dRule,
} from '../src/systems/cpu/binaryCa1d'
import { binaryToRgb } from '../src/ui/render'

const META = {
  spatialDim: 1 as const,
  isDiscrete: true,
  numStates: 2,
  systemName: 'BinaryCA1D',
  toRgb: binaryToRgb,
}

function row(data: Int32Array, t: number, W: number): number[] {
  return Array.from(data.subarray(t * W, (t + 1) * W))
}

describe('BinaryCA1D', () => {
  it('rule 30 from a single seed matches the hand-verified evolution', async () => {
    const W = 16
    const dims = { B: 1, C: 1, H: 1, W }
    const rng = makeRng(0)
    const rule = expandRule(sampleBinaryCa1dRule(3, 30, rng), 1)
    const x0 = discreteSeed(dims, 'single', 0.5, rng) // center = index 8
    const stepper = makeBinaryCa1dStepper({ K: 3, rule }, x0, dims)
    const r = await runRollout(stepper, dims, META, { steps: 3, every: 1, skip: 0 })
    const data = r.data as Int32Array
    // recorded frames are AFTER each step (initial state not included)
    expect(row(data, 0, W)).toEqual([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    expect(row(data, 1, W)).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    expect(row(data, 2, W)).toEqual([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
  })

  it('rule table lookup agrees with the boolean formula l XOR (c OR r) on random states', async () => {
    const W = 64
    const B = 2
    const dims = { B, C: 1, H: 1, W }
    const rng = makeRng(42)
    const rule = expandRule(sampleBinaryCa1dRule(3, 30, rng), B)
    const x0 = discreteSeed(dims, 'noise', 0.5, rng)
    const stepper = makeBinaryCa1dStepper({ K: 3, rule }, x0, dims)
    await stepper.step(1)
    const got = (await stepper.readState()) as Int32Array
    for (let b = 0; b < B; b++) {
      for (let x = 0; x < W; x++) {
        const l = x0[b * W + ((x - 1 + W) % W)]
        const c = x0[b * W + x]
        const rr = x0[b * W + ((x + 1) % W)]
        expect(got[b * W + x]).toBe(l ^ (c | rr))
      }
    }
  })

  it('wolfram rule decoding: bit i of ruleInt is the output for neighborhood index i', () => {
    const rng = makeRng(1)
    const rule110 = sampleBinaryCa1dRule(3, 110, rng)
    // 110 = 0b01101110
    expect(Array.from(rule110)).toEqual([0, 1, 1, 1, 0, 1, 1, 0])
  })
})

describe('rollout recording semantics', () => {
  it('records t where (t+1) % every == 0 and t >= skip', () => {
    // steps=10, every=3, skip=4 → t=5 (6%3), t=8 (9%3); t=2 excluded by skip
    expect(isRecorded(2, 3, 4)).toBe(false)
    expect(isRecorded(5, 3, 4)).toBe(true)
    expect(isRecorded(8, 3, 4)).toBe(true)
    expect(countRecordedFrames({ steps: 10, every: 3, skip: 4 })).toBe(2)
  })

  it('falls back to the final state as a single frame when nothing is recorded', async () => {
    const W = 8
    const dims = { B: 1, C: 1, H: 1, W }
    const rng = makeRng(0)
    const rule = expandRule(sampleBinaryCa1dRule(3, 30, rng), 1)
    const x0 = discreteSeed(dims, 'single', 0.5, rng)
    const stepper = makeBinaryCa1dStepper({ K: 3, rule }, x0, dims)
    const r = await runRollout(stepper, dims, META, { steps: 2, every: 5, skip: 0 })
    expect(r.T).toBe(1)
    // frame is the state after 2 steps, not x0
    const twoStep = makeBinaryCa1dStepper({ K: 3, rule }, x0, dims)
    await twoStep.step(2)
    expect(Array.from(r.data as Int32Array)).toEqual(Array.from((await twoStep.readState()) as Int32Array))
  })
})

describe('prng', () => {
  it('is deterministic per seed and differs across seeds', () => {
    const a1 = makeRng(123)
    const a2 = makeRng(123)
    const b = makeRng(124)
    const s1 = Array.from({ length: 8 }, () => a1.uint32())
    const s2 = Array.from({ length: 8 }, () => a2.uint32())
    const s3 = Array.from({ length: 8 }, () => b.uint32())
    expect(s1).toEqual(s2)
    expect(s1).not.toEqual(s3)
  })

  it('float() stays in [0,1) and looks roughly uniform', () => {
    const rng = makeRng(7)
    let sum = 0
    for (let i = 0; i < 10000; i++) {
      const v = rng.float()
      expect(v).toBeGreaterThanOrEqual(0)
      expect(v).toBeLessThan(1)
      sum += v
    }
    expect(sum / 10000).toBeGreaterThan(0.45)
    expect(sum / 10000).toBeLessThan(0.55)
  })
})
