import { describe, expect, it } from 'vitest'
import { makeRng } from '../src/core/prng'
import { discreteSeed } from '../src/systems/cpu/seeds'
import { bsTables, parseBs } from '../src/systems/rules'
import {
  expandTable,
  makeOuterTotalistic2dStepper,
} from '../src/systems/cpu/outerTotalistic'
import { makeBinaryCa2dStepper } from '../src/systems/cpu/binaryCa2d'
import { makeLangtonCa1dStepper, makeLangtonCa2dStepper } from '../src/systems/cpu/langtonCa'
import { makeCoupledLogistic1dStepper } from '../src/systems/cpu/coupledLogistic1d'
import { makeGrayScott2dStepper } from '../src/systems/cpu/grayScott2d'

const GLIDER: ReadonlyArray<readonly [number, number]> = [
  [1, 2],
  [2, 3],
  [3, 1],
  [3, 2],
  [3, 3],
]

function gliderGrid(H: number, W: number, cells: ReadonlyArray<readonly [number, number]>): Int32Array {
  const x = new Int32Array(H * W)
  for (const [y, xx] of cells) x[y * W + xx] = 1
  return x
}

describe('parseBs', () => {
  it('parses per-character digits', () => {
    const { birth, survive } = parseBs('B3/S23')
    expect([...birth]).toEqual([3])
    expect([...survive].sort()).toEqual([2, 3])
  })
  it('parses multi-digit numbers when separators are present', () => {
    const { birth, survive } = parseBs('B10,12/S2,11')
    expect([...birth].sort((a, b) => a - b)).toEqual([10, 12])
    expect([...survive].sort((a, b) => a - b)).toEqual([2, 11])
  })
  it('handles empty survive part', () => {
    const { birth, survive } = parseBs('B3')
    expect([...birth]).toEqual([3])
    expect(survive.size).toBe(0)
  })
})

describe('OuterTotalisticCA2D', () => {
  it('B3/S23 glider translates by (1,1) every 4 steps', async () => {
    const H = 8,
      W = 8
    const { birth, survive } = bsTables(parseBs('B3/S23'), 9)
    const stepper = makeOuterTotalistic2dStepper(
      { K: 3, birth: expandTable(birth, 1), survive: expandTable(survive, 1) },
      gliderGrid(H, W, GLIDER),
      { B: 1, C: 1, H, W },
    )
    await stepper.step(4)
    const got = (await stepper.readState()) as Int32Array
    const expected = gliderGrid(H, W, GLIDER.map(([y, x]) => [y + 1, x + 1] as const))
    expect(Array.from(got)).toEqual(Array.from(expected))
  })
})

describe('BinaryCA2D', () => {
  it('a rule table built for Life reproduces the glider (pins the bit-weight mapping)', async () => {
    // weights by offset, matching systems.py:449-453 after flip+cross-correlation
    const others = [64, 128, 256, 8, 32, 1, 2, 4]
    const rule = new Uint8Array(512)
    for (let idx = 0; idx < 512; idx++) {
      const center = (idx & 16) !== 0
      let nsum = 0
      for (const w of others) if (idx & w) nsum++
      rule[idx] = (center ? nsum === 2 || nsum === 3 : nsum === 3) ? 1 : 0
    }
    const H = 8,
      W = 8
    const stepper = makeBinaryCa2dStepper({ rule }, gliderGrid(H, W, GLIDER), { B: 1, C: 1, H, W })
    await stepper.step(4)
    const got = (await stepper.readState()) as Int32Array
    const expected = gliderGrid(H, W, GLIDER.map(([y, x]) => [y + 1, x + 1] as const))
    expect(Array.from(got)).toEqual(Array.from(expected))
  })
})

describe('LangtonCA', () => {
  const dims1d = { B: 1, C: 1, H: 1, W: 32 }
  const noise = (K: number) => discreteSeed(dims1d, 'noise', 0.5, makeRng(3), K)

  it('lambda=0 sends everything to the quiescent state', async () => {
    const stepper = makeLangtonCa1dStepper(
      { numStates: 4, kernelSize: 3, lambda: Float32Array.of(0), seed: Uint32Array.of(7) },
      noise(4),
      dims1d,
    )
    await stepper.step(1)
    expect(Array.from((await stepper.readState()) as Int32Array).every((v) => v === 0)).toBe(true)
  })

  it('lambda=1 leaves no quiescent cells', async () => {
    const stepper = makeLangtonCa1dStepper(
      { numStates: 4, kernelSize: 3, lambda: Float32Array.of(1), seed: Uint32Array.of(7) },
      noise(4),
      dims1d,
    )
    await stepper.step(1)
    const got = (await stepper.readState()) as Int32Array
    expect(Array.from(got).every((v) => v >= 1 && v <= 3)).toBe(true)
  })

  it('is deterministic per rule seed and differs across seeds (2D)', async () => {
    const dims = { B: 1, C: 1, H: 8, W: 8 }
    const x0 = discreteSeed(dims, 'noise', 0.5, makeRng(5), 4)
    const run = async (seed: number) => {
      const s = makeLangtonCa2dStepper(
        { numStates: 4, kernelSize: 3, lambda: Float32Array.of(0.5), seed: Uint32Array.of(seed) },
        x0,
        dims,
      )
      await s.step(3)
      return Array.from((await s.readState()) as Int32Array)
    }
    expect(await run(42)).toEqual(await run(42))
    expect(await run(42)).not.toEqual(await run(43))
  })
})

describe('CoupledLogistic1D', () => {
  it('eps=0 reduces to independent logistic maps', async () => {
    const W = 8
    const dims = { B: 1, C: 1, H: 1, W }
    const x0 = new Float32Array(W)
    for (let i = 0; i < W; i++) x0[i] = (i + 0.5) / W
    const stepper = makeCoupledLogistic1dStepper(
      { r: Float32Array.of(3.7), eps: Float32Array.of(0) },
      x0,
      dims,
    )
    await stepper.step(3)
    const got = (await stepper.readState()) as Float32Array
    for (let i = 0; i < W; i++) {
      let v = x0[i]
      for (let s = 0; s < 3; s++) v = Math.fround(Math.fround(3.7) * v * (1 - v))
      expect(got[i]).toBeCloseTo(v, 6)
    }
  })
})

describe('GrayScott2D', () => {
  it('the uniform state u=1, v=0 is a fixed point', async () => {
    const H = 6,
      W = 6
    const x0 = new Float32Array(2 * H * W)
    x0.fill(1, 0, H * W) // u = 1, v = 0
    const stepper = makeGrayScott2dStepper(
      { Du: Float32Array.of(0.16), Dv: Float32Array.of(0.08), F: Float32Array.of(0.035), k: Float32Array.of(0.06) },
      x0,
      { B: 1, C: 2, H, W },
    )
    await stepper.step(10)
    const got = (await stepper.readState()) as Float32Array
    for (let i = 0; i < H * W; i++) {
      expect(got[i]).toBe(1)
      expect(got[H * W + i]).toBe(0)
    }
  })
})
