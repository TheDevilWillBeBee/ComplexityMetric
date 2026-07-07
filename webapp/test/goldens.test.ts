/**
 * Parity tests against fixtures generated from the Python reference
 * implementations. Generate them from the repo root with:
 *
 *     uv run python make_goldens.py
 *
 * These tests SKIP (with a warning) when the fixture file is absent so the
 * suite stays green on a fresh checkout.
 */
import { readFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { describe, expect, it } from 'vitest'
import type { Stepper } from '../src/systems/types'
import { fmix32, makeCoeffU32 } from '../src/systems/langtonHash'
import { makeBinaryCa1dStepper, sampleBinaryCa1dRule, expandRule } from '../src/systems/cpu/binaryCa1d'
import { makeBinaryCa2dStepper } from '../src/systems/cpu/binaryCa2d'
import { bsTables, parseBs } from '../src/systems/rules'
import { expandTable, makeOuterTotalistic2dStepper } from '../src/systems/cpu/outerTotalistic'
import { makeLangtonCa1dStepper, makeLangtonCa2dStepper } from '../src/systems/cpu/langtonCa'
import { makeCoupledLogistic1dStepper } from '../src/systems/cpu/coupledLogistic1d'
import { makeGrayScott2dStepper } from '../src/systems/cpu/grayScott2d'
import { pcaProject } from '../src/projection/pca'
import { makeRng } from '../src/core/prng'
import type { EmbeddingData } from '../src/extractors/types'
import { makeEntropy } from '../src/metrics/entropy'
import { makeFutureStateMI } from '../src/metrics/mi'
import { makeCompressedRatio, makeEntropyMinusCompressedRatio } from '../src/metrics/compressedRatio'
import { makeDensityTransientTime } from '../src/metrics/transient'
import { makeOpenEndedness } from '../src/metrics/openEndedness'
import { metricWindows } from '../src/metrics/windows'

const FIXTURE = fileURLToPath(new URL('./fixtures/goldens.json', import.meta.url))
const hasFixture = existsSync(FIXTURE)
if (!hasFixture) {
  console.warn('goldens.json missing — run `uv run python make_goldens.py` from the repo root; skipping parity tests')
}
const g: any = hasFixture ? JSON.parse(readFileSync(FIXTURE, 'utf8')) : null

async function collectSteps(stepper: Stepper, n: number): Promise<number[][]> {
  const out: number[][] = []
  for (let i = 0; i < n; i++) {
    await stepper.step(1)
    out.push(Array.from(await stepper.readState()))
  }
  return out
}

function expectClose(got: number[], want: number[], tol: number) {
  expect(got.length).toBe(want.length)
  for (let i = 0; i < got.length; i++) {
    if (Math.abs(got[i] - want[i]) > tol) {
      // fall through to a diff-style failure with context
      expect(`i=${i}: ${got[i]}`).toBe(`i=${i}: ${want[i]}`)
    }
  }
}

describe.skipIf(!hasFixture)('Python parity: systems', () => {
  it('BinaryCA1D rule 30', async () => {
    const f = g.binary_ca1d_rule30
    const rule = expandRule(sampleBinaryCa1dRule(3, f.rule_int, makeRng(0)), 1)
    const stepper = makeBinaryCa1dStepper({ K: 3, rule }, Int32Array.from(f.x0), {
      B: 1, C: 1, H: 1, W: f.W,
    })
    expect(await collectSteps(stepper, f.steps.length)).toEqual(f.steps)
  })

  it('BinaryCA2D with an explicit Python-sampled table (pins bit weights)', async () => {
    const f = g.binary_ca2d
    const stepper = makeBinaryCa2dStepper(
      { rule: Uint8Array.from(f.rule) },
      Int32Array.from(f.x0),
      { B: 1, C: 1, H: f.H, W: f.W },
    )
    expect(await collectSteps(stepper, f.steps.length)).toEqual(f.steps)
  })

  it('OuterTotalisticCA2D B3/S23 glider', async () => {
    const f = g.ot2d_glider
    const { birth, survive } = bsTables(parseBs('B3/S23'), 9)
    const stepper = makeOuterTotalistic2dStepper(
      { K: 3, birth: expandTable(birth, 1), survive: expandTable(survive, 1) },
      Int32Array.from(f.x0),
      { B: 1, C: 1, H: f.H, W: f.W },
    )
    expect(await collectSteps(stepper, f.steps.length)).toEqual(f.steps)
  })

  it('fmix32 and coefficient generation match', () => {
    const f = g.langton_hash
    for (let i = 0; i < f.fmix32_inputs.length; i++) {
      expect(fmix32(f.fmix32_inputs[i] % 4294967296)).toBe(f.fmix32_outputs[i])
    }
    for (const L of [3, 9, 25]) {
      const unsigned = makeCoeffU32(L)
      const signed = Array.from(unsigned, (c) => (c >= 2147483648 ? c - 4294967296 : c))
      expect(signed).toEqual(f.coeff_signed[String(L)])
    }
  })

  for (const key of ['langton1d_lam50', 'langton1d_lam37']) {
    it(`LangtonCA1D bit-exact (${key})`, async () => {
      const f = g[key]
      const stepper = makeLangtonCa1dStepper(
        {
          numStates: f.K,
          kernelSize: f.kernel,
          lambda: Float32Array.of(f.lambda),
          seed: Uint32Array.of(f.seed),
        },
        Int32Array.from(f.x0),
        { B: 1, C: 1, H: 1, W: f.W },
      )
      expect(await collectSteps(stepper, f.steps.length)).toEqual(f.steps)
    })
  }

  it('LangtonCA2D bit-exact', async () => {
    const f = g.langton2d
    const stepper = makeLangtonCa2dStepper(
      {
        numStates: f.K,
        kernelSize: f.kernel,
        lambda: Float32Array.of(f.lambda),
        seed: Uint32Array.of(f.seed),
      },
      Int32Array.from(f.x0),
      { B: 1, C: 1, H: f.H, W: f.W },
    )
    expect(await collectSteps(stepper, f.steps.length)).toEqual(f.steps)
  })

  it('GrayScott2D within float32 tolerance over 5 steps', async () => {
    const f = g.gray_scott
    const stepper = makeGrayScott2dStepper(
      {
        Du: Float32Array.of(f.Du),
        Dv: Float32Array.of(f.Dv),
        F: Float32Array.of(f.F),
        k: Float32Array.of(f.k),
      },
      Float32Array.from(f.x0),
      { B: 1, C: 2, H: f.H, W: f.W },
    )
    const got = await collectSteps(stepper, f.steps.length)
    for (let s = 0; s < got.length; s++) expectClose(got[s], f.steps[s], 2e-5)
  })

  it('PCA projection matches torch SVD up to per-component sign', () => {
    const f = g.pca
    const got = pcaProject(Float32Array.from(f.z), f.T, f.D, 2)
    for (let c = 0; c < 2; c++) {
      // align sign on the largest-|value| entry of the reference column
      let ref = 0
      for (let t = 0; t < f.T; t++) if (Math.abs(f.y[t * 2 + c]) > Math.abs(f.y[ref * 2 + c])) ref = t
      const flip = Math.sign(f.y[ref * 2 + c]) === Math.sign(got[ref * 2 + c]) ? 1 : -1
      for (let t = 0; t < f.T; t++) {
        expect(flip * got[t * 2 + c]).toBeCloseTo(f.y[t * 2 + c], 3)
      }
    }
  })

  it('CoupledLogistic1D within float32 tolerance over 5 steps', async () => {
    const f = g.coupled_logistic
    const stepper = makeCoupledLogistic1dStepper(
      { r: Float32Array.of(f.r), eps: Float32Array.of(f.eps) },
      Float32Array.from(f.x0),
      { B: 1, C: 1, H: 1, W: f.W },
    )
    const got = await collectSteps(stepper, f.steps.length)
    for (let s = 0; s < got.length; s++) expectClose(got[s], f.steps[s], 2e-5)
  })
})

describe.skipIf(!hasFixture)('Python parity: metrics', () => {
  const emb = (key: 'disc2' | 'disc4' | 'cont'): EmbeddingData => {
    const m = g.metrics
    if (key === 'cont') {
      return { data: Float32Array.from(m.cont), B: m.B, T: m.T, D: m.D, isDiscrete: false, numStates: null }
    }
    return {
      data: Int32Array.from(m[key]),
      B: m.B,
      T: m.T,
      D: m.D,
      isDiscrete: true,
      numStates: key === 'disc2' ? 2 : 4,
    }
  }

  const check = (got: Float64Array, want: number[], tol = 1e-5) => {
    expect(got.length).toBe(want.length)
    for (let i = 0; i < got.length; i++) expect(got[i]).toBeCloseTo(want[i], -Math.log10(tol) - 1)
  }

  it('Entropy (all modes, k=2 and k=4)', async () => {
    const m = g.metrics
    check(await makeEntropy('time').compute(emb('disc2')), m.entropy_time_k2)
    check(await makeEntropy('space').compute(emb('disc2')), m.entropy_space_k2)
    check(await makeEntropy('both').compute(emb('disc2')), m.entropy_both_k2)
    check(await makeEntropy('time').compute(emb('disc4')), m.entropy_time_k4)
    check(await makeEntropy('both').compute(emb('disc4')), m.entropy_both_k4)
  })

  it('FutureStateMutualInformation (time/space)', async () => {
    const m = g.metrics
    check(await makeFutureStateMI('time').compute(emb('disc2')), m.mi_time_k2)
    check(await makeFutureStateMI('space').compute(emb('disc2')), m.mi_space_k2)
    check(await makeFutureStateMI('time').compute(emb('disc4')), m.mi_time_k4)
  })

  it('CompressedRatio matches Python brotli byte counts', async () => {
    const m = g.metrics
    check(await makeCompressedRatio().compute(emb('disc2')), m.compressed_ratio_k2, 1e-6)
    check(await makeCompressedRatio().compute(emb('disc4')), m.compressed_ratio_k4, 1e-6)
  })

  it('EntropyMinusCompressedRatio', async () => {
    check(await makeEntropyMinusCompressedRatio().compute(emb('disc2')), g.metrics.entropy_minus_cr_k2, 1e-5)
  })

  it('DensityTransientTime', async () => {
    check(await makeDensityTransientTime().compute(emb('disc2')), g.metrics.density_transient_k2)
  })

  it('OpenEndedness on the continuous fixture', async () => {
    check(await makeOpenEndedness().compute(emb('cont')), g.metrics.open_endedness_cont, 1e-4)
  })

  it('metric_windows cases', () => {
    for (const c of g.metric_windows) {
      const got = metricWindows(c.T, c.chunk, c.stride).map(([a, b]) => [a, b])
      expect(got).toEqual(c.windows)
    }
  })
})
