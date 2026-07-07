import type { Dims, RolloutData, StateArray, Stepper, ToRgbFn } from './types'
import type { ProgressReporter } from '../core/progress'
import { nullProgress } from '../core/progress'

export interface RolloutOptions {
  steps: number
  every: number
  skip: number
}

export interface RolloutMeta {
  spatialDim: 1 | 2
  isDiscrete: boolean
  numStates: number | null
  systemName: string
  toRgb: ToRgbFn
}

/** Recording rule (base.py:55): frame t is recorded iff (t+1) % every == 0 and t >= skip. */
export function isRecorded(t: number, every: number, skip: number): boolean {
  return (t + 1) % every === 0 && t >= skip
}

export function countRecordedFrames(opts: RolloutOptions): number {
  let n = 0
  for (let t = 0; t < opts.steps; t++) if (isRecorded(t, opts.every, opts.skip)) n++
  return n
}

const MEMORY_BUDGET_BYTES = 512 * 1024 * 1024

/**
 * Drives a stepper and records frames into a (B, T, C, H, W) rollout.
 * If no frame qualifies, the final state is recorded as a single frame
 * (matches base.py:57).
 */
export async function runRollout(
  stepper: Stepper,
  dims: Dims,
  meta: RolloutMeta,
  opts: RolloutOptions,
  progress: ProgressReporter = nullProgress,
): Promise<RolloutData> {
  const { B, C, H, W } = dims
  const steps = Math.max(0, Math.floor(opts.steps))
  const every = Math.max(1, Math.floor(opts.every))
  const skip = Math.max(0, Math.floor(opts.skip))
  const T = Math.max(1, countRecordedFrames({ steps, every, skip }))

  const frameSize = C * H * W
  const bytes = B * T * frameSize * 4
  if (bytes > MEMORY_BUDGET_BYTES) {
    throw new Error(
      `rollout would need ${(bytes / 1e6).toFixed(0)} MB ` +
        `(B=${B}, T=${T}, ${C}x${H}x${W}); raise "Every", lower "Steps", or shrink the grid`,
    )
  }

  const first = await stepper.readState()
  const data: StateArray =
    first instanceof Int32Array ? new Int32Array(B * T * frameSize) : new Float32Array(B * T * frameSize)

  const record = (state: StateArray, tRec: number) => {
    for (let b = 0; b < B; b++) {
      const src = state.subarray(b * frameSize, (b + 1) * frameSize)
      data.set(src, (b * T + tRec) * frameSize)
    }
  }

  let tRec = 0
  const fastPath = (stepper as { rolloutInto?: unknown }).rolloutInto
  if (typeof fastPath === 'function') {
    // GPU steppers record frames on-device and drain them in bulk
    tRec = await (stepper as import('./gpu/gpuStepper').GpuStepper).rolloutInto(
      data,
      T,
      { steps, every, skip },
      progress,
    )
  } else {
    for (let t = 0; t < steps; t++) {
      await stepper.step(1)
      if (isRecorded(t, every, skip)) {
        record(await stepper.readState(), tRec)
        tRec++
      }
      if ((t & 15) === 0) {
        progress.set(steps > 0 ? t / steps : 1)
        await progress.tick()
      }
    }
  }
  if (tRec === 0) record(await stepper.readState(), 0)
  progress.set(1)

  return {
    data,
    B,
    T,
    C,
    H,
    W,
    spatialDim: meta.spatialDim,
    isDiscrete: meta.isDiscrete,
    numStates: meta.numStates,
    steps,
    every,
    skip,
    systemName: meta.systemName,
    toRgb: meta.toRgb,
  }
}
