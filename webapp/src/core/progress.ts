/**
 * Cooperative progress reporting for the run pipeline.
 *
 * `tick()` yields to the event loop (via requestAnimationFrame when
 * available) if more than ~16ms elapsed since the last yield, keeping the
 * UI responsive during CPU-heavy stages.
 */

export interface ProgressReporter {
  stage(label: string): void
  /** Progress within the current stage, 0..1. */
  set(frac: number): void
  /** Yield to the event loop if we've been hogging it. */
  tick(): Promise<void>
}

export const nullProgress: ProgressReporter = {
  stage: () => {},
  set: () => {},
  tick: async () => {},
}

const nextFrame: () => Promise<void> =
  typeof requestAnimationFrame === 'function'
    ? () => new Promise((r) => requestAnimationFrame(() => r()))
    : () => new Promise((r) => setTimeout(r, 0))

export function makeProgress(
  onUpdate: (stage: string, frac: number) => void,
): ProgressReporter {
  let stage = ''
  let lastYield = performance.now()
  return {
    stage(label: string) {
      stage = label
      onUpdate(stage, 0)
    },
    set(frac: number) {
      onUpdate(stage, Math.min(1, Math.max(0, frac)))
    },
    async tick() {
      if (performance.now() - lastYield > 16) {
        await nextFrame()
        lastYield = performance.now()
      }
    },
  }
}
