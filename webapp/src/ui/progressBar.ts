import type { ProgressReporter } from '../core/progress'
import { makeProgress } from '../core/progress'
import { clear, el } from './dom'

export interface ProgressBar {
  reporter: ProgressReporter
  /** Weighted stages for this run; call before starting. */
  begin(stages: { label: string; weight: number }[]): void
  finish(): void
  hide(): void
}

/**
 * Progress bar with weighted stages: overall = completed weights + current
 * stage weight * stage-local fraction.
 */
export function mountProgressBar(container: HTMLElement): ProgressBar {
  const label = el('div', { className: 'progress-label' })
  const fill = el('div', { className: 'progress-fill' })
  const track = el('div', { className: 'progress-track' }, fill)
  const box = el('div', { className: 'progress hidden' }, label, track)
  container.append(box)

  let stages: { label: string; weight: number }[] = []
  let stageIdx = -1

  const update = (stageLabel: string, frac: number) => {
    let done = 0
    for (let i = 0; i < stageIdx; i++) done += stages[i]?.weight ?? 0
    const cur = stages[stageIdx]?.weight ?? 0
    const total = Math.min(1, done + cur * frac)
    label.textContent = stageLabel
    fill.style.width = `${(total * 100).toFixed(1)}%`
  }

  const inner = makeProgress(update)
  const reporter: ProgressReporter = {
    stage(l: string) {
      stageIdx = Math.min(stageIdx + 1, stages.length - 1)
      inner.stage(l)
    },
    set: (f) => inner.set(f),
    tick: () => inner.tick(),
  }

  return {
    reporter,
    begin(s) {
      const total = s.reduce((a, b) => a + b.weight, 0) || 1
      stages = s.map((x) => ({ label: x.label, weight: x.weight / total }))
      stageIdx = -1
      box.classList.remove('hidden')
      fill.style.width = '0%'
      label.textContent = ''
    },
    finish() {
      fill.style.width = '100%'
      label.textContent = 'Done'
    },
    hide() {
      box.classList.add('hidden')
      clear(label)
    },
  }
}
