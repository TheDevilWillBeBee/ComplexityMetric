import type { RolloutData } from '../systems/types'
import { el } from './dom'
import { drawScaled, render2dFrame } from './render'

/**
 * Playback player for 2D rollouts (replaces the Python app's mp4):
 * play/pause, frame scrubber, ~24 fps, frames rendered on demand.
 */
export function mountPlayback(container: HTMLElement, rollout: RolloutData, maxWidth = 900): void {
  const canvas = el('canvas')
  const playBtn = el('button', { className: 'play-btn' }, '▶')
  const scrub = el('input', {
    type: 'range',
    min: 0,
    max: rollout.T - 1,
    step: 1,
    value: 0,
  }) as HTMLInputElement
  const frameLabel = el('span', { className: 'frame-label' })
  const controls = el('div', { className: 'playback-controls' }, playBtn, scrub, frameLabel)
  container.append(canvas, controls)

  let frame = 0
  let playing = false
  let lastDraw = 0
  const FPS = 24

  const draw = () => {
    drawScaled(canvas, render2dFrame(rollout, frame), maxWidth)
    scrub.value = String(frame)
    frameLabel.textContent = `${frame + 1}/${rollout.T}`
  }

  const loop = (now: number) => {
    if (!playing) return
    if (now - lastDraw >= 1000 / FPS) {
      frame = (frame + 1) % rollout.T
      draw()
      lastDraw = now
    }
    requestAnimationFrame(loop)
  }

  playBtn.addEventListener('click', () => {
    playing = !playing
    playBtn.textContent = playing ? '❚❚' : '▶'
    if (playing) requestAnimationFrame(loop)
  })
  scrub.addEventListener('input', () => {
    frame = Number(scrub.value)
    draw()
  })

  draw()
}
