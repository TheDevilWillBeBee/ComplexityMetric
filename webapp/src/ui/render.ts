import type { RolloutData, StateArray, ToRgbFn } from '../systems/types'
import { frameView } from '../systems/types'

/** to_rgb for binary CAs (systems.py:209): gray = v*0.5 + 0.25. */
export function binaryToRgb(
  frame: StateArray,
  _C: number,
  H: number,
  W: number,
  out: Uint8ClampedArray,
): void {
  for (let i = 0; i < H * W; i++) {
    const g = frame[i] > 0.5 ? 191 : 64
    const j = i * 4
    out[j] = g
    out[j + 1] = g
    out[j + 2] = g
    out[j + 3] = 255
  }
}

/** Continuous single-channel states in [0,1] → grayscale (default_to_rgb). */
export function continuousToRgb(
  frame: StateArray,
  _C: number,
  H: number,
  W: number,
  out: Uint8ClampedArray,
): void {
  for (let i = 0; i < H * W; i++) {
    const g = Math.round(Math.min(1, Math.max(0, frame[i])) * 255)
    const j = i * 4
    out[j] = g
    out[j + 1] = g
    out[j + 2] = g
    out[j + 3] = 255
  }
}

/** Gray-Scott to_rgb (systems.py:1834-1838): v channel (index 1) as grayscale. */
export function grayScottToRgb(
  frame: StateArray,
  _C: number,
  H: number,
  W: number,
  out: Uint8ClampedArray,
): void {
  const vOff = H * W // channel 1
  for (let i = 0; i < H * W; i++) {
    const g = Math.round(Math.min(1, Math.max(0, frame[vOff + i])) * 255)
    const j = i * 4
    out[j] = g
    out[j + 1] = g
    out[j + 2] = g
    out[j + 3] = 255
  }
}

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const h6 = h * 6
  const i = Math.floor(h6) % 6
  const f = h6 - Math.floor(h6)
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  switch (i) {
    case 0: return [v, t, p]
    case 1: return [q, v, p]
    case 2: return [p, v, t]
    case 3: return [p, q, v]
    case 4: return [t, p, v]
    default: return [v, p, q]
  }
}

/**
 * Langton palette (systems.py:2002-2022): state 0 black; K=2 → white;
 * K>2 → golden-ratio-spaced hues at s=0.85, v=1.
 */
export function makeLangtonPalette(K: number): Uint8ClampedArray {
  const pal = new Uint8ClampedArray(K * 3)
  if (K === 2) {
    pal[3] = pal[4] = pal[5] = 255
    return pal
  }
  for (let i = 1; i < K; i++) {
    const hue = (i * 0.6180339887498949) % 1
    const [r, g, b] = hsvToRgb(hue, 0.85, 1.0)
    pal[i * 3] = Math.round(r * 255)
    pal[i * 3 + 1] = Math.round(g * 255)
    pal[i * 3 + 2] = Math.round(b * 255)
  }
  return pal
}

export function makeLangtonToRgb(K: number): ToRgbFn {
  const pal = makeLangtonPalette(K)
  return (frame, _C, H, W, out) => {
    for (let i = 0; i < H * W; i++) {
      const s = Math.min(K - 1, Math.max(0, frame[i] | 0)) * 3
      const j = i * 4
      out[j] = pal[s]
      out[j + 1] = pal[s + 1]
      out[j + 2] = pal[s + 2]
      out[j + 3] = 255
    }
  }
}

/**
 * 1D rollout → space-time diagram: T rows tall, batches tiled horizontally
 * (rollout.py "1d_rollout" layout). Returns native-resolution ImageData.
 */
export function render1dSpaceTime(r: RolloutData): ImageData {
  const width = r.B * r.W
  const img = new ImageData(width, r.T)
  const row = new Uint8ClampedArray(r.W * 4)
  for (let b = 0; b < r.B; b++) {
    for (let t = 0; t < r.T; t++) {
      r.toRgb(frameView(r, b, t), r.C, 1, r.W, row)
      img.data.set(row, (t * width + b * r.W) * 4)
    }
  }
  return img
}

/** 2D rollout frame t (default last) → batches tiled horizontally. */
export function render2dFrame(r: RolloutData, t = r.T - 1): ImageData {
  const width = r.B * r.W
  const img = new ImageData(width, r.H)
  const tile = new Uint8ClampedArray(r.H * r.W * 4)
  for (let b = 0; b < r.B; b++) {
    r.toRgb(frameView(r, b, t), r.C, r.H, r.W, tile)
    for (let y = 0; y < r.H; y++) {
      const srcOff = y * r.W * 4
      img.data.set(tile.subarray(srcOff, srcOff + r.W * 4), (y * width + b * r.W) * 4)
    }
  }
  return img
}

/**
 * Draw ImageData to a canvas with nearest-neighbor upscale at an integer
 * factor chosen to fit maxWidth.
 */
export function drawScaled(canvas: HTMLCanvasElement, img: ImageData, maxWidth: number): void {
  const scale = Math.max(1, Math.floor(maxWidth / img.width))
  canvas.width = img.width * scale
  canvas.height = img.height * scale
  const off = new OffscreenCanvas(img.width, img.height)
  off.getContext('2d')!.putImageData(img, 0, 0)
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height)
}
