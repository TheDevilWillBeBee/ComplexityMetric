import { el } from './dom'

/** Compact viridis approximation via linear interpolation of anchor colors. */
const VIRIDIS: ReadonlyArray<readonly [number, number, number]> = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
]

export function viridis(t: number): string {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1)
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x))
  const f = x - i
  const a = VIRIDIS[i]
  const b = VIRIDIS[i + 1]
  const r = Math.round(a[0] + (b[0] - a[0]) * f)
  const g = Math.round(a[1] + (b[1] - a[1]) * f)
  const bl = Math.round(a[2] + (b[2] - a[2]) * f)
  return `rgb(${r},${g},${bl})`
}

/**
 * One trajectory scatter (Embedding.visualize parity): connected polyline
 * (alpha 0.35), points colored by time, circle at t=0, X at t=T-1.
 * `points` is (T, 2) row-major.
 */
export function trajectoryPlot(points: Float32Array, T: number, size = 260, title = ''): HTMLCanvasElement {
  const canvas = el('canvas', { width: size, height: size })
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#10141d'
  ctx.fillRect(0, 0, size, size)

  if (T === 0) return canvas

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (let t = 0; t < T; t++) {
    minX = Math.min(minX, points[t * 2])
    maxX = Math.max(maxX, points[t * 2])
    minY = Math.min(minY, points[t * 2 + 1])
    maxY = Math.max(maxY, points[t * 2 + 1])
  }
  const pad = 18
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1
  const px = (t: number) => pad + ((points[t * 2] - minX) / spanX) * (size - 2 * pad)
  const py = (t: number) => size - pad - ((points[t * 2 + 1] - minY) / spanY) * (size - 2 * pad)

  ctx.globalAlpha = 0.35
  ctx.strokeStyle = '#9aa4b5'
  ctx.beginPath()
  for (let t = 0; t < T; t++) {
    if (t === 0) ctx.moveTo(px(t), py(t))
    else ctx.lineTo(px(t), py(t))
  }
  ctx.stroke()
  ctx.globalAlpha = 1

  for (let t = 0; t < T; t++) {
    ctx.fillStyle = viridis(T === 1 ? 0 : t / (T - 1))
    ctx.beginPath()
    ctx.arc(px(t), py(t), 2.2, 0, Math.PI * 2)
    ctx.fill()
  }

  // start marker: circle outline; end marker: X
  ctx.strokeStyle = '#e6e9ef'
  ctx.lineWidth = 1.5
  ctx.beginPath()
  ctx.arc(px(0), py(0), 5.5, 0, Math.PI * 2)
  ctx.stroke()
  const ex = px(T - 1)
  const ey = py(T - 1)
  ctx.beginPath()
  ctx.moveTo(ex - 5, ey - 5)
  ctx.lineTo(ex + 5, ey + 5)
  ctx.moveTo(ex - 5, ey + 5)
  ctx.lineTo(ex + 5, ey - 5)
  ctx.stroke()

  if (title) {
    ctx.fillStyle = '#9aa4b5'
    ctx.font = '11px system-ui'
    ctx.fillText(title, 6, 13)
  }
  return canvas
}

export interface Series {
  label: string
  xs: number[]
  ys: number[]
}

const SERIES_COLORS = ['#ff4b4b', '#4bd1ff', '#7dff8a', '#ffd84b', '#c78bff', '#ff9d5c', '#6a8dff', '#ff6ad5']

/** Minimal multi-series line chart with legend (metric window means over time). */
export function lineChart(series: Series[], width = 720, height = 300): HTMLCanvasElement {
  const canvas = el('canvas', { width, height })
  const ctx = canvas.getContext('2d')!
  ctx.fillStyle = '#10141d'
  ctx.fillRect(0, 0, width, height)

  const all = series.filter((s) => s.xs.length > 0)
  if (all.length === 0) return canvas
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity
  for (const s of all)
    for (let i = 0; i < s.xs.length; i++) {
      minX = Math.min(minX, s.xs[i])
      maxX = Math.max(maxX, s.xs[i])
      minY = Math.min(minY, s.ys[i])
      maxY = Math.max(maxY, s.ys[i])
    }
  const padL = 48
  const padR = 12
  const padT = 12
  const padB = 28
  const spanX = maxX - minX || 1
  const spanY = maxY - minY || 1
  const px = (x: number) => padL + ((x - minX) / spanX) * (width - padL - padR)
  const py = (y: number) => height - padB - ((y - minY) / spanY) * (height - padT - padB)

  // axes + a few y labels
  ctx.strokeStyle = '#2c3444'
  ctx.fillStyle = '#9aa4b5'
  ctx.font = '11px system-ui'
  for (let i = 0; i <= 4; i++) {
    const y = minY + (spanY * i) / 4
    ctx.beginPath()
    ctx.moveTo(padL, py(y))
    ctx.lineTo(width - padR, py(y))
    ctx.stroke()
    ctx.fillText(y.toFixed(3), 4, py(y) + 4)
  }
  for (let i = 0; i <= 4; i++) {
    const x = minX + (spanX * i) / 4
    ctx.fillText(x.toFixed(0), px(x) - 8, height - 8)
  }

  series.forEach((s, si) => {
    const color = SERIES_COLORS[si % SERIES_COLORS.length]
    ctx.strokeStyle = color
    ctx.lineWidth = 1.6
    ctx.beginPath()
    for (let i = 0; i < s.xs.length; i++) {
      if (i === 0) ctx.moveTo(px(s.xs[i]), py(s.ys[i]))
      else ctx.lineTo(px(s.xs[i]), py(s.ys[i]))
    }
    ctx.stroke()
    // legend
    ctx.fillStyle = color
    ctx.fillRect(padL + 8 + si * 170, padT + 2, 10, 10)
    ctx.fillStyle = '#e6e9ef'
    ctx.fillText(s.label.slice(0, 22), padL + 22 + si * 170, padT + 11)
  })
  return canvas
}
